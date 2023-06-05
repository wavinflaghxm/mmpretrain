# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from mmpretrain.registry import MODELS, TOKENIZER
from mmpretrain.structures import DataSample
from mmpretrain.utils import track_on_main_process
from mmpretrain.models import BaseClassifier, ModifiedResNet

from ..datasets.categories import OBJECTS365V2_CATEGORIES
from ..datasets.metadata import OPENAI_PROMPT, SIMPLE_PROMPT


PROTOTYPE_MAP = {'objects365v2': OBJECTS365V2_CATEGORIES}
PROMPT_MAP = {
    'openai': OPENAI_PROMPT,
    'simple': SIMPLE_PROMPT,
}


@MODELS.register_module(force=True)
class CLIP(BaseClassifier):
    """The implementation of `CLIP`_.

    Args:
        vision_backbone (dict): Config dict for vision backbone.
        text_backbone (dict): Config dict for text backbone.
        tokenizer (dict): Config dict for text tokenizer.
        proj_dims (int): Projection dimension for similarity computation.
        neck (dict, optional): The neck module to process features from
            backbone. Defaults to None.
        head (dict, optional): The head module to do prediction and calculate
            loss from processed features. Defaults to None.
        pretrained (str, optional): The pretrained checkpoint path, support
            local path and remote path. Defaults to None.
        text_prototype (str): Text prototype, which can be a key in
            `PROTOTYPE_MAP` or list of text.
        text_prompt (str): The prompt for text prototype. Defaults to 'openai'.
        context_length (int): The context length to use. Defaults to 52.
        data_preprocessor (Union[dict, nn.Module], optional): The config for
            preprocessing input data. If None or no specified type, it will use
            "MultiModalDataPreprocessor" as type.
            See :class:`MultiModalDataPreprocessor` for more details.
            Defaults to None.
        init_cfg (dict, optional): The config to control the initialization.
            Defaults to None.
    """

    def __init__(self,
                 vision_backbone: dict,
                 text_backbone: Optional[dict] = None,
                 neck: Optional[dict] = None,
                 head: Optional[dict] = None,
                 proj_dims: int = 512,
                 tokenizer: Optional[dict] = None,
                 context_length: int = 77,
                 text_prototype_path: str = '',
                 text_prototype: Union[str, List[str]] = 'none',
                 text_prompt: str = 'openai',
                 frozen_vision_stages: int = -1,
                 frozen_text_stages: int = -1,
                 pretrained: Optional[str] = None,
                 train_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None) -> None:
        if pretrained is not None:
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        data_preprocessor = data_preprocessor or {}

        if isinstance(data_preprocessor, dict):
            data_preprocessor.setdefault('type', 'ClsDataPreprocessor')
            data_preprocessor.setdefault('batch_augments', train_cfg)
            data_preprocessor = MODELS.build(data_preprocessor)
        elif not isinstance(data_preprocessor, nn.Module):
            raise TypeError('data_preprocessor should be a `dict` or '
                            f'`nn.Module` instance, but got '
                            f'{type(data_preprocessor)}')

        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        if not isinstance(vision_backbone, nn.Module):
            vision_backbone = MODELS.build(vision_backbone)
        if text_backbone is not None and not isinstance(text_backbone, nn.Module):
            text_backbone = MODELS.build(text_backbone)
        if neck is not None and not isinstance(neck, nn.Module):
            neck = MODELS.build(neck)
        if head is not None and not isinstance(head, nn.Module):
            head = MODELS.build(head)
        if tokenizer is not None and isinstance(tokenizer, dict):
            tokenizer = TOKENIZER.build(tokenizer)

        self.vision_backbone = vision_backbone
        self.text_backbone = text_backbone
        self.neck = neck
        self.head = head
        self.tokenizer = tokenizer

        # for zero-shot classification
        if text_prototype_path != '':
            embeds = torch.tensor(
                np.load(text_prototype_path), dtype=torch.float32)
            embeds = F.normalize(embeds, p=2, dim=-1)
            self.register_buffer('text_prototype_embeds', embeds)
        else:
            assert text_backbone is not None and tokenizer is not None
            self.context_length = context_length
            if isinstance(text_prototype,
                          str) and text_prototype in PROTOTYPE_MAP.keys():
                self.prototype = PROTOTYPE_MAP[text_prototype]
            else:
                assert text_prototype != 'none'
                self.prototype = text_prototype
            self.text_prototype_embeds = None

        self.prompt = PROMPT_MAP[text_prompt]

        if not isinstance(self.vision_backbone, ModifiedResNet) and \
                self.vision_backbone.out_type == 'cls_token':
            self.vision_projection = nn.Parameter(
                torch.empty(self.vision_backbone.embed_dims, proj_dims))
        if self.text_prototype_embeds is None:
            self.text_projection = nn.Parameter(
                torch.empty(self.text_backbone.width, proj_dims))

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # freeze stages only when self.frozen_stages > 0
        self.frozen_vision_stages = frozen_vision_stages
        self.frozen_text_stages = frozen_text_stages
        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze the parameters (stop grad and set eval mode) according to
           self.frozen_stages. 0 means not freezing any parameters, 1 means 
           freezing backbone, and 2 means freezing backbone and projection.
        """
        if self.frozen_vision_stages > 0:
            self.vision_backbone.eval()
            for param in self.vision_backbone.parameters():
                param.requires_grad = False
        if self.frozen_vision_stages > 1 and \
                hasattr(self, 'vision_projection'):
            self.vision_projection.requires_grad = False

        if self.frozen_text_stages > 0 and \
                self.text_backbone is not None:
            self.text_backbone.eval()
            for param in self.text_backbone.parameters():
                param.requires_grad = False
        if self.frozen_text_stages > 1 and \
                hasattr(self, 'text_projection'):
            self.text_projection.requires_grad = False

    def forward(self,
                inputs: Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'predict',
                **kwargs):
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor(s) without any
          post-processing, same as a common PyTorch Module.
        - "predict": Forward and return the predictions, which are fully
          processed to a list of :obj:`DataSample`.
        - "loss": Forward and return a dict of losses according to the given
          inputs and data samples.

        Args:
            inputs (Tensor): the preprocessed image tensor of shape
                ``(N, C, H, W)``.
            data_samples (List[DataSample], optional): The annotation data
                of every samples. Defaults to None.
            mode (str): Return what kind of value. Defaults to 'predict'.
        """
        if mode == 'tensor':
            feats = self.extract_feat(inputs, data_samples)
            return self.head(*feats) if self.with_head else feats
        elif mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def extract_feat(self, images: Tensor, data_samples: Optional[list] = None):
        image_feats = self.extract_image_feat(images)
        text_feats = self.prepare_text_feat(images.device, data_samples)
        logit_scale = self.logit_scale.exp()
        return image_feats, text_feats, logit_scale

    def extract_image_feat(self, images: Tensor) -> Tensor:
        """The function to extract image latent features."""
        if not hasattr(self, 'vision_projection'):
            image_feats = self.vision_backbone(images)
            if isinstance(image_feats, tuple):
                image_feats = image_feats[-1]
        else:
            image_feats = self.vision_backbone(images)[-1] @ self.vision_projection
        image_feats = F.normalize(image_feats, p=2, dim=-1)
        return image_feats

    def extract_text_feat(self, texts: Tensor) -> Tensor:
        """The function to extract text latent features."""
        text_feats = self.text_backbone(texts) @ self.text_projection
        text_feats = F.normalize(text_feats, p=2, dim=-1)
        return text_feats

    def prepare_text_feat(self, device, data_samples: Optional[list] = None):
        sample_item = data_samples[0] if data_samples is not None else None
        if sample_item is None or sample_item.get('text', None) is None:
            if self.text_prototype_embeds is None:
                self.prepare_text_prototype(device=device)
            return self.text_prototype_embeds
        else:
            texts = [ds.text for ds in data_samples]
            texts = self.tokenize(texts).to(device)
            text_feats = self.extract_text_feat(texts)
            return text_feats

    def prepare_text_prototype(self, device) -> None:
        """The function to prepare text prototypes with prompt."""
        class_embeds = []
        for classname in track_on_main_process(self.prototype,
                                               'Prepare text prototype...'):
            # format with class
            texts = [prompt(classname) for prompt in self.prompt]
            tokenized_texts = self.tokenize(texts)
            class_feats = self.extract_text_feat(tokenized_texts.to(device))
            class_feat = F.normalize(class_feats.mean(dim=0), p=2, dim=-1)
            class_embeds.append(class_feat.detach().clone())
        self.text_prototype_embeds = torch.stack(class_embeds).to(device)

    def tokenize(self, texts: Union[str, List[str]]) -> torch.LongTensor:
        """Returns the tokenized representation of given input string(s)

        Args:
            texts (Union[str, List[str]]): An input string or a list of input
                strings to tokenize
            context_length (int): The context length to use. Defaults to 52.

        Returns:
            Tensor: Resulting tokens.
        """
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.tokenizer.encoder["<start_of_text>"]
        eot_token = self.tokenizer.encoder["<end_of_text>"]
        all_tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts]
        result = torch.zeros(len(all_tokens), self.context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > self.context_length:
                tokens = tokens[:self.context_length]  # Truncate
                tokens[-1] = eot_token
            result[i, :len(tokens)] = torch.tensor(tokens)
        return result
    
    def loss(self, 
             images: Tensor,
             data_samples: List[DataSample]) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            images (Tensor): The input images.
            data_samples (List[DataSample]): The annotation data of
                every samples.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        feats = self.extract_feat(images, data_samples)
        return self.head.loss(*feats, data_samples)
    
    def predict(self,
                images: Tensor,
                data_samples: Optional[List[DataSample]] = None,
                **kwargs) -> List[DataSample]:
        """Predict results from a batch of inputs.

        Args:
            images (Tensor): The input images.
            data_samples (List[DataSample], optional): The annotation
                data of every samples. Defaults to None.
            **kwargs: Other keyword arguments accepted by the ``predict``
                method of :attr:`head`.
        """
        feats = self.extract_feat(images)
        return self.head.predict(*feats, data_samples, **kwargs)
