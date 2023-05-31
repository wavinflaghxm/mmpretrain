# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from typing import List, Tuple, Union, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from mmcv.cnn.bricks import build_norm_layer
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from mmpretrain.models import ImageClassifier, VisionTransformer
from mmpretrain.models.backbones.base_backbone import BaseBackbone

from .transformer import TextTransformer, AttentionPool2d


@MODELS.register_module()
class CLIP(BaseBackbone):

    VISUAL_OUT_TYPES = {'raw', 'cls_token', 'patch_token'}

    def __init__(self,
                 visual: dict,
                 text: Optional[dict] = None,
                 visual_proj: bool = True,
                 output_dims: int = 512,
                 out_type: str = 'cls_token',
                 gap_cls_token: bool = False,
                 attn_pool: bool = False,
                 attn_pool_heads: int = 8,
                 attn_pool_queries: int = 256,
                 frozen_visual_stages: int = -1,
                 frozen_text_stages: int = -1,
                 init_cfg: Optional[dict] = None) -> None:
        super(CLIP, self).__init__(init_cfg=init_cfg)

        # build image tower
        visual_cfg = copy.deepcopy(visual)
        visual_cfg = self._update_cfg(visual_cfg, {'out_type': 'raw', 'final_norm': False})
        self.visual = MODELS.build(visual_cfg)
        # build visual pooling and projection
        if out_type not in self.VISUAL_OUT_TYPES:
            raise ValueError(f'Unsupported `out_type` {out_type}, please '
                             f'choose from {self.VISUAL_OUT_TYPES}')
        self.out_type = out_type
        self.gap_cls_token = gap_cls_token

        norm_cfg = visual_cfg.get('norm_cfg', dict(type='LN'))
        embed_dims = self.visual.embed_dims
        if attn_pool:
            dims = output_dims
            self.visual_attn_pool = AttentionPool2d(output_dims, embed_dims,
                                                    num_heads=attn_pool_heads,
                                                    num_queries=attn_pool_queries)
        else:
            dims = embed_dims
            self.visual_attn_pool = None

        self.visual_final_norm = build_norm_layer(norm_cfg, dims)[1]
        if visual_proj:
            scale = embed_dims ** -0.5
            self.visual_projection = nn.Parameter(scale * torch.randn(dims, output_dims))
        else:
            self.visual_projection = None
        self.freeze_image_tower(frozen_visual_stages)

        # build text tower
        if text is not None:
            text_cfg = copy.deepcopy(text)
            text_cfg = self._update_cfg(text_cfg, {'output_dims': output_dims})
            t = MODELS.build(text_cfg)
            assert isinstance(t, TextTransformer)
            self.transformer = t.transformer
            self.context_length = t.context_length
            self.vocab_size = t.vocab_size
            self.token_embedding = t.token_embedding
            self.positional_embedding = t.positional_embedding
            self.ln_final = t.ln_final
            self.text_projection = t.text_projection
            self.register_buffer('attn_mask', t.attn_mask, persistent=False)      
            self.freeze_text_tower(frozen_text_stages)

        # logit_scale
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def _update_cfg(self, original: dict, updated: dict) -> dict:
        for key, value in updated.items():
            current = original.get(key, None)
            if current is None:
                original.update({key: value})
            else:
                if current != value:
                    warnings.warn(
                        f'The value of {key} should be {value}, but it is currently '
                        f'set to {current}. Please update it in your config file.')
                    original.update({key: value})
        return original

    def _global_pool(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if self.gap_cls_token:
            return x.mean(dim=1), x
        else:
            return x[:, 0], x[:, 1:]

    def freeze_image_tower(self, frozen_stages: int):
        """Freeze the parameters in the image tower.
        
        Args: 
            frozen_stages (int):  Stages to be frozen (stop grad and set eval mode).
                -1 means not freezing any parameters, 0 means freezing tower, and
                 1 means freezing tower and projection.
        """
        if frozen_stages >= 0:
            # freeze image tower
            if isinstance(self.visual, VisionTransformer):
                self.visual.frozen_stages = self.visual.num_layers
                self.visual._freeze_stages()
            # freeze extra modules
            modules = ['visual_attn_pool', 'visual_final_norm']
            for module in modules:
                m = getattr(self, module)
                if m is not None:
                    m.eval()
                    for param in m.parameters():
                        param.requires_grad = False
        if frozen_stages >= 1 and self.visual_projection is not None:
            # freeze visual projection
            self.visual_projection.requires_grad = False

    def freeze_text_tower(self, frozen_stages: int):
        if frozen_stages >= 0:
            # freeze token embedding
            for param in self.token_embedding.parameters():
                param.requires_grad = False
            # freeze position embedding
            self.positional_embedding.requires_grad = False
            # freeze layers
            self.transformer.eval()
            for param in self.transformer.parameters():
                param.requires_grad = False
            # freeze ln_final
            self.ln_final.eval()
            for param in self.ln_final.parameters():
                param.requires_grad = False
        if frozen_stages >= 1:
            # freeze text projection
            self.text_projection.requires_grad = False

    def encode_image(self, image: Tensor):
        outs = []
        feats = self.visual(image)
        for x in feats:
            if self.visual_attn_pool is not None:
                x = self.visual_attn_pool(x)

            x = self.visual_final_norm(x)
            pooled, tokens = self._global_pool(x)

            if self.visual_projection is not None:
                pooled = pooled @ self.visual_projection

            if self.out_type == 'cls_token':
                outs.append(pooled)
            elif self.out_type == 'patch_token':
                outs.append(tokens)
            elif self.out_type == 'raw':
                outs.append(x)
        return tuple(outs)

    def encode_text(self, text: Tensor):
        # ensure that the text tower is built
        assert hasattr(self, 'transformer')
        # text tower forward
        B, N, ctx = text.shape
        text = text.reshape(-1, ctx)
        cast_dtype = self.transformer.get_cast_dtype()
        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x.reshape(B, N, -1)

    def extract_feat(self, inputs: List[Tensor]):
        image, text = inputs[0], inputs[1]
        image = self.encode_image(image) if image is not None else None
        text = self.encode_text(text) if text is not None else None
        return image, text, self.logit_scale.exp()

    def forward(self, inputs: Union[Tensor, List[Tensor]], 
                data_samples: List[DataSample]):
        if not isinstance(inputs, list):
            assert isinstance(inputs, Tensor)
            inputs = [inputs, None]
        return self.extract_feat(inputs)


@MODELS.register_module()
class CLIPClassifier(ImageClassifier):

    def extract_feat(self, inputs: Tensor, data_samples: List[DataSample]):
        x = self.backbone(inputs, data_samples)
        return self.neck(x) if self.with_neck else x

    def loss(self, inputs: List[Tensor],
             data_samples: List[DataSample]) -> dict:
        """The forward function in training.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample]): The annotation data of
                every samples.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        feats = self.extract_feat(inputs, data_samples)
        return self.head.loss(feats, data_samples)

    def predict(self,
                inputs: List[Tensor],
                data_samples: Optional[List[DataSample]] = None,
                **kwargs) -> List[DataSample]:
        """Predict results from a batch of inputs.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample], optional): The annotation
                data of every samples. Defaults to None.
            **kwargs: Other keyword arguments accepted by the ``predict``
                method of :attr:`head`.
        """
        feats = self.extract_feat(inputs, data_samples)
        return self.head.predict(feats, data_samples, **kwargs)
