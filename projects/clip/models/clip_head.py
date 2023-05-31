# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample


@MODELS.register_module()
class CLIPClsHead(BaseModule):
    """CLIP Classification head.

    Args:
        loss (dict): Config of classification loss. Defaults to
            ``dict(type='CLIPLoss', loss_weight=1.0)``.
        topk (int | Tuple[int]): Top-k accuracy. Defaults to ``(1, )``.
        cal_acc (bool): Whether to calculate accuracy during training.
            If you use batch augmentations like Mixup and CutMix during
            training, it is pointless to calculate accuracy.
            Defaults to False.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to None.
    """

    def __init__(self,
                 loss: dict = dict(type='CLIPLoss', loss_weight=1.0),
                 topk: Union[int, Tuple[int]] = (1, ),
                 cal_acc: bool = False,
                 text_weight_path: str = '',
                 init_cfg: Optional[dict] = None):
        super(CLIPClsHead, self).__init__(init_cfg=init_cfg)

        self.topk = topk
        if not isinstance(loss, nn.Module):
            loss = MODELS.build(loss)
        self.loss_module = loss
        self.cal_acc = cal_acc

        if text_weight_path != '':
            text_weight = torch.tensor(
                np.load(text_weight_path), dtype=torch.float32)  # D x C
            
            text_weight = F.normalize(text_weight, p=2, dim=-1)
            self.register_buffer('text_weight', text_weight)

    def pre_logits(self, feats, data_samples):
        """The process before the final classification head."""
        image, text, logit_scale = feats

        image = image[-1]  # Obtain feature of the last scale.
        # For backward-compatibility with the previous ViT output
        image = image[-1] if isinstance(image, list) else image  # cls token
        image = F.normalize(image, dim=-1)

        if text is not None:
            text = F.normalize(text.mean(dim=1), dim=-1)
        else:          
            if 'gt_score' in data_samples[0]:
                # Batch augmentation may convert labels to one-hot format scores.
                target = torch.stack([i.gt_score for i in data_samples])
            else:
                target = torch.cat([i.gt_label for i in data_samples])

            assert hasattr(self, 'text_weight')
            text = self.text_weight[target]

        return image, text, logit_scale

    def forward(self, feats, data_samples):
        """The forward process."""
        pre_logits = self.pre_logits(feats, data_samples)
        return pre_logits

    def loss(self, feats, data_samples, **kwargs):
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # The part can be traced by torch.fx
        preds = self(feats, data_samples)

        # The part can not be traced by torch.fx
        losses = self._get_loss(preds, data_samples, **kwargs)
        return losses

    def _get_loss(self, preds, data_samples, **kwargs):
        """compute loss."""
        image, text, logit_scale = preds
        
        losses = dict()
        loss, acc = self.loss_module(image, text, logit_scale,
                                     cal_acc=self.cal_acc, **kwargs)
        losses['loss'] = loss

        if acc is not None:
            losses.update(
                {f'accuracy_top-{k}': a
                 for k, a in zip(self.topk, acc)})
        return losses

    def predict(self, feats, data_samples=None) -> List[DataSample]:
        """Inference without augmentation.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample | None], optional): The annotation
                data of every samples. If not None, set ``pred_label`` of
                the input data samples. Defaults to None.

        Returns:
            List[DataSample]: A list of data samples which contains the
            predicted results.
        """
        # The part can be traced by torch.fx
        preds = self(feats, data_samples)

        # The part can not be traced by torch.fx
        predictions = self._get_predictions(preds, data_samples)
        return predictions

    def _get_predictions(self, preds, data_samples):
        """Post-process the output of head.

        Including softmax and set ``pred_label`` of data samples.
        """
        image, text, _ = preds
        cls_score = image @ text.T
        pred_scores = F.softmax(cls_score, dim=1)
        pred_labels = pred_scores.argmax(dim=1, keepdim=True).detach()

        out_data_samples = []
        if data_samples is None:
            data_samples = [None for _ in range(pred_scores.size(0))]

        for data_sample, score, label in zip(data_samples, pred_scores,
                                             pred_labels):
            if data_sample is None:
                data_sample = DataSample()

            data_sample.set_pred_score(score).set_pred_label(label)
            out_data_samples.append(data_sample)
        return out_data_samples
