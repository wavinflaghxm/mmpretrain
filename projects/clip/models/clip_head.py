# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from mmengine.dist import get_dist_info
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from mmpretrain.evaluation.metrics import Accuracy

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(image_feats,
                    text_feats,
                    local_loss=False,
                    gather_with_grad=False,
                    rank=0,
                    world_size=1,
                    use_horovod=False):
    assert has_distributed, 'torch.distributed did not import correctly, ' \
                            'please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_feats = hvd.allgather(image_feats)
            all_text_feats = hvd.allgather(text_feats)
        else:
            with torch.no_grad():
                all_image_feats = hvd.allgather(image_feats)
                all_text_feats = hvd.allgather(text_feats)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_feats = list(all_image_feats.chunk(world_size, dim=0))
                gathered_text_feats = list(all_text_feats.chunk(world_size, dim=0))
                gathered_image_feats[rank] = image_feats
                gathered_text_feats[rank] = text_feats
                all_image_feats = torch.cat(gathered_image_feats, dim=0)
                all_text_feats = torch.cat(gathered_text_feats, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_feats = torch.cat(torch.distributed.nn.all_gather(image_feats), dim=0)
            all_text_feats = torch.cat(torch.distributed.nn.all_gather(text_feats), dim=0)
        else:
            gathered_image_feats = [torch.zeros_like(image_feats) for _ in range(world_size)]
            gathered_text_feats = [torch.zeros_like(text_feats) for _ in range(world_size)]
            dist.all_gather(gathered_image_feats, image_feats)
            dist.all_gather(gathered_text_feats, text_feats)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_feats[rank] = image_feats
                gathered_text_feats[rank] = text_feats
            all_image_feats = torch.cat(gathered_image_feats, dim=0)
            all_text_feats = torch.cat(gathered_text_feats, dim=0)

    return all_image_feats, all_text_feats


@MODELS.register_module(force=True)
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
                 loss: dict = dict(type='CrossEntropyLoss', loss_weight=1.0),
                 pred_mode: str = 'classifier',
                 topk: Union[int, Tuple[int]] = (1, ),
                 cal_acc: bool = False,
                 local_loss: bool = False,
                 gather_with_grad: bool = False,
                 cache_labels: bool = False,
                 use_horovod: bool = False,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)

        if not isinstance(loss, nn.Module):
            loss = MODELS.build(loss)
        self.loss_module = loss
        self.pred_mode = pred_mode
        self.topk = topk
        self.cal_acc = cal_acc

        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_feats, text_feats, logit_scale):
        if self.world_size > 1:
            all_image_feats, all_text_feats = gather_features(
                image_feats, text_feats,
                self.local_loss, self.gather_with_grad, self.rank, 
                self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_feats @ all_text_feats.t()
                logits_per_text = logit_scale * text_feats @ all_image_feats.t()
            else:
                logits_per_image = logit_scale * all_image_feats @ all_text_feats.t()
                logits_per_text = logits_per_image.t()
        else:
            logits_per_image = logit_scale * image_feats @ text_feats.t()
            logits_per_text = logit_scale * text_feats @ image_feats.t()
        
        return logits_per_image, logits_per_text

    def pre_logits(self, 
                   image_feats: Tensor, 
                   text_feats: Tensor,
                   logit_scale: Tensor, 
                   mode: str) -> Union[Tensor, List[Tensor]]:
        if mode == 'classifier':
            return logit_scale * image_feats @ text_feats.t()
        elif mode == 'contrastive':
            logits_per_image, logits_per_text = self.get_logits(
                image_feats, text_feats, logit_scale)
            return logits_per_image, logits_per_text
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def forward(self, images, texts, logit_scale, mode='classifier'):
        """The forward process."""
        pre_logits = self.pre_logits(images, texts, logit_scale, mode)
        return pre_logits

    def loss(self, images, texts, logit_scale, data_samples, **kwargs):
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
        cls_score = self(images, texts, logit_scale, mode=self.pred_mode)

        # The part can not be traced by torch.fx
        losses = self._get_loss(cls_score, data_samples, **kwargs)
        return losses

    def _get_loss(self, cls_score, data_samples, **kwargs):
        """compute loss."""
        losses = dict()
        if self.pred_mode == 'classifier':
            if 'gt_score' in data_samples[0]:
                # Batch augmentation may convert labels to one-hot format scores.
                target = torch.stack([i.gt_score for i in data_samples])
            else:
                target = torch.cat([i.gt_label for i in data_samples])
            loss = self.loss_module(
                cls_score, target, avg_factor=cls_score.size(0), **kwargs)
        elif self.pred_mode == 'contrastive':
            assert isinstance(cls_score, tuple)
            logits_per_image, logits_per_text = cls_score
            target = self.get_ground_truth(
                logits_per_image.device, logits_per_image.shape[0])
            loss_per_image = self.loss_module(
                logits_per_image, target, reduction_override='sum', **kwargs)
            loss_per_text = self.loss_module(
                logits_per_text, target, reduction_override='sum', **kwargs)
            loss = (loss_per_image + loss_per_text) / 2
            cls_score = logits_per_image
        else:
            raise RuntimeError(f'Invalid mode "{self.pred_mode}".')
        
        losses['loss'] = loss

        # compute accuracy
        if self.cal_acc:
            assert target.ndim == 1, 'If you enable batch augmentation ' \
                'like mixup during training, `cal_acc` is pointless.'
            acc = Accuracy.calculate(cls_score, target, topk=self.topk)
            losses.update(
                {f'accuracy_top-{k}': a
                 for k, a in zip(self.topk, acc)})

        return losses

    def predict(self, images, texts, logit_scale, 
                data_samples=None) -> List[DataSample]:
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
        cls_score = self(images, texts, logit_scale, mode='classifier')

        # The part can not be traced by torch.fx
        predictions = self._get_predictions(cls_score, data_samples)
        return predictions

    def _get_predictions(self, cls_score, data_samples):
        """Post-process the output of head.

        Including softmax and set ``pred_label`` of data samples.
        """
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


import numpy as np
from mmpretrain.models import ClsHead

@MODELS.register_module(force=True)  # avoid bug
class ZeroShotClsHead(ClsHead):

    def __init__(self,
                 num_classes: int,
                 zs_weight_path: str,
                 zs_weight_dims: int = 512,
                 init_cfg: Optional[dict] = None,
                 **kwargs) -> None:
        super(ZeroShotClsHead, self).__init__(init_cfg=init_cfg, **kwargs)
        if zs_weight_path == 'rand':
            zs_weight = torch.randn((num_classes, zs_weight_dims))
            nn.init.normal_(zs_weight, std=0.01)
        else:
            zs_weight = torch.tensor(np.load(zs_weight_path),
                                     dtype=torch.float32)
        zs_weight = F.normalize(zs_weight, p=2, dim=-1)

        if zs_weight_path == 'rand':
            self.zs_weight = nn.Parameter(zs_weight)
        else:
            self.register_buffer('zs_weight', zs_weight)

        assert self.zs_weight.shape == (num_classes, zs_weight_dims)
    
    def pre_logits(self, image_feats, text_feats, logit_scale):
        """The process before the final classification head."""
        if text_feats is None:
            assert hasattr(self, 'zs_weight')
            text_feats = self.zs_weight
        return logit_scale * image_feats @ text_feats.T
    
    def forward(self, images, texts, logit_scale):
        """The forward process."""
        pre_logits = self.pre_logits(images, texts, logit_scale)
        return pre_logits

    def loss(self, images, texts, logit_scale, data_samples, **kwargs):
        # The part can be traced by torch.fx
        cls_score = self(images, texts, logit_scale)

        # The part can not be traced by torch.fx
        losses = self._get_loss(cls_score, data_samples, **kwargs)
        return losses
    
    def predict(self, images, texts, logit_scale, 
                data_samples=None) -> List[DataSample]:
        # The part can be traced by torch.fx
        cls_score = self(images, texts, logit_scale)

        # The part can not be traced by torch.fx
        predictions = self._get_predictions(cls_score, data_samples)
        return predictions
