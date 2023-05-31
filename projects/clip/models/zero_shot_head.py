# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mmpretrain.registry import MODELS
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

    def pre_logits(self, feats):
        """The process before the final classification head."""
        image_feats, text_feats, logit_scale = feats

        image_feats = image_feats[-1]  # Obtain feature of the last scale.
        # For backward-compatibility with the previous ViT output
        if isinstance(image_feats, list):
            image_feats = image_feats[-1]
        image_feats = F.normalize(image_feats, dim=-1)

        if text_feats is not None:
            text_feats = F.normalize(text_feats.mean(dim=1), dim=-1)
        else:
            assert hasattr(self, 'zs_weight')
            text_feats = self.zs_weight

        x = logit_scale * image_feats @ text_feats.T

        return x
