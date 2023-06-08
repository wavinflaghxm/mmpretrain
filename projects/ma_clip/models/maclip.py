# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch import Tensor
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmengine.model import BaseModule, ModuleList, Sequential
from mmengine.model.weight_init import trunc_normal_

from mmpretrain.registry import MODELS
from mmpretrain.models.utils import (LayerNorm2d, build_2d_sincos_position_embedding, 
                                     to_2tuple)
from mmpretrain.models.backbones.vision_transformer import TransformerEncoderLayer

from projects.clip.models import CLIP


@MODELS.register_module()
class MACLIP(CLIP):

    def init_weights(self) -> None:
        """Initialize logit scale."""
        super().init_weights()
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))

    def extract_image_feat(self, images: Tensor) -> Tensor:
        """The function to extract image latent features."""
        images, masks = torch.split(images, [images.shape[1] - 1, 1], dim=1)
        image_feats = super().extract_image_feat(images)
        if self.with_neck:
            image_feats = self.neck(image_feats, masks)
        return image_feats


@MODELS.register_module(force=True)
class MACLIPNeck(BaseModule):

    def __init__(self,
                 encoder: dict,
                 decoder: dict,
                 init_cfg: dict = None) -> None:
        super().__init__(init_cfg)
        self.encoder = MODELS.build(encoder)
        self.decoder = MODELS.build(decoder)

    def forward(self, images: Tensor, masks: Tensor) -> Tensor:
        masks = self.encoder(masks)
        x = self.decoder(images, masks)
        return x


@MODELS.register_module(force=True)
class MACLIPMaskEncoder(BaseModule):
    """Encodes masks for input to MA-CLIP's mask decoder.

    Args:
        embed_dims (int): The prompts' embedding dimension.
        hidden_dims (int): The number of hidden channels used for 
            encoding input masks.
        act_cfg (dict): The activation to use when encoding input masks.
    """

    def __init__(self,
                 embed_dims: int,
                 hidden_dims: int,
                 down_stride: int = 16,
                 act_cfg: dict = dict(type='GELU'),
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.hidden_dims = hidden_dims

        down_stride = int(np.log2(down_stride))
        self.fore_layers = self._make_layer(down_stride, act_cfg)
        self.back_layers = self._make_layer(down_stride, act_cfg)
    
    def _make_layer(self, stride: int, act_cfg: dict):
        layers = []
        last_dims = 1
        for i in range(stride - 1, -1, -1):
            dims = self.hidden_dims // 4**i
            layers.extend([
                nn.Conv2d(last_dims, dims, kernel_size=2, stride=2),
                LayerNorm2d(dims),
                build_activation_layer(act_cfg)])
            last_dims = dims
        layers.append(nn.Conv2d(self.hidden_dims, self.embed_dims, kernel_size=1))
        return Sequential(*layers)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """The forward function."""
        return self.fore_layers(x), self.back_layers(1 - x)


@MODELS.register_module(force=True)
class MACLIPMaskDecoder(BaseModule):
    """Maskd ecoder using a tranformer architecture."""
    num_extra_tokens = 1  # cls_token
    OUT_TYPES = {'raw', 'cls_token', 'featmap'}

    def __init__(self,
                 embed_dims: int,
                 hidden_dims: int = 512,
                 output_dims: int = 0,
                 img_size: Union[int, tuple] = 224,
                 patch_size: int = 16,
                 num_layers: int = 8,
                 num_heads: int = 16,
                 mlp_ratio: int = 4,
                 out_type: str = 'cls_token',
                 with_cls_token: bool = True,
                 norm_cfg: dict = dict(type='LN', eps=1e-6),
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        img_size = to_2tuple(img_size)
        self.patch_resolution = (img_size[1] // patch_size, img_size[0] // patch_size)
        self.num_patches = self.patch_resolution[0] * self.patch_resolution[1]
        self.output_dims = output_dims

        # Set out type
        if out_type not in self.OUT_TYPES:
            raise ValueError(f'Unsupported `out_type` {out_type}, please '
                             f'choose from {self.OUT_TYPES}')
        self.out_type = out_type

        # used to convert the dim of features from encoder to the dim
        # compatible with that of decoder
        self.patch_embed = nn.Linear(embed_dims, hidden_dims, bias=True)

        # Set cls token
        if with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        else:
            self.cls_token = None

        # create new position embedding, different from that in encoder
        # and is not learnable
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + self.num_extra_tokens, hidden_dims),
            requires_grad=False)

        self.layers = ModuleList([
            TransformerEncoderLayer(
                hidden_dims,
                num_heads,
                int(hidden_dims * mlp_ratio),
                qkv_bias=True,
                norm_cfg=norm_cfg) for _ in range(num_layers)])

        self.ln = build_norm_layer(norm_cfg, hidden_dims)[1]

        if self.output_dims > 0:
            self.projection = nn.Linear(hidden_dims, output_dims)

    def init_weights(self) -> None:
        """Initialize position embedding."""
        super().init_weights()

        pos_embed = build_2d_sincos_position_embedding(
            int(self.num_patches**.5),
            self.pos_embed.shape[-1],
            cls_token=True)
        self.pos_embed.data.copy_(pos_embed.float())

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress custom init if use pretrained model.
            return

        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)

    @property
    def norm(self):
        return self.ln
    
    def _format_input(self, images: Tensor, 
                      masks: Tuple[Tensor, Tensor]) -> Tensor:
        B, L, C = images.shape
        mask_fore, mask_back = masks[0], masks[1]

        cls_token, patches = torch.split(images, [1, L - 1], dim=1)
        # reshape mask (B, C, H, W) -> (B, H, W, C) -> (B, N, C)
        x_fore = patches + mask_fore.permute(0, 2, 3,
                                             1).reshape(B, L - 1, C)
        x_back = patches + mask_back.permute(0, 2, 3,
                                             1).reshape(B, L - 1, C)
        x = torch.cat((x_fore, x_back), dim=-1)
        # set cls token
        x_cls = cls_token.repeat(1, 1, 2)
        if self.cls_token is not None:
            x_cls = x_cls + self.cls_token.expand(B, -1, -1)
        x = torch.cat((x_cls, x), dim=1)
        return x

    def _format_output(self, x: Tensor, hw: Tuple[int, int]) -> Tensor:
        if self.out_type == 'raw':
            return x
        if self.out_type == 'cls_token':
            return x[:, 0]

        patch_token = x[:, self.num_extra_tokens:]
        if self.out_type == 'featmap':
            B = x.size(0)
            # (B, N, C) -> (B, H, W, C) -> (B, C, H, W)
            return patch_token.reshape(B, *hw, -1).permute(0, 3, 1, 2)

    def forward(self, images: Tensor, masks: Tuple[Tensor, Tensor]) -> Tensor:
        """The forward function."""
        x = self._format_input(images, masks)
        # embed patch tokens
        x = self.patch_embed(x)

        # add pos embed
        x = x + self.pos_embed

        # apply Transformer blocks
        for blk in self.layers:
            x = blk(x)
        x = self.ln(x)

        if self.output_dims > 0:
            x = self.projection(x)

        return self._format_output(x, self.patch_resolution)
