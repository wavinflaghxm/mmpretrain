# Copyright (c) OpenMMLab. All rights reserved.
import math
from einops import rearrange
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch import Tensor

from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.transformer import FFN
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS
from mmpretrain.models.utils import LayerNorm2d, to_2tuple


class SamAttention(BaseModule):
    """
    A cross multi-head attention layer that allows for downscaling the size
    of the embedding after projection to queries, keys, and values.
    """
    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 downsample_rate: int = 1,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.internal_dims = embed_dims // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dims % num_heads == 0, "num_heads must divide embed_dims."

        self.q_proj = nn.Linear(embed_dims, self.internal_dims)
        self.k_proj = nn.Linear(embed_dims, self.internal_dims)
        self.v_proj = nn.Linear(embed_dims, self.internal_dims)

        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(self.internal_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)        

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Get output
        x = attn @ v
        x = self._recombine_heads(x)
        x = self.out_proj(x)
        x = self.proj_drop(x)

        return x


class TwoWayAttentionBlock(BaseModule):
    """
    A transformer block with four layers: (1) self-attention of sparse
    inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
    block on sparse inputs, and (4) cross attention of dense inputs to sparse
    inputs.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Defaults to 2.
        act_cfg (dict): The activation config for FFNs.
            Defaluts to ``dict(type='GELU')``.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        skip_first_layer_pe (bool): skip the PE on the first layer.
    """
    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 feedforward_channels: int = 2048,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 num_fcs: int = 2,
                 act_cfg: dict = dict(type='ReLU'),
                 norm_cfg: dict = dict(type='LN'),
                 attn_downsample_rate: int = 2,
                 skip_first_layer_pe: bool = False) -> None:
        super().__init__()
        self.self_attn = SamAttention(embed_dims, num_heads)
        self.ln1 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.cross_attn_qk = SamAttention(
            embed_dims, num_heads, downsample_rate=attn_downsample_rate)
        self.ln2 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)
        
        self.ln3 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.ln4 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.cross_attn_kq = SamAttention(
            embed_dims, num_heads, downsample_rate=attn_downsample_rate)

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self,
                queries: Tensor,
                keys: Tensor,
                query_pe: Tensor,
                key_pe: Tensor) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.ln1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_qk(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.ln2(queries)

        # FFN block
        queries = self.ln3(self.ffn(queries))

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_kq(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.ln4(keys)

        return queries, keys


@MODELS.register_module(force=True)
class TwoWayTransformer(BaseModule):

    def __init__(self,
                 depth: int,
                 embed_dims: int,
                 num_heads: int,
                 feedforward_channels: int,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 num_fcs: int = 2,
                 act_cfg: dict = dict(type='ReLU'),
                 norm_cfg: dict = dict(type='LN'),
                 attn_downsample_rate: int = 2) -> None:
        super().__init__()
        self.depth = depth
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.feedforward_channels = feedforward_channels

        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=feedforward_channels,
                    drop_rate=drop_rate,
                    drop_path_rate=drop_path_rate,
                    num_fcs=num_fcs,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    attn_downsample_rate=attn_downsample_rate,
                    skip_first_layer_pe=(i == 0)))

        self.final_attn = SamAttention(
            embed_dims, num_heads, downsample_rate=attn_downsample_rate)
        self.ln_final = build_norm_layer(norm_cfg, embed_dims)[1]

    def forward(self,
                queries: Tensor, 
                keys: Tensor, 
                query_pe: Tensor, 
                key_pe: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embed_dims x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embed_dims for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=query_pe,
                key_pe=key_pe)

        # Apply the final self-attenion layer
        queries = torch.cat([queries + query_pe, keys + key_pe], dim=1)
        attn_out = self.final_attn(q=queries, k=queries, v=queries)
        queries = queries + attn_out
        queries = self.ln_final(queries)

        queries, keys = torch.split(queries, queries.shape[1] // 2, dim=1)

        return queries, keys


@MODELS.register_module(force=True)
class MACLIPNeck(BaseModule):

    def __init__(self,
                 mask_encoder: dict,
                 mask_decoder: dict, 
                 init_cfg: dict = None):
        super().__init__(init_cfg)
        self.mask_encoder = MODELS.build(mask_encoder)
        self.mask_decoder = MODELS.build(mask_decoder)

    def forward(self, x: Tensor, mask: Tensor):
        cls_token, x = x[:, 0:1], x[:, 1:]
        mask_fore, mask_back = self.mask_encoder(mask)
        x_fore = x + rearrange(mask_fore, 'b c h w -> b (h w) c')
        x_back = x + rearrange(mask_back, 'b c h w -> b (h w) c')

        x = self.mask_decoder(torch.cat([cls_token, x_fore], dim=1),
                              torch.cat([cls_token, x_back], dim=1))
        return x


@MODELS.register_module(force=True)
class MACLIPMaskEncoder(BaseModule):

    def __init__(self,
                 embed_dims: int,
                 hidden_channels: int,
                 act_cfg: dict = dict(type='GELU'),
                 init_cfg: Optional[dict] = None) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dims (int): The prompts' embedding dimension
          image_embed_size (int): The spatial size of the
            image embedding, as (H, W).
          image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          hidden_channels (int): The number of hidden channels 
            used for encoding input masks.
          act_cfg (dict): The activation to use when encoding
            input masks.
        """
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims

        self.fore_downscaling, self.back_downscaling = [nn.Sequential(
            nn.Conv2d(1, hidden_channels // 4, kernel_size=2, stride=2),
            LayerNorm2d(hidden_channels // 4),
            build_activation_layer(act_cfg),
            nn.Conv2d(hidden_channels // 4,
                      hidden_channels, kernel_size=2, stride=2),
            LayerNorm2d(hidden_channels),
            build_activation_layer(act_cfg),
            nn.Conv2d(hidden_channels, embed_dims, kernel_size=1)) for _ in range(2)]

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          x (Tensor): masks to embed

        Returns:
          Tuple[Tensor, Tensor]: dense embeddings for the masks, in the shape
            B x (embed_dims) x (embed_H) x (embed_W)
        """
        mask_fore = self.fore_downscaling(x)
        mask_back = self.back_downscaling(1 - x)

        return mask_fore, mask_back


@MODELS.register_module(force=True)
class MACLIPMaskDecoder(BaseModule):

    def __init__(self,
                 transformer: dict,
                 patch_size: int = 14,
                 embed_dims: int = 1024,
                 transformer_dims: int = 512,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.transformer = MODELS.build(transformer)
        self.patch_size = to_2tuple(patch_size)
        # used to convert the dim of features from encoder to the dim
        # compatible with that of decoder
        self.decoder_embed = nn.Linear(embed_dims, transformer_dims, bias=True)
        self.pe_layer = RandomPositionEncoding(transformer_dims, with_cls=True)

    def forward(self, x_fore: Tensor, x_back: Tensor) -> Tensor:
        """The forward function.

        The process computes the visible patches' features vectors and the mask
        tokens to output feature vectors, which will be used for
        reconstruction.

        Args:
            x (torch.Tensor): hidden features, which is of shape
                    B x (L * mask_ratio) x C.

        Returns:
            torch.Tensor: The reconstructed feature vectors, which is of
            shape B x (num_patches) x C.
        """
        # embed tokens and append class token to sequence
        x_fore = self.decoder_embed(x_fore)
        x_back = self.decoder_embed(x_back)

        x_fore_pe = self.pe_layer(self.patch_size)
        x_back_pe = self.pe_layer(self.patch_size)
        # apply Transformer
        x_fore, x_back = self.transformer(
            queries=x_fore,
            keys=x_back,
            query_pe=x_fore_pe,
            key_pe=x_back_pe)

        x = torch.cat([x_fore[:, 0:1], x_back[:, 0:1]], dim=1).mean(dim=1)

        return x


class RandomPositionEncoding(BaseModule):
    """
    Positional encoding using random spatial frequencies.
    """
    def __init__(self, 
                 embed_dims: int, 
                 scale: Optional[float] = None,
                 with_cls: bool = False) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, embed_dims // 2)))
        self.cls_embed = nn.Parameter(torch.zeros(1, embed_dims)) if with_cls else None

    def _pe_encoding(self, coords: Tensor) -> Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * torch.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pos = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        pos = rearrange(pos, 'h w c -> (h w) c')  # L x C
        if self.cls_embed is not None:
            pos = torch.cat([self.cls_embed, pos], dim=0)

        return pos  # (L + 1) x C
