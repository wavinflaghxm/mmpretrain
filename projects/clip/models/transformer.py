from typing import Optional
from collections import OrderedDict

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from mmcv.cnn.bricks import build_activation_layer, build_norm_layer
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS


class AttentionPool2d(BaseModule):
    def __init__(
            self,
            embed_dims: int,
            output_dims: int,
            num_heads: int = 8,
            num_queries: int = 256,
            norm_cfg: dict = dict(type='LN'),
            init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)

        self.query = nn.Parameter(torch.randn(num_queries, output_dims))
        self.attn = nn.MultiheadAttention(output_dims, num_heads, kdim=embed_dims, vdim=embed_dims)
        self.ln_q = build_norm_layer(norm_cfg, output_dims)[1]
        self.ln_k = build_norm_layer(norm_cfg, embed_dims)[1]

    def forward(self, x: Tensor) -> Tensor:
        N = x.shape[0]
        x = self.ln_k(x).permute(1, 0, 2)  # NLD -> LND
        q = self.ln_q(self.query)
        out = self.attn(self._repeat(q, N), x, x, need_weights=False)[0]
        return out.permute(1, 0, 2)  # LND -> NLD

    def _repeat(self, query: Tensor, N: int) -> Tensor:
        return query.unsqueeze(1).repeat(1, N, 1)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class ResidualAttentionBlock(BaseModule):
    def __init__(self,
                 d_model: int,
                 n_head: int,
                 mlp_ratio: float = 4.0,
                 ls_init_value: Optional[float] = None,
                 act_cfg: dict = dict(type='GELU'),
                 norm_cfg: dict = dict(type='LN'),
                 is_cross_attention: bool = False,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)

        self.ln_1 = build_norm_layer(norm_cfg, d_model)[1]
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        if is_cross_attention:
            self.ln_1_kv = build_norm_layer(norm_cfg, d_model)[1]

        self.ln_2 = build_norm_layer(norm_cfg, d_model)[1]
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", build_activation_layer(act_cfg)),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def attention(self,
                  q_x: Tensor,
                  k_x: Optional[Tensor] = None,
                  v_x: Optional[Tensor] = None,
                  attn_mask: Optional[Tensor] = None) -> Tensor:
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(
            q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self,
                q_x: Tensor,
                k_x: Optional[Tensor] = None,
                v_x: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None) -> Tensor:
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None

        x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class Transformer(BaseModule):

    def __init__(self,
                 width: int,
                 num_layers: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 ls_init_value: Optional[float] = None,
                 act_cfg: dict = dict(type='GELU'),
                 norm_cfg: dict = dict(type='LN'),
                 with_cp: bool = False,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.width = width
        self.num_layers = num_layers
        self.with_cp = with_cp

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width, num_heads, mlp_ratio, 
                ls_init_value=ls_init_value, 
                act_cfg=act_cfg, 
                norm_cfg=norm_cfg) for _ in range(num_layers)])

    def get_cast_dtype(self) -> torch.dtype:
        if hasattr(self.resblocks[0].mlp.c_fc, 'int8_original_dtype'):
            return self.resblocks[0].mlp.c_fc.int8_original_dtype
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        for r in self.resblocks:
            if self.with_cp and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x


@MODELS.register_module()
class CLIPTextTransformer(BaseModule):
    num_extra_tokens = 1  # class token
    OUT_TYPES = {'raw', 'token', 'both'}

    def __init__(self,
                 context_length: int = 77,
                 vocab_size: int = 49408,
                 embed_dims: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 12,
                 ls_init_value: Optional[float] = None,
                 act_cfg: dict = dict(type='GELU'),
                 norm_cfg: dict = dict(type='LN'),
                 pad_id: int = 0,
                 out_type: str = 'token',
                 with_cls_token: bool = False,
                 frozen_stages: int = -1,
                 with_cp: bool = False,
                 init_cfg: Optional[dict] = None) -> None:
        super(CLIPTextTransformer, self).__init__(init_cfg=init_cfg)
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width = embed_dims
        self.num_heads = num_heads
        self.pad_id = pad_id

        # Set out type
        if out_type not in self.OUT_TYPES:
            raise ValueError(f'Unsupported `out_type` {out_type}, please '
                             f'choose from {self.OUT_TYPES}')
        self.out_type = out_type

        if with_cls_token:
            self.cls_token = nn.Parameter(torch.empty(width))
        else:
            self.cls_token = None
            self.num_extra_tokens = 0

        num_pos = self.context_length + self.num_extra_tokens
        self.token_embed = nn.Embedding(vocab_size, width)
        self.pos_embed = nn.Parameter(torch.empty(num_pos, width))
        self.transformer = Transformer(
            width=width,
            num_layers=num_layers,
            num_heads=num_heads,
            ls_init_value=ls_init_value,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            with_cp=with_cp)
        self.ln_final = build_norm_layer(norm_cfg, width)[1]

        self.frozen_stages = frozen_stages
         # freeze stages only when self.frozen_stages > 0
        if self.frozen_stages > 0:
            self._freeze_stages()

        self.register_buffer('attn_mask', self.build_attention_mask(num_pos), persistent=False)

    def init_weights(self):
        """Weight initialization."""
        super().init_weights()

        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            nn.init.normal_(self.token_embed.weight, std=0.02)
            nn.init.normal_(self.pos_embed, std=0.01)
            if self.cls_token is not None:
                nn.init.normal_(self.cls_token, std=0.01)

            width = self.transformer.width
            proj_std = (width ** -0.5) * ((2 * self.transformer.num_layers) ** -0.5)
            attn_std = width ** -0.5
            fc_std = (2 * width) ** -0.5
            for block in self.transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def build_attention_mask(self, num_dims):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(num_dims, num_dims)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def build_cls_mask(self, x: Tensor, cast_dtype: torch.dtype):
        cls_mask = (x != self.pad_id).unsqueeze(1)
        cls_mask = F.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=1.0)
        additive_mask = torch.empty(cls_mask.shape, dtype=cast_dtype, device=cls_mask.device)
        additive_mask.fill_(0)
        additive_mask.masked_fill_(~cls_mask, float("-inf"))
        additive_mask = torch.repeat_interleave(additive_mask, self.num_heads, 0)
        return additive_mask

    def _freeze_stages(self):
        # freeze position embedding
        if self.pos_embed is not None:
            self.pos_embed.requires_grad = False
        # freeze token embedding
        self.token_embed.eval()
        for param in self.token_embed.parameters():
            param.requires_grad = False
        # freeze cls_token
        if self.cls_token is not None:
            self.cls_token.requires_grad = False
        # freeze layers
        for i in range(1, self.frozen_stages + 1):
            m = self.transformer.resblocks[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        # freeze the last layer norm
        if self.frozen_stages == len(self.transformer.resblocks):
            self.ln_final.eval()
            for param in self.ln_final.parameters():
                param.requires_grad = False
    
    def _repeat(self, x: Tensor, N: int) -> Tensor:
        return x.reshape(1, 1, -1).repeat(N, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        cast_dtype = self.transformer.get_cast_dtype()
        seq_len = x.shape[1]

        res_x = x
        x = self.token_embed(x).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        attn_mask = self.attn_mask
        if self.cls_token is not None:
            seq_len += 1
            x = torch.cat([x, self._repeat(self.cls_token, x.shape[0])], dim=1)
            cls_mask = self.build_cls_mask(res_x, cast_dtype)
            attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]

        x = x + self.pos_embed[:seq_len].to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        return self._format_output(x, res_x)

    def _format_output(self, x: Tensor, res_x: Tensor):
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if self.cls_token is not None:
            pooled, tokens = x[:, -1], x[:, :-1]
            pooled = self.ln_final(pooled)
        else:
            x = self.ln_final(x)
            pooled, tokens = x[torch.arange(x.shape[0]), res_x.argmax(dim=-1)], x

        if self.out_type == 'raw':
            return tokens
        if self.out_type == 'token':
            return pooled
        if self.out_type == 'both':
            return pooled, tokens
