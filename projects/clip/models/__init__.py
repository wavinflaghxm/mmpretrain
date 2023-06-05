# Copyright (c) OpenMMLab. All rights reserved.
from .clip_head import CLIPClsHead, ZeroShotClsHead
from .clip import CLIP
from .tokenizer import SimpleTokenizer
from .transformer import CLIPTextTransformer

__all__ = [
    'CLIPClsHead', 'CLIP', 'SimpleTokenizer', 'CLIPTextTransformer', 'ZeroShotClsHead'
]
