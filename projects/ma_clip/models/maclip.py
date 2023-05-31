# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
from torch import Tensor

from mmpretrain.registry import MODELS
from projects.clip.models import CLIP


@MODELS.register_module()
class MACLIP(CLIP):

    def extract_feat(self, inputs: List[Tensor]):
        image, text = inputs[0], inputs[1]
        # split image and mask
        assert image is not None
        image, mask = torch.split(image, [image.shape[1] - 1, 1], dim=1)
        # encode image and text
        image = self.encode_image(image)
        text = self.encode_text(text) if text is not None else None

        return image, mask, text, self.logit_scale.exp()