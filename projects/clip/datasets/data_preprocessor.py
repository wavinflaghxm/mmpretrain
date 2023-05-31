# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.model import stack_batch

from mmpretrain.registry import MODELS
from mmpretrain.models import ClsDataPreprocessor


@MODELS.register_module()
class VisionLanguageDataPreprocessor(ClsDataPreprocessor):

    def forward(self, data: dict, training: bool = False) -> dict:
        """Preprocesses the vision-language data into the model input format.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        inputs = self.cast_data(data['inputs'])
        data_samples = data.get('data_samples', None)

        assert isinstance(inputs, list)
        vision, language = inputs[0], inputs[1]

        if not isinstance(language, torch.Tensor):
            language = stack_batch(language, self.pad_size_divisor,
                                   self.pad_value)

        data = super().forward(
            {'inputs': vision, 'data_samples': data_samples}, training)

        data['inputs'] = [data['inputs'], language]
        return data
