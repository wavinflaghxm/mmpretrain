# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

from mmpretrain.registry import TRANSFORMS
from mmpretrain.structures import DataSample
from mmpretrain.datasets import PackInputs


@TRANSFORMS.register_module()
class PackMultiInputs(PackInputs):

    def transform(self, results: dict) -> dict:
        """Method to pack the input data."""
        packed_results = dict(inputs=[])

        self.input_key = self.input_key \
            if isinstance(self.input_key, Sequence) else [self.input_key]
        for input_key in self.input_key:
            if input_key in results:
                input_ = results[input_key]
                packed_results['inputs'].append(self.format_input(input_))

        data_sample = DataSample()

        # Set default keys
        if 'gt_label' in results:
            data_sample.set_gt_label(results['gt_label'])
        if 'gt_score' in results:
            data_sample.set_gt_score(results['gt_score'])
        if 'mask' in results:
            data_sample.set_mask(results['mask'])

        # Set custom algorithm keys
        for key in self.algorithm_keys:
            if key in results:
                data_sample.set_field(results[key], key)

        # Set meta keys
        for key in self.meta_keys:
            if key in results:
                data_sample.set_field(results[key], key, field_type='metainfo')

        packed_results['data_samples'] = data_sample
        return packed_results
