# Copyright (c) OpenMMLab. All rights reserved.
from .dataset_wapper import InstanceDataset
from .transforms import LoadInstanceImage

__all__ = [
    'InstanceDataset', 'LoadInstanceImage'
]
