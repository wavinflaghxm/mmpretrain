import copy
import logging
from typing import Union, Sequence, Optional, List, Tuple, Any

from mmengine.logging import print_log
from mmengine.dataset import BaseDataset, force_full_init, Compose
from mmpretrain.registry import DATASETS, TRANSFORMS


@DATASETS.register_module()
class InstanceDataset:

    def __init__(self,
                 dataset: Union[BaseDataset, dict],
                 filter_cfg: Optional[dict] = None,
                 pipeline: Sequence = (),
                 lazy_init: bool = False) -> None:
        self.dataset: BaseDataset
        if isinstance(dataset, dict):
            self.dataset = DATASETS.build(dataset)
        elif isinstance(dataset, BaseDataset):
            self.dataset = dataset
        else:
            raise TypeError(
                'elements in datasets sequence should be config or '
                f'`BaseDataset` instance, but got {type(dataset)}')
        self._metainfo = self.dataset.metainfo
        self.filter_cfg = copy.deepcopy(filter_cfg)

        transforms = []
        for transform in pipeline:
            if isinstance(transform, dict):
                transforms.append(TRANSFORMS.build(transform))
            else:
                transforms.append(transform)
        self.pipeline = Compose(transforms)

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    @property
    def metainfo(self) -> dict:
        return copy.deepcopy(self._metainfo)

    def full_init(self):
        """Loop to ``full_init`` each dataset."""
        if self._fully_initialized:
            return

        self.dataset.full_init()

        instance_indices = self._get_instance_and_filter()
        self.instance_indices = instance_indices

        self._fully_initialized = True

    def _get_instance_and_filter(self) -> List[Tuple[int, int]]:
        min_size = self.filter_cfg.get('min_size', 0)

        instance_indices = []
        for img_idx, data in enumerate(self.dataset):
            instances = data['instances']
            for ins_idx, instance in enumerate(instances):
                x1, y1, x2, y2 = instance['bbox']
                if min((x2 - x1), (y2 - y1)) >= min_size:
                    instance_indices.append((img_idx, ins_idx))
        return instance_indices
    
    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.

        Args:
            idx (int): Global index of ``ConcatDataset``.

        Returns:
            dict: The idx-th annotation of the datasets.
        """
        img_idx, ins_idx = self.instance_indices[idx]
        img_data_info = self.dataset.get_data_info(img_idx)
        instances = img_data_info['instances']
        ins_data_info = {
            'img_path': img_data_info['img_path'],
            'gt_label': instances[ins_idx]['bbox_label'],
            'bbox': instances[ins_idx]['bbox'],
            'mask': instances[ins_idx]['mask']
        }
        return ins_data_info

    def __getitem__(self, idx):
        if not self._fully_initialized:
            print_log(
                'Please call `full_init` method manually to accelerate the '
                'speed.',
                logger='current',
                level=logging.WARNING)
            self.full_init()
        
        data = self.prepare_data(idx)
        return data

    @force_full_init
    def __len__(self):
        return len(self.instance_indices)

    def prepare_data(self, idx) -> Any:
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        data_info = self.get_data_info(idx)
        return self.pipeline(data_info)
