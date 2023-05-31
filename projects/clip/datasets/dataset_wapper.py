import copy
import logging
from typing import Union, Sequence, Any

from mmengine.logging import print_log
from mmengine.dataset import BaseDataset, force_full_init, Compose
from mmpretrain.registry import DATASETS, TRANSFORMS

from .tokenizer import tokenize
from .metadata import OPENAI_IMAGENET_TEMPLATES, SIMPLE_IMAGENET_TEMPLATES


@DATASETS.register_module()
class VisionTemplateLanguageDataset:

    TEMPLATE_TYPES = {
        'openai': OPENAI_IMAGENET_TEMPLATES,
        'simple': SIMPLE_IMAGENET_TEMPLATES,
    }

    def __init__(self,
                 dataset: Union[BaseDataset, dict],
                 template: str = 'openai',
                 pipeline: Sequence = (),
                 lazy_init: bool = False) -> None:
        if isinstance(dataset, dict):
            self.dataset = DATASETS.build(dataset)
        else:
            self.dataset = dataset
        self._metainfo = self.dataset.metainfo

        if template not in self.TEMPLATE_TYPES.keys():
            raise ValueError(f'Unsupported `template` {template}, please '
                             f'choose from {self.TEMPLATE_TYPES.keys()}')
        self.templates = self.TEMPLATE_TYPES[template]

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

        use_format = isinstance(self.templates[0], str)
        num_templates = len(self.templates)

        class_names = self._metainfo['classes']
        num_classes = len(class_names)

        text = [t.format(c) if use_format else t(c) 
                for c in class_names for t in self.templates]
        text = tokenize(text).reshape(num_classes, num_templates, -1)
        self.text = text

        self._fully_initialized = True
    
    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.

        Args:
            idx (int): Global index of ``ConcatDataset``.

        Returns:
            dict: The idx-th annotation of the datasets.
        """
        img_data_info = self.dataset.get_data_info(idx)
        gt_label = img_data_info['gt_label']
        img_data_info['text'] = self.text[gt_label]
        return img_data_info

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
        return len(self.dataset)

    def prepare_data(self, idx) -> Any:
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        data_info = self.get_data_info(idx)
        return self.pipeline(data_info)
