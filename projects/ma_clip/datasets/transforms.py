import warnings
from typing import Optional

import cv2
import numpy as np
import mmcv
from mmcv.transforms import BaseTransform
import mmengine.fileio as fileio
from mmpretrain.registry import TRANSFORMS

import pycocotools.mask as maskUtils


@TRANSFORMS.register_module()
class LoadInstanceImage(BaseTransform):

    def __init__(self,
                 with_mask: bool = True,
                 exp_factor: float = 1.0,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 channel_order: str = 'bgr',
                 imdecode_backend: str = 'cv2',
                 file_client_args: Optional[dict] = None,
                 ignore_empty: bool = False,
                 *,
                 backend_args: Optional[dict] = None) -> None:
        self.with_mask = with_mask
        self.exp_factor = exp_factor
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        self.imdecode_backend = imdecode_backend

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, filename)
                img_bytes = file_client.get(filename)
            else:
                img_bytes = fileio.get(
                    filename, backend_args=self.backend_args)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, 
                channel_order=self.channel_order, 
                backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        if self.to_float32:
            img = img.astype(np.float32)

        x1, y1, x2, y2 = [int(x) for x in results['bbox']]
        exp_width = int((x2 - x1) * (self.exp_factor - 1) / 2)
        exp_height = int((y2 - y1) * (self.exp_factor - 1) / 2)
        exp_x1, exp_y1 = max(0, x1 - exp_width), max(0, y1 - exp_height)
        exp_x2, exp_y2 = x2 + exp_width, y2 + exp_height

        ori_shape = img.shape[:2]
        img = img[exp_y1: exp_y2, exp_x1: exp_x2]

        mask = results.pop('mask')
        if self.with_mask:
            mask = maskUtils.decode(mask)
            if mask.shape[:2] != ori_shape: # write in json
                mask = cv2.resize(mask, (ori_shape[1], ori_shape[0]))
            mask = mask[exp_y1: exp_y2, exp_x1: exp_x2]
            if self.to_float32:
                mask = mask.astype(np.float32)
            img = np.concatenate((img, mask[..., None]), axis=-1)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'with_mask={self.with_mask}, '
                    f'exp_factor={self.exp_factor}, '
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"channel_order='{self.channel_order}', "
                    f"imdecode_backend='{self.imdecode_backend}', ")

        if self.file_client_args is not None:
            repr_str += f'file_client_args={self.file_client_args})'
        else:
            repr_str += f'backend_args={self.backend_args})'

        return repr_str
