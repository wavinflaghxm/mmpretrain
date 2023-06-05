from typing import Optional

import numpy as np
from mmcv.transforms import BaseTransform
from mmpretrain.registry import TRANSFORMS

import pycocotools.mask as maskUtils


@TRANSFORMS.register_module()
class CropInstanceFromImage(BaseTransform):

    def __init__(self, exp_factor: float = 1.0) -> None:
        self.exp_factor = exp_factor

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to crop the instance from the image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        x1, y1, x2, y2 = [int(x) for x in results['bbox']]
        exp_width = int((x2 - x1) * (self.exp_factor - 1) / 2)
        exp_height = int((y2 - y1) * (self.exp_factor - 1) / 2)
        exp_x1, exp_y1 = max(0, x1 - exp_width), max(0, y1 - exp_height)
        exp_x2, exp_y2 = x2 + exp_width, y2 + exp_height

        img = results['img']
        ori_shape = img.shape[:2]
        img = img[exp_y1: exp_y2, exp_x1: exp_x2]

        mask = results.get('mask', None)
        if mask is not None:
            mask = maskUtils.decode(mask)
            if mask.shape[:2] != ori_shape: # write in json
                import warnings, cv2
                # warnings.warn(
                #         f'The image size {ori_shape} is different from '
                #         f'the mask size {mask.shape[:2]}.')
                mask = cv2.resize(mask, (ori_shape[1], ori_shape[0]))
            mask = mask[exp_y1: exp_y2, exp_x1: exp_x2]
            results['mask'] = mask

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results
    
    def __repr__(self):
        return self.__class__.__name__ + \
            f'(exp_factor={self.exp_factor})'


@TRANSFORMS.register_module()
class InstanceMaskPacker(BaseTransform):

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to concatenate the mask and the image
           and convert the image channel to RGB.
        """
        img = results['img']
        if img.ndim == 3 and img.shape[-1] == 3:
            img = np.flip(img, axis=-1)
        elif img.ndim == 2:
            img = np.expand_dims(img, -1)
        else:
            raise RuntimeError(f'Invalid input image size {img.shape}.')

        mask = results.pop('mask', None)
        if mask is not None:
            mask = mask.astype(img.dtype)
            mask = np.expand_dims(mask, -1) if mask.ndim == 2 else mask
            img = np.concatenate((img, mask), axis=-1)

        results['img'] = img
        return results
