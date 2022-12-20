from torchvision.datasets import ImageNet

import numpy as np

from albumentations.core.transforms_interface import BasicTransform
from albumentations.core.composition import BaseCompose


class AlbumentationsImageNet(ImageNet):
    def __init__(self, transform=None, **kwargs):

        if isinstance(transform, BasicTransform) or isinstance(transform, BaseCompose):
            def transform_fn(image):
                np_image = np.array(image)
                return transform(image=np_image, bboxes=[])['image']

        else:
            transform_fn = transform

        super().__init__(transform=transform_fn, **kwargs)
