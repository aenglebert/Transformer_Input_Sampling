from torchvision.datasets.imagenet import ImageNet, extract_archive

import numpy as np

from albumentations.core.transforms_interface import BasicTransform
from albumentations.core.composition import BaseCompose

from typing import Any, Dict, Iterator, List, Optional, Tuple
import os

import xml.etree.ElementTree as ET


class AlbumentationsImageNet(ImageNet):
    def __init__(self, transform=None, **kwargs):

        if isinstance(transform, BasicTransform) or isinstance(transform, BaseCompose):
            def transform_fn(image):
                np_image = np.array(image)
                return transform(image=np_image, bboxes=[])['image']

        else:
            transform_fn = transform

        super().__init__(transform=transform_fn, **kwargs)


class BboxesImageNet(ImageNet):
    def __init__(self, root: str, split: str = "val", **kwargs: Any) -> None:
        super().__init__(root, split, **kwargs)

        self.bbox_root = os.path.join(self.root, "bbox")
        self.bbox_split_root = os.path.join(self.bbox_root, split)

        if not os.path.isdir(os.path.join(self.bbox_split_root, split)):
            parse_bbox_archive(self.root, split=split)

        self.bboxes_paths = []
        self.bboxes = []

        for idx, (img, class_idx) in enumerate(self.imgs):
            img_path = img
            filename = os.path.split(img_path)[1]
            index = filename.split(".")[0].split("_")[-1]

            bbox_filename = "ILSVRC2012_" + split + "_" + index + ".xml"
            bbox_path = os.path.join(self.bbox_split_root, bbox_filename)
            wnid = img_path.split("/")[-2]

            self.bboxes_paths.append(bbox_path)

            bboxes = self.get_bboxes(bbox_path, wnid)

            self.bboxes.append(bboxes)

            self.imgs[idx] = (img, bboxes)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, bboxes = self.imgs[index]
        img = self.loader(path)

        if isinstance(self.transform, BaseCompose) or isinstance(self.transform, BasicTransform):
            img = np.array(img)
            transformed = self.transform(image=img, bboxes=bboxes)

            img = transformed['image']
            bboxes = transformed['bboxes']

        else:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                bboxes = self.target_transform(bboxes)

        return img, bboxes

    def get_bboxes(self, path, target_wnid):
        root = ET.parse(path).getroot()
        size = root.find('size')

        width = int(size.find('width').text)
        height = int(size.find('height').text)

        bboxes = []

        for obj in root.findall('object'):
            wnid = obj.find("name").text
            if wnid == target_wnid:
                coord_dict = dict([(coord.tag, int(coord.text)) for coord in obj.find("bndbox")])
                bboxes.append([coord_dict['xmin'],
                               coord_dict['ymin'],
                               coord_dict['xmax'],
                               coord_dict['ymax'],
                               self.wnid_to_idx[target_wnid]])

        return bboxes


def parse_bbox_archive(root: str, file: Optional[str] = None, folder: str = "bbox", split="val") -> None:
    """Parse the bbox validation archive of the ImageNet2012 classification dataset and
    prepare it for usage with the ImageNet dataset.
    Args:
        root (str): Root directory containing the bbox_val archive
        file (str, optional): Name of train images archive. Defaults to
            'ILSVRC2012_bbox_val_v3.tgz'
        folder (str, optional): Optional name for bbox folder. Defaults to
            'bbox'
    """
    if file is None:
        file = "ILSVRC2012_bbox_val_v3.tgz"

    bbox_root = os.path.join(root, folder)
    bbox_split_root = os.path.join(bbox_root, split)
    extract_archive(os.path.join(root, file), bbox_root)

    bbox = sorted(os.path.join(bbox_split_root, bbox) for bbox in os.listdir(bbox_split_root))
