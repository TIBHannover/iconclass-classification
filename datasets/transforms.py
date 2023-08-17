import numpy as np
import random

import PIL
from numpy.lib.arraysetops import isin
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from .utils import get_element


def coverage_bbox(outer_bbox, object_bbox):
    # determine the coordinates of the intersection rectangle
    x_left = max(outer_bbox[0], object_bbox[0])
    y_top = max(outer_bbox[1], object_bbox[1])
    x_right = min(outer_bbox[0] + outer_bbox[2], object_bbox[0] + object_bbox[2])
    y_bottom = min(outer_bbox[1] + outer_bbox[3], object_bbox[1] + object_bbox[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    object_bbox_area = (object_bbox[0] + object_bbox[2] - object_bbox[0]) * (
        object_bbox[1] + object_bbox[3] - object_bbox[1]
    )

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    if object_bbox_area <= 0:
        return 0.0
    iou = intersection_area / float(object_bbox_area)
    return iou


def intersection_over_union(bbox1, bbox2):
    # determine the coordinates of the intersection rectangle
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    y_bottom = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bbox1_area = (bbox1[0] + bbox1[2] - bbox1[0]) * (bbox1[1] + bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[0] + bbox2[2] - bbox2[0]) * (bbox2[1] + bbox2[3] - bbox2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
    return iou


def resize(image, size, max_size=None):
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    return rescaled_image


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, size, self.max_size)


class CenterCropWithBBox:
    def __init__(self, image_key=None, output_key=None, crop_key=None):
        self.image_key = image_key
        self.crop_key = crop_key
        if self.crop_key is None:
            self.crop_key = "crop"
        self.output_key = output_key
        if self.output_key is None:
            self.output_key = "image"

    def __call__(self, sample):
        if self.image_key is not None:
            image = get_element(sample, self.image_key)
        else:
            image = sample
        image = np.array(image)
        assert image is not None, f"Could find a image under {self.image_key}"

        if image.shape[1] > image.shape[0]:
            w = image.shape[0]
            h = image.shape[0]
            x = int((image.shape[1] - w) / 2)
            y = 0
            crop = [x, y, w, h]
        else:
            w = image.shape[1]
            h = image.shape[1]
            x = 0
            y = int((image.shape[0] - h) / 2)
            crop = [x, y, w, h]

        output = image[crop[1] : crop[1] + crop[3], crop[0] : crop[0] + crop[2], ...]

        if isinstance(sample, dict):
            return {**sample, self.output_key: output, self.crop_key: crop}
        else:
            return {self.output_key: output, self.crop_key: crop}


class RandomCropWithBBox:
    def __init__(
        self, image_key=None, output_key=None, crop_key=None, area=[0.6, 0.9], crop_ar=[0.75, 1.33], max_try=100
    ):
        self.crop_ar = crop_ar
        self.image_key = image_key
        self.area = area
        self.max_try = max_try

        self.crop_key = crop_key
        if self.crop_key is None:
            self.crop_key = "crop"
        self.output_key = output_key
        if self.output_key is None:
            self.output_key = "image"

    def __call__(self, sample):
        if self.image_key is not None:
            image = get_element(sample, self.image_key)
        else:
            image = sample
        image = np.array(image)
        assert image is not None, f"Could find a image under {self.image_key}"

        crop = [0, 0, image.shape[1], image.shape[0]]
        for _ in range(self.max_try):
            ar = random.uniform(*self.crop_ar)
            area = random.uniform(*self.area) * image.shape[0] * image.shape[1]

            h = (area / ar) ** (1 / 2)
            w = int(area / h)
            h = int(h)
            if h >= image.shape[0]:
                continue

            if w >= image.shape[1]:
                continue

            y = random.randint(0, image.shape[0] - h)
            x = random.randint(0, image.shape[1] - w)

            crop = [x, y, w, h]
            break
        output = image[crop[1] : crop[1] + crop[3], crop[0] : crop[0] + crop[2], ...]

        if isinstance(sample, dict):
            return {**sample, self.output_key: output, self.crop_key: crop}
        else:
            return {self.output_key: output, self.crop_key: crop}
