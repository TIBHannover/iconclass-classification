import imageio
import numpy as np
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from .transforms import intersection_over_union, coverage_bbox
from .pipeline import Pipeline, MapDataset
from .transforms import CenterCropWithBBox, RandomCropWithBBox, RandomResize


class IconclassImagePreprocessingPipeline(Pipeline):
    def __init__(self, transformation=None, min_size=None, sample_additional=None):
        self.min_size = min_size
        self.sample_additional = sample_additional
        self.transformation = transformation

    def call(self, datasets=None, **kwargs):
        def decode(sample):
            has_additional = True
            if "additional" not in sample:
                has_additional = False
            if has_additional and len(sample["additional"]) == 0:
                has_additional = False

            if self.sample_additional is not None and has_additional:
                if isinstance(self.sample_additional, float):
                    prob = self.sample_additional
                else:
                    prob = 0.5

                if random.random() < prob:
                    # print(random.choice(sample["additional"]).keys())
                    image = np.asarray(imageio.imread(random.choice(sample["additional"])[b"image"]))
                else:
                    image = np.asarray(imageio.imread(sample["image_data"]))

            else:
                image = np.asarray(imageio.imread(sample["image_data"]))

            if self.min_size is not None and self.min_size > 0:
                if image.shape[0] < self.min_size or image.shape[1] < self.min_size:
                    return None

            if "additional" in sample:
                del sample["additional"]

            del sample["image_data"]
            if self.transformation:
                image = self.transformation(image)
            else:
                image = image

            output_sample = {
                **sample,
                "image": image,
                "image_mask": torch.ones(image.shape[1:3], dtype=torch.bool, device=image.device),
            }

            return output_sample

        return MapDataset(datasets, map_fn=decode)


class ImageDecodePipeline(Pipeline):
    def __init__(self, transformation=None):
        self.transformation = transformation

    def call(self, datasets=None, **kwargs):
        def decode(sample):
            image = sample["image"]
            if self.transformation:
                image = self.transformation(image)
            else:
                image = image
            return {
                **sample,
                "image": image,
                "image_mask": torch.ones(image.shape[1:3], dtype=torch.bool, device=image.device),
            }

        return MapDataset(datasets, map_fn=decode)


class CocoImageTrainPreprocessingPipeline(Pipeline):
    def __init__(self, output_sizes=[244], max_size=244, min_size=None, min_coverage=0.6):
        # for x in range
        # self.output_sizes = [[x,x] for x in ]
        self.min_size = min_size
        self.min_coverage = min_coverage
        self.cropper = RandomCropWithBBox()
        self.transforms = T.Compose(
            [
                T.ToPILImage(),
                T.RandomHorizontalFlip(),
                RandomResize(output_sizes, max_size=max_size),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def call(self, datasets=None, **kwargs):
        def decode(sample):
            image = np.asarray(imageio.imread(sample["image_data"]))

            if self.min_size is not None and self.min_size > 0:
                if image.shape[0] < self.min_size or image.shape[1] < self.min_size:
                    return None

            del sample["image_data"]

            image_with_crop = self.cropper(image)

            crop = image_with_crop["crop"]

            classes = []
            ious = []
            for c in sample["classes"]:
                iou = coverage_bbox(crop, c["bbox"])
                ious.append(iou)
                if iou < self.min_coverage:
                    continue

                classes.append(c["id"])

            classes = list(set(classes))

            if len(classes) == 0:
                # print(f"{ious} {crop} {image.shape} {sample['classes']}")
                return None

            image = self.transforms(image_with_crop["image"])

            return {
                **sample,
                "image": image,
                "image_mask": torch.ones(image.shape[1:3], dtype=torch.bool, device=image.device),
                "classes": classes,
            }

        return MapDataset(datasets, map_fn=decode)


class CocoImageTestPreprocessingPipeline(Pipeline):
    def __init__(self, output_sizes=[244], min_coverage=0.6):
        # for x in range
        # self.output_sizes = [[x,x] for x in ]
        self.min_size = None
        self.min_coverage = min_coverage
        self.cropper = CenterCropWithBBox()
        self.transforms = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(output_sizes, interpolation=PIL.Image.BICUBIC),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def call(self, datasets=None, **kwargs):
        def decode(sample):
            image = np.asarray(imageio.imread(sample["image_data"]))

            if self.min_size is not None and self.min_size > 0:
                if image.shape[0] < self.min_size or image.shape[1] < self.min_size:
                    return None

            del sample["image_data"]

            image_with_crop = self.cropper(image)

            crop = image_with_crop["crop"]

            classes = []
            ious = []
            for c in sample["classes"]:
                iou = coverage_bbox(crop, c["bbox"])
                ious.append(iou)
                if iou < self.min_coverage:
                    continue

                classes.append(c["id"])

            classes = list(set(classes))

            if len(classes) == 0:
                # print(f"{ious} {crop} {image.shape} {sample['classes']}")
                return None

            image = self.transforms(image_with_crop["image"])

            return {
                **sample,
                "image": image,
                "image_mask": torch.ones(image.shape[1:3], dtype=torch.bool, device=image.device),
                "classes": classes,
            }

        return MapDataset(datasets, map_fn=decode)
