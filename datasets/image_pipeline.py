import imageio
import numpy as np
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from datasets.pipeline import Pipeline, MapDataset


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


class ImagePreprocessingPipeline(Pipeline):
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

            # print(f" #### {output_sample.keys()} {output_sample['image'].shape} {output_sample['image'].dtype}")
            return output_sample

        return MapDataset(datasets, map_fn=decode)


class ImageDecodePreprocessingPipeline(Pipeline):
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
            }

        return MapDataset(datasets, map_fn=decode)
