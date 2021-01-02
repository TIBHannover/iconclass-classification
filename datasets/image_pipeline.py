import imageio
import numpy as np
import random

from datasets.pipeline import Pipeline, MapDataset


class ImagePreprocessingPipeline(Pipeline):
    def __init__(self, transformation=None, min_size=None, sample_additional=None):
        self.min_size = min_size
        self.sample_additional = sample_additional
        self.transformation = transformation

    def call(self, datasets=None, **kwargs):
        def decode(sample):
            if self.sample_additional is not None:
                if "additional" not in sample:
                    return None
                if isinstance(self.sample_additional, float):
                    prob = self.sample_additional
                else:
                    prob = 0.5

                if random.random() < prob:
                    image = np.asarray(imageio.imread(random.choice(sample["additional"])))
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
            return {
                **sample,
                "image": image,
            }

        return MapDataset(datasets, map_fn=decode)
