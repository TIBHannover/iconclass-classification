import imageio
import numpy as np

from datasets.pipeline import Pipeline, MapDataset


class ImagePreprocessingPipeline(Pipeline):
    def __init__(self, transformation=None):

        self.transformation = transformation

    def call(self, datasets=None, **kwargs):
        def decode(sample):
            original_image = np.asarray(imageio.imread(sample["image_data"]))
            del sample["image_data"]
            if self.transformation:
                image = self.transformation(original_image)
            else:
                image = original_image
            return {
                **sample,
                "image": image,
            }

        return MapDataset(datasets, map_fn=decode)
