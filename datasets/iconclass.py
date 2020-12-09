import argparse

import torch

from datasets.datasets import DatasetsManager
from datasets.pipeline import (
    Pipeline,
    MapDataset,
    MsgPackPipeline,
    MapPipeline,
    SequencePipeline,
    FilterPipeline,
    ConcatShufflePipeline,
    RepeatPipeline,
    ImagePipeline,
)
from datasets.utils import read_jsonl


class IconclassDecoderPipeline(Pipeline):
    def __init__(self, num_classes=79, annotation=None):
        self.num_classes = num_classes
        self.annotation = annotation

    def call(self, datasets=None, **kwargs):
        def decode(sample):
            print(sample.keys())
            if b"additional" in sample:
                # print(sample[b"additional"])
                pass

            sample = {
                "image_data": sample[b"image"],
                "id": sample[b"id"].decode("utf-8"),
                "path": sample[b"path"].decode("utf-8"),
            }

            if sample["id"] not in self.annotation:
                logging.info(f"Dataset: {sample['id']} not in annotation")
                return None
            else:
                sample.update(self.annotation[sample["id"]])

            print(sample.keys())
            print(f'{sample["classes"]} {sample["name"]}')

            return sample

        return MapDataset(datasets, map_fn=decode)


class ImagePreprocessingPipeline(Pipeline):
    def __init__(self, teacher_image=False, transformation=None, teacher_transform=None):

        self.teacher_image = teacher_image
        self.transformation = transformation
        self.teacher_transform = teacher_transform

    def call(self, datasets=None, **kwargs):
        def decode(sample):
            original_image = np.asarray(imageio.imread(sample["image_data"]))
            del sample["image_data"]
            if self.transformation:
                image = self.transformation(original_image)

            if self.teacher_image:
                if self.teacher_transform:
                    teacher_image = self.teacher_transform(original_image)
                elif self.transformation:
                    teacher_image = self.transformation(original_image)
                else:
                    teacher_image = original_image

                return {
                    **sample,
                    "image": image,
                    "teacher_image": teacher_image,
                }
            return {
                **sample,
                "image": image,
            }

        return MapDataset(datasets, map_fn=decode)


@DatasetsManager.export("iconclass")
class IconClassDataloader:
    def __init__(self, args=None, **kwargs):
        if args is not None:
            dict_args = vars(args)
            dict_args.update(kwargs)
        else:
            dict_args = kwargs

        self.train_path = dict_args.get("train_path", None)
        self.batch_size = dict_args.get("batch_size", None)
        self.num_workers = dict_args.get("num_workers", None)
        self.train_annotation_path = dict_args.get("train_annotation_path", None)

        self.train_annotation = {}
        if self.train_annotation_path is not None:
            for path in self.train_annotation_path:
                self.train_annotation.update(read_jsonl(path, dict_key="id"))

    def train(self):
        pipeline_stack = [
            ConcatShufflePipeline([MsgPackPipeline(path=p) for p in self.train_path]),
            IconclassDecoderPipeline(annotation=self.train_annotation),
        ]

        pipeline = SequencePipeline(pipeline_stack)
        dataloader = torch.utils.data.DataLoader(
            pipeline(), batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True
        )
        return dataloader

    def val(self):
        pass

    def test(self):
        pass

    @classmethod
    def add_args(cls, parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--train_path", nargs="+", type=str)
        parser.add_argument("--train_annotation_path", nargs="+", type=str)

        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--val_path", type=str)
        return parser
