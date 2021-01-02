import argparse
import logging

import torch
import torchvision

from datasets.image_pipeline import ImagePreprocessingPipeline
from datasets.datasets import DatasetsManager
from datasets.pipeline import (
    Pipeline,
    MapDataset,
    MsgPackPipeline,
    SequencePipeline,
    ConcatShufflePipeline,
    ConcatPipeline,
    DummyPipeline,
)
from datasets.utils import read_jsonl


class IconclassDecoderPipeline(Pipeline):
    def __init__(self, num_classes=79, annotation=None):
        self.num_classes = num_classes
        self.annotation = annotation

    def call(self, datasets=None, **kwargs):
        def decode(sample):

            sample = {
                "image_data": sample[b"image"],
                "id": sample[b"id"].decode("utf-8"),
                "path": sample[b"path"].decode("utf-8"),
            }

            if b"additional" in sample:
                sample.update({"additional": sample[b"additional"]})

            if sample["id"] not in self.annotation:
                logging.info(f"Dataset: {sample['id']} not in annotation")
                return None
            else:
                anno = self.annotation[sample["id"]]

            return {**sample, **anno}

        return MapDataset(datasets, map_fn=decode)


@DatasetsManager.export("iconclass_flatten")
class IconClassFlattenDataloader:
    def __init__(self, args=None, **kwargs):
        if args is not None:
            dict_args = vars(args)
            dict_args.update(kwargs)
        else:
            dict_args = kwargs

        self.train_path = dict_args.get("train_path", None)
        self.train_annotation_path = dict_args.get("train_annotation_path", None)

        self.train_filter_min_dim = dict_args.get("train_filter_min_dim", None)
        self.train_sample_additional = dict_args.get("train_sample_additional", None)

        self.val_path = dict_args.get("val_path", None)
        self.val_annotation_path = dict_args.get("val_annotation_path", None)
        self.val_filter_min_dim = dict_args.get("val_filter_min_dim", None)

        self.batch_size = dict_args.get("batch_size", None)
        self.num_workers = dict_args.get("num_workers", None)

        self.train_annotation = {}
        if self.train_annotation_path is not None:
            for path in self.train_annotation_path:
                self.train_annotation.update(read_jsonl(path, dict_key="id"))

        self.val_annotation = {}
        if self.val_annotation_path is not None:
            # for path in self.val_annotation_path:
            self.val_annotation.update(read_jsonl(self.val_annotation_path, dict_key="id"))

    def train_image_pipeline(self):
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomRotation(10),
                torchvision.transforms.RandomResizedCrop(size=224, scale=(0.5, 1.0)),
                torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        return ImagePreprocessingPipeline(
            transforms, min_size=self.train_filter_min_dim, sample_additional=self.train_sample_additional
        )

    def train_decode_pieline(self):
        return SequencePipeline(
            [
                ConcatShufflePipeline([MsgPackPipeline(path=p) for p in self.train_path]),
                IconclassDecoderPipeline(annotation=self.train_annotation),
            ]
        )

    def train_mapping_pipeline(self):
        return DummyPipeline()

    def train(self):
        pipeline = SequencePipeline(
            [self.train_decode_pieline(), self.train_mapping_pipeline(), self.train_image_pipeline()]
        )

        dataloader = torch.utils.data.DataLoader(
            pipeline(), batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True
        )
        return dataloader

    def val_image_pipeline():
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.CenterCrop(size=224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        return ImagePreprocessingPipeline(transforms, min_size=self.val_filter_min_dim,)

    def val_decode_pieline(self):
        return SequencePipeline(
            [
                ConcatShufflePipeline([MsgPackPipeline(path=p) for p in self.val_path]),
                IconclassDecoderPipeline(annotation=self.val_annotation),
            ]
        )

    def val_mapping_pipeline(self):
        return DummyPipeline()

    def val(self):
        pipeline = SequencePipeline([self.val_decode_pieline(), self.val_mapping_pipeline(), self.val_image_pipeline()])

        dataloader = torch.utils.data.DataLoader(
            pipeline(), batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True
        )
        return dataloader

    def test(self):
        pass

    @classmethod
    def add_args(cls, parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--train_path", nargs="+", type=str)
        parser.add_argument("--train_annotation_path", nargs="+", type=str)
        parser.add_argument("--train_filter_min_dim", type=int, default=128, help="delete images with smaller size")
        parser.add_argument("--train_sample_additional", type=float)

        parser.add_argument("--num_workers", type=int, default=8)

        parser.add_argument("--val_path", type=str)
        parser.add_argument("--val_annotation_path", type=str)
        parser.add_argument("--val_filter_min_dim", type=int, default=128, help="delete images with smaller size")

        return parser
