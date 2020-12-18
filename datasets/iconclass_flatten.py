import argparse
import logging
import json
import random

import numpy as np
import torch
import torchvision
import imageio

from datasets.image_pipeline import ImagePreprocessingPipeline
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
    ConcatPipeline,
)
from datasets.utils import read_jsonl


class IconclassFlattenDecoderPipeline(Pipeline):
    def __init__(self, num_classes=79, annotation=None, mapping=None, classifier=None):
        self.num_classes = num_classes
        self.annotation = annotation
        self.mapping = mapping
        self.classifier = classifier

    def call(self, datasets=None, **kwargs):
        def decode(sample):
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
                anno = self.annotation[sample["id"]]

            y_onehot_0 = torch.zeros(len(self.mapping))
            for x in anno["classes"]:
                y_onehot_0.scatter_(0, torch.tensor(self.mapping[x]["class_id"]), 1)

            return {**sample, "target": y_onehot_0}
            return sample

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

        self.val_path = dict_args.get("val_path", None)
        self.val_annotation_path = dict_args.get("val_annotation_path", None)

        self.batch_size = dict_args.get("batch_size", None)
        self.num_workers = dict_args.get("num_workers", None)

        self.mapping_path = dict_args.get("mapping_path", None)
        self.classifier_path = dict_args.get("classifier_path", None)

        self.train_annotation = {}
        if self.train_annotation_path is not None:
            for path in self.train_annotation_path:
                self.train_annotation.update(read_jsonl(path, dict_key="id"))

        self.val_annotation = {}
        if self.val_annotation_path is not None:
            # for path in self.val_annotation_path:
            self.val_annotation.update(read_jsonl(self.val_annotation_path, dict_key="id"))

        self.mapping = {}
        if self.mapping_path is not None:
            self.mapping = read_jsonl(self.mapping_path, dict_key="id")

        self.classifier = {}
        if self.classifier_path is not None:
            self.classifier = read_jsonl(self.classifier_path)

    def train(self):
        train_image_transform = torchvision.transforms.Compose(
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

        pipeline_stack = [
            ConcatShufflePipeline([MsgPackPipeline(path=p) for p in self.train_path]),
            IconclassFlattenDecoderPipeline(
                annotation=self.train_annotation, mapping=self.mapping, classifier=self.classifier
            ),
            ImagePreprocessingPipeline(train_image_transform),
        ]

        pipeline = SequencePipeline(pipeline_stack)
        dataloader = torch.utils.data.DataLoader(
            pipeline(), batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True
        )
        return dataloader

    def val(self):
        val_image_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.CenterCrop(size=224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        pipeline_stack = [
            ConcatPipeline([MsgPackPipeline(path=p) for p in [self.val_path]]),
            IconclassFlattenDecoderPipeline(
                annotation=self.val_annotation, mapping=self.mapping, classifier=self.classifier
            ),
            ImagePreprocessingPipeline(val_image_transform),
        ]

        pipeline = SequencePipeline(pipeline_stack)
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

        parser.add_argument("--mapping_path", type=str)
        parser.add_argument("--classifier_path", type=str)

        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--num_workers", type=int, default=8)

        parser.add_argument("--val_path", type=str)
        parser.add_argument("--val_annotation_path", type=str)
        return parser
