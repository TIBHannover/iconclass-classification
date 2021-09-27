import argparse
import logging
import json
import random

import numpy as np
import torch
import torchvision
import imageio

from datasets.image_pipeline import IconclassImagePreprocessingPipeline
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

from datasets.iconclass import IconclassDataloader


class IconclassFlattenDecoderPipeline(Pipeline):
    def __init__(self, mapping=None, classifier=None):
        self.mapping = mapping
        self.classifier = classifier
        self.mapping_list = list(mapping)

    def call(self, datasets=None, **kwargs):
        def decode(sample):
            y_onehot_0 = torch.zeros(len(self.mapping))

            for x in sample["classes"]:
                if x in self.mapping:
                    y_onehot_0.scatter_(0, torch.tensor(self.mapping_list.index(x)), 1)

            if "additional" in sample:
                return {"image_data": sample["image_data"], "additional": sample["additional"], "target": y_onehot_0}
            return {"image_data": sample["image_data"], "target": y_onehot_0}

        return MapDataset(datasets, map_fn=decode)


@DatasetsManager.export("iconclass_flatten")
class IconclassFlattenDataloader(IconclassDataloader):
    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)
        if args is not None:
            dict_args = vars(args)
            dict_args.update(kwargs)
        else:
            dict_args = kwargs

        self.mapping_path = dict_args.get("mapping_path", None)
        self.classifier_path = dict_args.get("classifier_path", None)
        self.filter_label_by_count = dict_args.get("filter_label_by_count", None)

        self.mapping = {}
        mm = {}
        if self.mapping_path is not None:
            mm = read_jsonl(self.mapping_path, dict_key="id")
            for k, v in mm.items():
                if v["count"] > self.filter_label_by_count:
                    self.mapping[k] = v

        self.classifier = {}
        if self.classifier_path is not None:
            self.classifier = read_jsonl(self.classifier_path)

    def train_mapping_pipeline(self):
        return IconclassFlattenDecoderPipeline(mapping=self.mapping, classifier=self.classifier)

    def val_mapping_pipeline(self):
        return IconclassFlattenDecoderPipeline(mapping=self.mapping, classifier=self.classifier)

    @classmethod
    def add_args(cls, parent_parser):
        parent_parser = super().add_args(parent_parser)
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")

        parser.add_argument("--mapping_path", type=str)
        parser.add_argument("--classifier_path", type=str)
        parser.add_argument("--filter_label_by_count", type=int, default=0)
        return parser
