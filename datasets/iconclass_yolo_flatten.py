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
from datasets.utils import read_line_data


from datasets.iconclass import IconclassDataloader


class IconclassYoloFlattenDecoderPipeline(Pipeline):
    def __init__(self, mapping=None, classifier=None):
        self.mapping = mapping
        self.classifier = classifier

    def call(self, datasets=None, **kwargs):
        def decode(sample):

            y_onehot_0 = torch.zeros(len(self.mapping))
            for x in sample["all_ids"]:
                y_onehot_0.scatter_(0, torch.tensor(x), 1)

            y_onehot_1 = torch.zeros(len(self.classifier))
            for x in sample["all_cls_ids"]:
                y_onehot_1.scatter_(0, torch.tensor(x), 1)

            y_onehot_2 = torch.zeros(len(self.mapping))
            # print(a['all_cls_ids'])
            for x in sample["all_cls_ids"]:
                for y in range(self.classifier[x]["range"][0], self.classifier[x]["range"][1]):
                    # print(y)
                    y_onehot_2.scatter_(0, torch.tensor(y), 1)

            if "additional" in sample:
                return {
                    "image_data": sample["image_data"],
                    "additional": sample["additional"],
                    "ids_vec": y_onehot_0,
                    "cls_vec": y_onehot_1,
                    "cls_ids_mask_vec": y_onehot_2,
                }
            return {
                "image_data": sample["image_data"],
                "ids_vec": y_onehot_0,
                "cls_vec": y_onehot_1,
                "cls_ids_mask_vec": y_onehot_2,
            }

        return MapDataset(datasets, map_fn=decode)


@DatasetsManager.export("iconclass_yolo_flatten")
class IconclassYoloFlattenDataloader(IconclassDataloader):
    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)
        if args is not None:
            dict_args = vars(args)
            dict_args.update(kwargs)
        else:
            dict_args = kwargs

        self.mapping_path = dict_args.get("mapping_path", None)
        self.classifier_path = dict_args.get("classifier_path", None)

        self.mapping = {}
        if self.mapping_path is not None:
            self.mapping = read_line_data(self.mapping_path, dict_key="id")

        self.classifier = {}
        if self.classifier_path is not None:
            self.classifier = read_line_data(self.classifier_path)

    def train_mapping_pipeline(self):
        return IconclassYoloFlattenDecoderPipeline(mapping=self.mapping, classifier=self.classifier)

    def val_mapping_pipeline(self):
        return IconclassYoloFlattenDecoderPipeline(mapping=self.mapping, classifier=self.classifier)

    def test_mapping_pipeline(self):
        return IconclassYoloFlattenDecoderPipeline(mapping=self.mapping, classifier=self.classifier)

    @classmethod
    def add_args(cls, parent_parser):
        parent_parser = super().add_args(parent_parser)
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")

        parser.add_argument("--mapping_path", type=str)
        parser.add_argument("--classifier_path", type=str)
        return parser
