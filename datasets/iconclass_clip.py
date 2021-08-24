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

from datasets.iconclass import IconclassDataloader
from tools import tokenizer_clip 

class IconclassCLIPDecoderPipeline(Pipeline):
    def __init__(self, tokenizer=None,mapping=None, classifier=None, labels=None):
        self.mapping = mapping
        self.classifier = classifier
        self.mapping_list = list(mapping)
        self.labels = labels
        self.tokenizer=tokenizer
        
    def call(self, datasets=None, **kwargs):
        def decode(sample):
            
            trace_index = random.randint(0, len(sample["classes"]) - 1)
            rand_trace_label = sample["classes"][trace_index]
            txt_label = self.labels.get(rand_trace_label)["txt"]

            txt_label_str = " ".join(txt_label)
            # print(txt_label)
            # txt_label = ", ".join(self.labels.get(rand_trace_label)["kw"]["en"]) + ", " + self.labels.get(rand_trace_label)["txt"]["en"]
            # if txt_label.count(" ")+1 > 77: # cut the first 77 words from a long string
            #     txt_label = txt_label.replace(", ","," )
            #     txt_label = txt_label.replace(" ", ",")
            #     txt_label = txt_label.split(",")
            #     txt_label = " ".join(txt_label)
            # print("***{} --{}--{} $$".format(sample["id"], txt_label, txt_label.count(" ")+1))
            try:
                token_lb = tokenizer_clip.tokenize(self.tokenizer, txt_label_str, truncate = True)
            except RuntimeError:
                print(txt_label_str)
                print(len(txt_label))
                print(rand_trace_label)
                print(sample["classes"])
            # print(token_lb.shape)
            token_lb = torch.squeeze(token_lb, dim=0)
            if "additional" in sample:
                return {"image_data": sample["image_data"], "additional": sample["additional"], "txt_label": txt_label_str}
            return {"image_data": sample["image_data"], "txt_label": txt_label_str, "txt_mask": [1], "token_lb" : token_lb}

        return MapDataset(datasets, map_fn=decode)


@DatasetsManager.export("iconclass_clip")
class IconclassCLIPDataloader(IconclassDataloader):
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
        self.label_path = dict_args.get("label_path", None)
        
        
        self.tokenizer = tokenizer_clip._Tokenizer(args)
        
        if self.label_path is not None:
            self.labels = read_jsonl(self.label_path, dict_key="id")
            
        
        self.mapping = {}
        mm = {}
        if self.mapping_path is not None:
            mm = read_jsonl(self.mapping_path, dict_key="id")
            for k,v in mm.items():
                if v['count'] >self.filter_label_by_count:
                    self.mapping[k] = v
            
        self.classifier = {}
        if self.classifier_path is not None:
            self.classifier = read_jsonl(self.classifier_path)

    def train_mapping_pipeline(self):
        return IconclassCLIPDecoderPipeline(tokenizer= self.tokenizer, mapping=self.mapping, classifier=self.classifier, labels = self.labels)

    def val_mapping_pipeline(self):
        return IconclassCLIPDecoderPipeline(tokenizer= self.tokenizer,mapping=self.mapping, classifier=self.classifier, labels = self.labels)

    @classmethod
    def add_args(cls, parent_parser):
        parent_parser = super().add_args(parent_parser)
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")

        parser.add_argument("--mapping_path", type=str)
        parser.add_argument("--classifier_path", type=str)
        parser.add_argument("--filter_label_by_count", type=int, default=0)
        parser.add_argument("--label_path", type=str)
        return parser
