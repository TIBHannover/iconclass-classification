import argparse
import logging
import random

import numpy as np
import torch

from .datasets import DatasetsManager
from .pipeline import (
    Pipeline,
    MapDataset,
)
from .utils import read_line_data
from .iconclass import IconclassDataloader


def build_level_map(mapping):
    level_map = torch.zeros(len(mapping), dtype=torch.int64)
    for m in mapping:
        level_map[m["index"]] = len(m["parents"])

    return level_map


class IconclassAllDecoderPipeline(Pipeline):
    def __init__(
        self,
        annotation=None,
        mapping=None,
        classifier=None,
        ontology=None,
        random_trace=None,
        merge_one_hot=None,
        filter_label_by_count=None,
        max_traces=None,
    ):
        self.annotation = annotation
        self.mapping = mapping
        self.classifier = classifier
        self.ontology = ontology
        self.random_trace = random_trace
        self.merge_one_hot = merge_one_hot
        self.filter_label_by_count = filter_label_by_count
        self.max_traces = max_traces

        self.classes_vec_max_length = max([len(x["tokenizer"]) for x in self.ontology])
        self.pad_id = self.ontology[0]["tokenizer"].index("#PAD")
        self.start_id = self.ontology[0]["tokenizer"].index("#START")

        logging.info(f"IconclassAll::__init__ -> Len mapping {len(self.mapping)}")

        logging.info(f"IconclassAll::__init__ -> Build level map")
        self.level_map = build_level_map(self.mapping.values())

        logging.info(f"IconclassAll::__init__ -> Build classifier map")
        self.classifier_map = {}
        for c in self.classifier:
            self.classifier_map[c["name"]] = c

    def build_flat_target(self, sample):
        y_onehot_flat = torch.zeros(len(self.mapping))

        for c in sample["classes"]:
            if c in self.mapping:
                y_onehot_flat.scatter_(0, torch.tensor(self.mapping[c]["index"]), 1)
        return {"flat_target": y_onehot_flat}

    def build_yolo_target(self, sample):
        result = {}
        y_onehot_yolo_labels = torch.zeros(len(self.mapping))
        for x in sample["all_ids"]:
            y_onehot_yolo_labels.scatter_(0, torch.tensor(x), 1)
        result["yolo_target"] = y_onehot_yolo_labels

        y_onehot_yolo_classes = torch.zeros(len(self.classifier))
        for x in sample["all_cls_ids"]:
            y_onehot_yolo_classes.scatter_(0, torch.tensor(x), 1)
        result["yolo_classes"] = y_onehot_yolo_classes

        y_onehot_yolo_labels_weights = torch.zeros(len(self.mapping))
        for x in sample["all_cls_ids"]:
            for y in range(self.classifier[x]["range"][0], self.classifier[x]["range"][1]):
                # print(y)
                y_onehot_yolo_labels_weights.scatter_(0, torch.tensor(y), 1)
        result["yolo_target_mask"] = y_onehot_yolo_labels_weights

        return result

    def build_ontology_target(self, sample):
        ontology_target = []
        ontology_mask = []
        ontology_trace_mask = []
        ontology_ranges = []
        ontology_indexes = []
        for c in sample["classes"]:
            token_id_sequence = self.mapping[c]["token_id_sequence"]

            token_sequence = self.mapping[c]["parents"] + [c]
            if self.filter_label_by_count is not None and self.filter_label_by_count > 0:
                counts = [self.mapping[x]["count"]["yolo"] for x in token_sequence]
                ignore_labels = [False if x > self.filter_label_by_count else True for x in counts]
            else:
                ignore_labels = [False for x in token_sequence]

            # build vectors and masks
            parents = [None] + self.mapping[c]["parents"]
            ranges = []
            one_hot = torch.zeros(len(self.mapping))
            mask = torch.zeros(len(self.mapping))
            for i, p in enumerate(parents):
                if not ignore_labels[i]:
                    classifier = self.classifier_map[p]
                    ranges.append([classifier["range"][0], classifier["range"][1]])
                    mask[classifier["range"][0] : classifier["range"][1]] = 1
                    one_hot[self.mapping[token_sequence[i]]["index"]] = 1

            for _ in range(len(ranges), len(self.ontology)):
                ranges.append([0, 0])

            ontology_mask.append(mask)
            ontology_target.append(one_hot)
            ontology_ranges.append(torch.tensor(ranges, dtype=torch.int32))
            ontology_trace_mask.append(torch.ones([], dtype=torch.int32))

            # build indexes sequence
            indexes = []
            for i, t in enumerate(token_sequence):
                if not ignore_labels[i]:
                    indexes.append(self.mapping[t]["index"])

            for _ in range(len(indexes), len(self.ontology)):
                indexes.append(-1)
            ontology_indexes.append(torch.tensor(indexes, dtype=torch.int32))

        # add ones form other traces
        if self.merge_one_hot:
            for i, m in enumerate(ontology_mask):
                target = ontology_target[i]
                for j, m2 in enumerate(ontology_mask):
                    target = target + m * ontology_mask[j] * ontology_target[j]
                target[target > 0] = 1
                ontology_target[i] = target

        # randomly shuffle everything
        if self.random_trace:
            ontology_mask, ontology_target, ontology_indexes, ontology_ranges, ontology_trace_mask = zip(
                *random.sample(
                    list(zip(ontology_mask, ontology_target, ontology_indexes, ontology_ranges, ontology_trace_mask)),
                    k=len(ontology_mask),
                )
            )

        if self.max_traces is not None and self.max_traces > 0:
            ontology_mask = ontology_mask[: self.max_traces]
            ontology_target = ontology_target[: self.max_traces]
            ontology_indexes = ontology_indexes[: self.max_traces]
            ontology_ranges = ontology_ranges[: self.max_traces]
            ontology_trace_mask = ontology_trace_mask[: self.max_traces]
        return {
            "ontology_mask": torch.stack(ontology_mask, dim=0),
            "ontology_target": torch.stack(ontology_target, dim=0),
            "ontology_indexes": torch.stack(ontology_indexes, dim=0),
            "ontology_ranges": torch.stack(ontology_ranges, dim=0),
            "ontology_trace_mask": torch.stack(ontology_trace_mask, dim=0),
            "ontology_levels": self.level_map,
        }

    def call(self, datasets=None, **kwargs):
        def decode(sample):
            out_sample = {
                "id": sample["id"],
                "image_data": sample["image_data"],
            }

            ####
            # build flat vec
            flat_target = self.build_flat_target(sample)
            out_sample.update(flat_target)

            ####
            # build yolo vec (all parents are anotated)
            yolo_target = self.build_yolo_target(sample)
            out_sample.update(yolo_target)

            ####
            # build new ontology vec
            ontology_target = self.build_ontology_target(sample)
            out_sample.update(ontology_target)

            if "additional" in sample:
                out_sample["additional"] = sample["additional"]

            return out_sample

        return MapDataset(datasets, map_fn=decode)


@DatasetsManager.export("iconclass_all")
class IconclassAllDataloader(IconclassDataloader):
    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)
        if args is not None:
            dict_args = vars(args)
            dict_args.update(kwargs)
        else:
            dict_args = kwargs

        self.mapping_path = dict_args.get("mapping_path", None)
        self.classifier_path = dict_args.get("classifier_path", None)
        self.ontology_path = dict_args.get("ontology_path", None)
        self.filter_label_by_count = dict_args.get("filter_label_by_count", None)

        self.mapping = {}
        if self.mapping_path is not None:
            self.mapping = read_line_data(self.mapping_path, dict_key="id")

        self.classifier = []
        if self.classifier_path is not None:
            self.classifier = read_line_data(self.classifier_path)

        self.ontology = []
        if self.ontology_path is not None:
            self.ontology = read_line_data(self.ontology_path)

        self.max_traces = dict_args.get("max_traces", None)

        self.train_random_trace = dict_args.get("train_random_trace", None)
        self.train_merge_one_hot = dict_args.get("train_merge_one_hot", None)

        self.pad_id = None
        for level in self.ontology:
            if self.pad_id is None:
                self.pad_id = level["tokenizer"].index("#PAD")

            if self.pad_id != level["tokenizer"].index("#PAD"):
                assert False, "#PAD should always have the same index"

    def train_mapping_pipeline(self):
        return IconclassAllDecoderPipeline(
            mapping=self.mapping,
            classifier=self.classifier,
            ontology=self.ontology,
            random_trace=self.train_random_trace,
            merge_one_hot=self.train_merge_one_hot,
            filter_label_by_count=self.filter_label_by_count,
            max_traces=self.max_traces,
        )

    def val_mapping_pipeline(self):
        return IconclassAllDecoderPipeline(
            mapping=self.mapping,
            classifier=self.classifier,
            ontology=self.ontology,
            merge_one_hot=self.train_merge_one_hot,
            filter_label_by_count=self.filter_label_by_count,
            max_traces=self.max_traces,
        )

    def test_mapping_pipeline(self):
        return IconclassAllDecoderPipeline(
            mapping=self.mapping,
            classifier=self.classifier,
            ontology=self.ontology,
            merge_one_hot=self.train_merge_one_hot,
            filter_label_by_count=self.filter_label_by_count,
            max_traces=None,
        )

    @classmethod
    def add_args(cls, parent_parser):
        parent_parser = super().add_args(parent_parser)
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")

        parser.add_argument("--mapping_path", type=str)
        parser.add_argument("--classifier_path", type=str)
        parser.add_argument("--ontology_path", type=str)
        parser.add_argument("--filter_label_by_count", type=int, default=0)

        parser.add_argument("--max_traces", type=int)

        parser.add_argument("--train_random_trace", action="store_true")
        parser.add_argument("--train_merge_one_hot", action="store_true")

        return parser
