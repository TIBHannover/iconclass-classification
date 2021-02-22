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
    MapPipeline,
    Pipeline,
    MapDataset,
    MsgPackPipeline,
    SequencePipeline,
    ConcatShufflePipeline,
    ConcatPipeline,
)
from datasets.utils import read_jsonl

from datasets.pad_collate import PadCollate


from datasets.iconclass import IconclassDataloader


class IconclassSequenceDecoderPipeline(Pipeline):
    def __init__(
        self,
        annotation=None,
        mapping=None,
        classifier=None,
        random_trace=None,
        last_trace=None,
        merge_one_hot=None,
        pad_max_shape=None,
        filter_label_by_count=None,
    ):
        self.annotation = annotation
        self.mapping = mapping
        self.classifier = classifier
        self.random_trace = random_trace
        self.last_trace = last_trace
        self.merge_one_hot = merge_one_hot
        self.pad_max_shape = pad_max_shape
        self.filter_label_by_count = filter_label_by_count

    def call(self, datasets=None, **kwargs):
        def decode(sample):

            classes_vec_max_length = max([len(x["tokenizer"]) for x in self.classifier])
            pad_id = self.classifier[0]["tokenizer"].index("#PAD")
            start_id = self.classifier[0]["tokenizer"].index("#START")

            classes_vec = []
            classes_sequences = []
            parents = []
            for c in sample["classes"]:
                class_vec = []
                sequences = []
                token_id_sequence = self.mapping[c]["token_id_sequence"]

                token_sequence = self.mapping[c]["parents"] + [c]
                if self.filter_label_by_count is not None and self.filter_label_by_count > 0:
                    counts = [self.mapping[x]["count"] for x in token_sequence]
                    ignore_labels = [False if x > self.filter_label_by_count else True for x in counts]
                else:
                    ignore_labels = [False for x in token_sequence]

                for l, classifier in enumerate(self.classifier):

                    one_hot = np.zeros([len(classifier["tokenizer"])])
                    if l < len(token_id_sequence) and not ignore_labels[l]:
                        one_hot[token_id_sequence[l]] = 1
                    else:
                        one_hot[classifier["tokenizer"].index("#PAD")] = 1
                    class_vec.append(one_hot)

                    if l < len(token_id_sequence) and not ignore_labels[l]:
                        sequences.append(token_id_sequence[l])
                    else:
                        sequences.append(classifier["tokenizer"].index("#PAD"))

                parents_sequence = []
                for l, classifier in enumerate(self.classifier):
                    if l < len(self.mapping[c]["parents"]) and not ignore_labels[l]:

                        parents_sequence.append(self.mapping[c]["parents"][l])
                    else:
                        parents_sequence.append("#PAD")

                parents.append(parents_sequence)

                classes_vec.append(class_vec)
                classes_sequences.append(sequences)

            if self.random_trace:
                trace_index = random.randint(0, len(sample["classes"]) - 1)
                id_sequence = classes_sequences[trace_index]
                target_vec = classes_vec[trace_index]

                trace_class = sample["classes"][trace_index]

                if self.merge_one_hot:
                    target_vec[0] = np.amax(np.stack([classes_vec[i][0] for i in range(len(classes_vec))]), axis=0)

                    # merge one_hot traces until parent don't match
                    for d, vec in enumerate(target_vec):
                        if d < 1:
                            continue
                        vecs_to_merged = []
                        for i, seq in enumerate(classes_sequences):
                            if seq[d - 1] == id_sequence[d - 1]:
                                vecs_to_merged.append(classes_vec[i][d])

                        target_vec[d] = np.amax(np.stack(vecs_to_merged), axis=0)
                out_sample = {
                    "image_data": sample["image_data"],
                    "id": sample["id"],
                    "id_sequence": id_sequence,
                    "source_id_sequence": [start_id] + id_sequence[:-1],
                    "target_vec": target_vec,
                    "parents": parents[trace_index],
                }
            elif self.last_trace:
                trace_index = len(sample["classes"]) - 1
                id_sequence = classes_sequences[trace_index]
                target_vec = classes_vec[trace_index]

                trace_class = sample["classes"][trace_index]

                if self.merge_one_hot:
                    target_vec[0] = np.amax(np.stack([classes_vec[i][0] for i in range(len(classes_vec))]), axis=0)

                    # merge one_hot traces until parent don't match
                    for d, vec in enumerate(target_vec):
                        if d < 1:
                            continue
                        vecs_to_merged = []
                        for i, seq in enumerate(classes_sequences):
                            if seq[d - 1] == id_sequence[d - 1]:
                                vecs_to_merged.append(classes_vec[i][d])

                        target_vec[d] = np.amax(np.stack(vecs_to_merged), axis=0)
                out_sample = {
                    "image_data": sample["image_data"],
                    "id": sample["id"],
                    "id_sequence": id_sequence,
                    "source_id_sequence": [start_id] + id_sequence[:-1],
                    "target_vec": target_vec,
                    "parents": parents[trace_index],
                }
            elif self.pad_max_shape:
                id_sequence_list = []
                target_vec_list = []
                mask = []
                classes_vec_padded = []
                for i, c in enumerate(classes_vec):
                    classes_vec_padded_inter = []
                    for j, k in enumerate(c):
                        padded_classes = np.pad(k, [0, classes_vec_max_length - len(k)], constant_values=pad_id)
                        classes_vec_padded_inter.append(padded_classes)

                    classes_vec_padded.append(classes_vec_padded_inter)

                classes_vec_padded = np.asarray(classes_vec_padded)

                for i, trace_class in enumerate(sample["classes"]):
                    id_sequence = classes_sequences[i]
                    target_vec = classes_vec_padded[i]

                    target_vec = np.asarray(target_vec)

                    if self.merge_one_hot:
                        target_vec[0] = np.amax(
                            np.stack([classes_vec_padded[i][0] for i in range(len(classes_vec))]), axis=0
                        )

                        # merge one_hot traces until parent don't match
                        for d, vec in enumerate(classes_vec_padded[i]):
                            if d < 1:
                                continue
                            vecs_to_merged = []
                            for i, seq in enumerate(classes_sequences):
                                if seq[d - 1] == id_sequence[d - 1]:
                                    vecs_to_merged.append(classes_vec_padded[i][d])

                            target_vec[d] = np.amax(np.stack(vecs_to_merged), axis=0)
                    id_sequence_list.append(id_sequence)

                    target_vec_list.append(np.asarray(target_vec))
                    mask.append(1)

                out_sample = {
                    "image_data": sample["image_data"],
                    "id": sample["id"],
                    "id_sequence": torch.tensor(np.asarray(id_sequence_list, dtype=np.int64)),
                    "source_id_sequence": [start_id] + id_sequence[:-1],
                    "target_vec": torch.tensor(np.asarray(target_vec_list, dtype=np.float32)),
                    "mask": torch.tensor(np.asarray(mask, dtype=np.int8)),
                }

            if "additional" in sample:
                out_sample["additional"] = sample["additional"]

            return out_sample

        return MapDataset(datasets, map_fn=decode)


@DatasetsManager.export("iconclass_sequence")
class IconclassSequenceDataloader(IconclassDataloader):
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
        if self.mapping_path is not None:
            self.mapping = read_jsonl(self.mapping_path, dict_key="id")

        self.classifier = {}
        if self.classifier_path is not None:
            self.classifier = read_jsonl(self.classifier_path)

        self.train_random_trace = dict_args.get("train_random_trace", None)
        self.train_merge_one_hot = dict_args.get("train_merge_one_hot", None)

        self.val_last_trace = dict_args.get("val_last_trace", None)
        self.val_pad_max_shape = dict_args.get("val_pad_max_shape", None)

        self.test_last_trace = dict_args.get("test_last_trace", None)
        self.test_pad_max_shape = dict_args.get("test_pad_max_shape", None)

        self.pad_id = None
        for classifier in self.classifier:
            if self.pad_id is None:
                self.pad_id = classifier["tokenizer"].index("#PAD")

            if self.pad_id != classifier["tokenizer"].index("#PAD"):
                assert False, "#PAD should always have the same index"

    def train_mapping_pipeline(self):
        return IconclassSequenceDecoderPipeline(
            mapping=self.mapping,
            classifier=self.classifier,
            random_trace=self.train_random_trace,
            merge_one_hot=self.train_merge_one_hot,
            filter_label_by_count=self.filter_label_by_count,
        )

    def val_mapping_pipeline(self):
        return IconclassSequenceDecoderPipeline(
            mapping=self.mapping,
            classifier=self.classifier,
            last_trace=self.val_last_trace,
            merge_one_hot=self.train_merge_one_hot,
            pad_max_shape=self.val_pad_max_shape,
            filter_label_by_count=self.filter_label_by_count,
        )

    def test_mapping_pipeline(self):
        return IconclassSequenceDecoderPipeline(
            mapping=self.mapping,
            classifier=self.classifier,
            last_trace=self.test_last_trace,
            pad_max_shape=self.test_pad_max_shape,
            filter_label_by_count=self.filter_label_by_count,
        )

    @classmethod
    def add_args(cls, parent_parser):
        parent_parser = super().add_args(parent_parser)
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")

        parser.add_argument("--mapping_path", type=str)
        parser.add_argument("--classifier_path", type=str)
        parser.add_argument("--filter_label_by_count", type=int, default=0)

        parser.add_argument("--train_random_trace", action="store_true")
        parser.add_argument("--train_merge_one_hot", action="store_true")

        parser.add_argument("--val_last_trace", action="store_true")
        parser.add_argument("--val_pad_max_shape", action="store_true")

        parser.add_argument("--test_last_trace", action="store_true")
        parser.add_argument("--test_pad_max_shape", action="store_true")

        return parser
