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


class IconclassSequenceDecoderPipeline(Pipeline):
    def __init__(
        self,
        num_classes=79,
        annotation=None,
        mapping=None,
        classifier=None,
        random_trace=None,
        last_trace=None,
        merge_one_hot=None,
        pad_max_shape=None,
    ):
        self.num_classes = num_classes
        self.annotation = annotation
        self.mapping = mapping
        self.classifier = classifier
        self.random_trace = random_trace
        self.last_trace = last_trace
        self.merge_one_hot = merge_one_hot
        self.pad_max_shape = pad_max_shape

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
                sample.update(self.annotation[sample["id"]])

            classes_vec_max_length = max([len(x["tokenizer"]) for x in self.classifier])
            pad_id = self.classifier[0]["tokenizer"].index("#PAD")

            classes_vec = []
            classes_sequences = []
            parents = []
            for c in sample["classes"]:
                class_vec = []
                sequences = []
                token_sequence = self.mapping[c]["token_id_sequence"]
                for l, classifier in enumerate(self.classifier):
                    one_hot = np.zeros([len(classifier["tokenizer"])])
                    if l < len(token_sequence):
                        one_hot[token_sequence[l]] = 1
                    class_vec.append(one_hot)

                    if l < len(token_sequence):
                        sequences.append(token_sequence[l])
                    else:
                        sequences.append(classifier["tokenizer"].index("#PAD"))

                parents_sequence = []
                for l, classifier in enumerate(self.classifier):
                    if l < len(self.mapping[c]["parents"]):

                        parents_sequence.append(self.mapping[c]["parents"][l])
                    else:
                        parents_sequence.append("#PAD")

                parents.append(parents_sequence)

                # print(parents_sequence)
                # print(parents)
                # exit()
                classes_vec.append(class_vec)
                classes_sequences.append(sequences)

            if self.random_trace:
                trace_index = random.randint(0, len(sample["classes"]) - 1)
                source_id_sequence = classes_sequences[trace_index]
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
                            if seq[d - 1] == source_id_sequence[d - 1]:
                                vecs_to_merged.append(classes_vec[i][d])

                        target_vec[d] = np.amax(np.stack(vecs_to_merged), axis=0)
                sample = {
                    "image_data": sample["image_data"],
                    "id": sample["id"],
                    "source_id_sequence": source_id_sequence,
                    "target_vec": target_vec,
                    "parents": parents[trace_index],
                }
            elif self.last_trace:
                trace_index = len(sample["classes"]) - 1
                source_id_sequence = classes_sequences[trace_index]
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
                            if seq[d - 1] == source_id_sequence[d - 1]:
                                vecs_to_merged.append(classes_vec[i][d])

                        target_vec[d] = np.amax(np.stack(vecs_to_merged), axis=0)
                sample = {
                    "image_data": sample["image_data"],
                    "id": sample["id"],
                    "source_id_sequence": source_id_sequence,
                    "target_vec": target_vec,
                    "parents": parents[trace_index],
                }
            elif self.pad_max_shape:
                source_id_sequence_list = []
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
                    source_id_sequence = classes_sequences[i]
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
                                if seq[d - 1] == source_id_sequence[d - 1]:
                                    vecs_to_merged.append(classes_vec_padded[i][d])

                            target_vec[d] = np.amax(np.stack(vecs_to_merged), axis=0)
                    source_id_sequence_list.append(source_id_sequence)

                    target_vec_list.append(np.asarray(target_vec))
                    mask.append(1)

                sample = {
                    "image_data": sample["image_data"],
                    "id": sample["id"],
                    "source_id_sequence": torch.tensor(np.asarray(source_id_sequence_list, dtype=np.int64)),
                    "target_vec": torch.tensor(np.asarray(target_vec_list, dtype=np.float32)),
                    "mask": torch.tensor(np.asarray(mask, dtype=np.int8)),
                }

            return sample

        return MapDataset(datasets, map_fn=decode)


@DatasetsManager.export("iconclass_sequence")
class IconclassSequenceDataloader:
    def __init__(self, args=None, **kwargs):
        if args is not None:
            dict_args = vars(args)
            dict_args.update(kwargs)
        else:
            dict_args = kwargs

        self.train_path = dict_args.get("train_path", None)
        self.train_annotation_path = dict_args.get("train_annotation_path", None)
        self.train_random_trace = dict_args.get("train_random_trace", None)
        self.train_merge_one_hot = dict_args.get("train_merge_one_hot", None)

        self.val_path = dict_args.get("val_path", None)
        self.val_annotation_path = dict_args.get("val_annotation_path", None)
        self.val_last_trace = dict_args.get("val_last_trace", None)
        self.val_pad_max_shape = dict_args.get("val_pad_max_shape", None)

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
            for path in self.val_annotation_path:
                self.val_annotation.update(read_jsonl(path, dict_key="id"))

        self.mapping = {}
        if self.mapping_path is not None:
            self.mapping = read_jsonl(self.mapping_path, dict_key="id")

        self.classifier = {}
        if self.classifier_path is not None:
            self.classifier = read_jsonl(self.classifier_path)

        self.pad_id = None
        for classifier in self.classifier:
            if self.pad_id is None:
                self.pad_id = classifier["tokenizer"].index("#PAD")

            if self.pad_id != classifier["tokenizer"].index("#PAD"):
                assert False, "#PAD should always have the same index"

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
            IconclassSequenceDecoderPipeline(
                annotation=self.train_annotation,
                mapping=self.mapping,
                classifier=self.classifier,
                random_trace=self.train_random_trace,
                merge_one_hot=self.train_merge_one_hot,
            ),
            ImagePreprocessingPipeline(train_image_transform),
        ]

        pipeline = SequencePipeline(pipeline_stack)
        dataloader = torch.utils.data.DataLoader(
            pipeline(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=PadCollate({"mask": 100, "source_id_sequence": self.pad_id, "target_vec": self.pad_id}),
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
            IconclassSequenceDecoderPipeline(
                annotation=self.val_annotation,
                mapping=self.mapping,
                classifier=self.classifier,
                last_trace=self.val_last_trace,
                merge_one_hot=self.train_merge_one_hot,
                pad_max_shape=self.val_pad_max_shape,
            ),
            ImagePreprocessingPipeline(val_image_transform),
        ]

        pipeline = SequencePipeline(pipeline_stack)
        dataloader = torch.utils.data.DataLoader(
            pipeline(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=PadCollate({"mask": 100, "source_id_sequence": self.pad_id, "target_vec": self.pad_id}),
        )
        return dataloader

    def test(self):
        pass

    @classmethod
    def add_args(cls, parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")

        parser.add_argument("--train_path", nargs="+", type=str)
        parser.add_argument("--train_annotation_path", nargs="+", type=str)
        parser.add_argument("--train_random_trace", action="store_true")
        parser.add_argument("--train_merge_one_hot", action="store_true")

        parser.add_argument("--mapping_path", type=str)
        parser.add_argument("--classifier_path", type=str)

        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--num_workers", type=int, default=8)

        parser.add_argument("--val_path", type=str)
        parser.add_argument("--val_annotation_path", nargs="+", type=str)
        parser.add_argument("--val_last_trace", action="store_true")
        parser.add_argument("--val_pad_max_shape", action="store_true")

        return parser

