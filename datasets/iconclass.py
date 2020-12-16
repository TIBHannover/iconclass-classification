import argparse
import logging
import json
import random

import numpy as np
import torch
import torchvision
import imageio

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


class IconclassDecoderPipeline(Pipeline):
    def __init__(
        self, num_classes=79, annotation=None, mapping=None, classifier=None, random_trace=None, last_trace=None, merge_one_hot=None
    ):
        self.num_classes = num_classes
        self.annotation = annotation
        self.mapping = mapping
        self.classifier = classifier
        self.random_trace = random_trace
        self.last_trace = last_trace
        self.merge_one_hot = merge_one_hot
        
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

            classes_vec = []
            classes_sequences = []
            for c in sample["classes"]:
                class_vec = []
                sequences = []
                token_sequence = self.mapping[c]["token_id_sequnce"]
                for l, classifier in enumerate(self.classifier):
                    one_hot = np.zeros([len(classifier["tokenizer"])])
                    if l < len(token_sequence):
                        one_hot[token_sequence[l]] = 1
                    class_vec.append(one_hot)

                    if l < len(token_sequence):
                        sequences.append(token_sequence[l])
                    else:
                        sequences.append(classifier["tokenizer"].index("#PAD"))

                classes_vec.append(class_vec)
                classes_sequences.append(sequences)

            if self.random_trace:
                trace_index = random.randint(0, len(sample["classes"]) - 1)
                source_id_sequnce = classes_sequences[trace_index]
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
                            if seq[d - 1] == source_id_sequnce[d - 1]:
                                vecs_to_merged.append(classes_vec[i][d])

                        target_vec[d] = np.amax(np.stack(vecs_to_merged), axis=0)
                sample = {
                    "image_data": sample["image_data"],
                    "id": sample["id"],
                    "source_id_sequnce": source_id_sequnce,
                    "target_vec": target_vec,
                }
            elif self.last_trace:
                trace_index = len(sample["classes"]) - 1
                source_id_sequnce = classes_sequences[trace_index]
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
                            if seq[d - 1] == source_id_sequnce[d - 1]:
                                vecs_to_merged.append(classes_vec[i][d])

                        target_vec[d] = np.amax(np.stack(vecs_to_merged), axis=0)
                sample = {
                    "image_data": sample["image_data"],
                    "id": sample["id"],
                    "source_id_sequnce": source_id_sequnce,
                    "target_vec": target_vec,
                }

            return sample

        return MapDataset(datasets, map_fn=decode)


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


@DatasetsManager.export("iconclass")
class IconClassDataloader:
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
            IconclassDecoderPipeline(
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
            IconclassDecoderPipeline(
                annotation=self.val_annotation, mapping=self.mapping, classifier=self.classifier,
                last_trace=self.val_last_trace,
                merge_one_hot=self.train_merge_one_hot,
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
        parser.add_argument("--train_random_trace", action="store_true")
        parser.add_argument("--train_merge_one_hot", action="store_true")

        parser.add_argument("--mapping_path", type=str)
        parser.add_argument("--classifier_path", type=str)

        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--num_workers", type=int, default=8)

        parser.add_argument("--val_path", type=str)
        parser.add_argument("--val_annotation_path",  nargs="+", type=str)
        parser.add_argument("--val_last_trace",action="store_true")
        
        return parser


class IconclassYoloFlattenDecoderPipeline(Pipeline):
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
                print(sample["id"])
                return None
            else:
                anno = self.annotation[sample["id"]]

            y_onehot_0 = torch.zeros(len(self.mapping))
            for x in anno["all_ids"]:
                y_onehot_0.scatter_(0, torch.tensor(x), 1)

            y_onehot_1 = torch.zeros(len(self.classifier))
            for x in anno["all_cls_ids"]:
                y_onehot_1.scatter_(0, torch.tensor(x), 1)

            y_onehot_2 = torch.zeros(len(self.mapping))
            # print(a['all_cls_ids'])
            for x in anno["all_cls_ids"]:
                for y in range(self.classifier[x]["range"][0], self.classifier[x]["range"][1]):
                    # print(y)
                    y_onehot_2.scatter_(0, torch.tensor(y), 1)
            return {**sample, "ids_vec": y_onehot_0, "cls_vec": y_onehot_1, "cls_ids_mask_vec": y_onehot_2}
            return sample

        return MapDataset(datasets, map_fn=decode)


@DatasetsManager.export("iconclass_yolo_flatten")
class IconClassYoloFlattenDataloader:
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
            IconclassYoloFlattenDecoderPipeline(
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
            IconclassYoloFlattenDecoderPipeline(
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
