import argparse
import logging
import random

import numpy as np
import torch
import torchvision


from PIL import Image

from datasets.datasets import DatasetsManager
from datasets.pipeline import (
    Pipeline,
    MapDataset,
)

from datasets.utils import read_jsonl

# from models.utils import build_level_map
from datasets.image_pipeline import (
    CocoImageTrainPreprocessingPipeline,
    ImageDecodePipeline,
    RandomResize,
    CocoImageTestPreprocessingPipeline,
)
from datasets.datasets import DatasetsManager
from datasets.pipeline import (
    Pipeline,
    MapDataset,
    MsgPackPipeline,
    SequencePipeline,
    ConcatShufflePipeline,
    ConcatPipeline,
    DummyPipeline,
    ImagePipeline,
    split_chunk_by_nodes,
    split_chunk_by_workers,
)
from datasets.pad_collate import PadCollate


def build_level_map(mapping):
    level_map = torch.zeros(len(mapping), dtype=torch.int64)
    for m in mapping:
        level_map[m["index"]] = len(m["parents"])

    return level_map


class CocoAllDecoderPipeline(Pipeline):
    def __init__(
        self,
        annotation=None,
        mapping=None,
        classifier=None,
        ontology=None,
        random_trace=None,
        last_trace=None,
        merge_one_hot=None,
        pad_max_shape=None,
        filter_label_by_count=None,
        max_traces=None,
    ):
        self.annotation = annotation
        self.mapping = mapping
        self.classifier = classifier
        self.ontology = ontology
        self.random_trace = random_trace
        self.last_trace = last_trace
        self.merge_one_hot = merge_one_hot
        self.pad_max_shape = pad_max_shape
        self.filter_label_by_count = filter_label_by_count
        self.max_traces = max_traces

        self.classes_vec_max_length = max([len(x["tokenizer"]) for x in self.ontology])
        self.pad_id = 0
        self.start_id = 1

        logging.info(f"CocoAll::__init__ -> Len mapping {len(self.mapping)}")

        logging.info(f"CocoAll::__init__ -> Build level map")
        self.level_map = build_level_map(self.mapping.values())

        logging.info(f"CocoAll::__init__ -> Build classifier map")
        self.classifier_map = {}
        self.cls_ids_map = {}
        for c in self.classifier:
            self.classifier_map[c["id"]] = c
            for x in range(c["range"][0], c["range"][1]):

                self.cls_ids_map[x] = c["index"]

        self.yolo_map = {}
        for k, x in self.mapping.items():
            # self.yolo_map
            all_ids = [x["index"]]
            all_cls_ids = [self.cls_ids_map[x["index"]]]
            parent = x["parent"]
            while parent is not None:
                all_ids.append(self.mapping[parent]["index"])
                all_cls_ids.append(self.cls_ids_map[self.mapping[parent]["index"]])
                parent = self.mapping[parent]["parent"]

            self.yolo_map[k] = {"all_ids": all_ids, "all_cls_ids": all_cls_ids}

    def build_flat_target(self, sample):
        y_onehot_flat = torch.zeros(len(self.mapping))
        for c in sample["classes"]:
            if c in self.mapping:
                y_onehot_flat.scatter_(0, torch.tensor(self.mapping[c]["index"]), 1)
        return {"flat_target": y_onehot_flat}

    def build_yolo_target(self, sample):
        # print(sample["classes"])
        all_ids = []
        all_cls_ids = []
        for c in sample["classes"]:
            k = self.yolo_map[c]
            all_ids.extend(k["all_ids"])
            all_cls_ids.extend(k["all_cls_ids"])

        result = {}
        y_onehot_yolo_labels = torch.zeros(len(self.mapping))
        for x in set(all_ids):
            y_onehot_yolo_labels.scatter_(0, torch.tensor(x), 1)
        result["yolo_target"] = y_onehot_yolo_labels

        y_onehot_yolo_classes = torch.zeros(len(self.classifier))
        for x in set(all_cls_ids):
            y_onehot_yolo_classes.scatter_(0, torch.tensor(x), 1)
        result["yolo_classes"] = y_onehot_yolo_classes

        y_onehot_yolo_labels_weights = torch.zeros(len(self.mapping))
        for x in set(all_cls_ids):
            for y in range(self.classifier[x]["range"][0], self.classifier[x]["range"][1]):
                # print(y)
                y_onehot_yolo_labels_weights.scatter_(0, torch.tensor(y), 1)
        result["yolo_target_mask"] = y_onehot_yolo_labels_weights

        return result

    def build_new_ontology_target(self, sample):
        ontology_target = []
        ontology_mask = []
        ontology_trace_mask = []
        ontology_ranges = []
        ontology_indexes = []
        for c in sample["classes"]:
            #            token_id_sequence = self.mapping[c]["token_id_sequence"]

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

    def build_ontology_target(self, sample):
        result = {}
        classes_vec = []
        classes_sequences = []
        parents = []
        for c in sample["classes"]:
            class_vec = []
            sequences = []
            token_id_sequence = self.mapping[c]["token_id_sequence"]

            token_sequence = self.mapping[c]["parents"] + [c]
            if self.filter_label_by_count is not None and self.filter_label_by_count > 0:
                counts = [self.mapping[x]["count"]["yolo"] for x in token_sequence]
                ignore_labels = [False if x > self.filter_label_by_count else True for x in counts]
            else:
                ignore_labels = [False for x in token_sequence]

            for l, level in enumerate(self.ontology):

                one_hot = np.zeros([len(level["tokenizer"])])
                if l < len(token_id_sequence) and not ignore_labels[l]:
                    one_hot[token_id_sequence[l]] = 1
                else:
                    one_hot[level["tokenizer"].index("#PAD")] = 1
                class_vec.append(one_hot)

                if l < len(token_id_sequence) and not ignore_labels[l]:
                    sequences.append(token_id_sequence[l])
                else:
                    sequences.append(level["tokenizer"].index("#PAD"))

            parents_sequence = []
            for l, level in enumerate(self.ontology):
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
            result.update(
                {
                    "id_sequence": id_sequence,
                    "source_id_sequence": [self.start_id] + id_sequence[:-1],
                    "target_vec": target_vec,
                    "parents": parents[trace_index],
                }
            )
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
            result.update(
                {
                    "id_sequence": id_sequence,
                    "source_id_sequence": [self.start_id] + id_sequence[:-1],
                    "target_vec": target_vec,
                    "parents": parents[trace_index],
                }
            )
        elif self.pad_max_shape:
            id_sequence_list = []
            target_vec_list = []
            mask = []
            classes_vec_padded = []
            for i, c in enumerate(classes_vec):
                classes_vec_padded_inter = []
                for j, k in enumerate(c):
                    padded_classes = np.pad(k, [0, self.classes_vec_max_length - len(k)], constant_values=self.pad_id)
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

            result.update(
                {
                    "id_sequence": torch.tensor(np.asarray(id_sequence_list, dtype=np.int32)),
                    "source_id_sequence": [self.start_id] + id_sequence[:-1],
                    "target_vec": torch.tensor(np.asarray(target_vec_list, dtype=np.float32)),
                    "mask": torch.tensor(np.asarray(mask, dtype=np.int8)),
                    "parents": parents,
                }
            )

        return result

    def call(self, datasets=None, **kwargs):
        def decode(sample):
            out_sample = {"id": sample["id"], "image": sample["image"]}

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
            new_ontology_target = self.build_new_ontology_target(sample)
            out_sample.update(new_ontology_target)

            ####
            # build ontology vec
            # ontology_target = self.build_ontology_target(sample)
            # out_sample.update(ontology_target)

            if "additional" in sample:
                out_sample["additional"] = sample["additional"]

            return out_sample

        return MapDataset(datasets, map_fn=decode)


class CocoDecoderPipeline(Pipeline):
    def __init__(self, num_classes=79, annotation=None):
        self.num_classes = num_classes
        self.annotation = annotation

    def call(self, datasets=None, **kwargs):
        def decode(sample):
            out_sample = {
                "image_data": sample[b"image"],
                "id": sample[b"id"],
                "path": sample[b"path"].decode("utf-8"),
            }

            if out_sample["id"] not in self.annotation:
                logging.info(f"Dataset: {out_sample['id']} not in annotation")
                return None
            else:
                anno = self.annotation[out_sample["id"]]

            return {**out_sample, **anno}

        return MapDataset(datasets, map_fn=decode)


@DatasetsManager.export("coco_all")
class CocoDataloader:
    def __init__(self, args=None, **kwargs):
        if args is not None:
            dict_args = vars(args)
            dict_args.update(kwargs)
        else:
            dict_args = kwargs

        self.use_center_crop = dict_args.get("use_center_crop", False)

        self.train_path = dict_args.get("train_path", None)
        self.train_annotation_path = dict_args.get("train_annotation_path", None)

        self.train_filter_min_dim = dict_args.get("train_filter_min_dim", None)
        self.train_sample_additional = dict_args.get("train_sample_additional", None)

        self.train_random_sizes = dict_args.get("train_random_sizes", None)
        self.max_size = dict_args.get("max_size", 800)

        self.val_path = [dict_args.get("val_path", None)]
        self.val_annotation_path = dict_args.get("val_annotation_path", None)
        self.val_filter_min_dim = dict_args.get("val_filter_min_dim", None)
        self.val_size = dict_args.get("val_size", None)

        self.test_path = [dict_args.get("test_path", None)]
        self.test_annotation_path = dict_args.get("test_annotation_path", None)
        self.test_filter_min_dim = dict_args.get("test_filter_min_dim", None)
        self.test_size = dict_args.get("test_size", None)

        self.infer_path = [dict_args.get("infer_path", None)]
        self.infer_size = [dict_args.get("infer_size", None)]

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

        self.test_annotation = {}
        if self.test_annotation_path is not None:
            # for path in self.test_annotation_path:
            self.test_annotation.update(read_jsonl(self.test_annotation_path, dict_key="id"))

        self.mapping_path = dict_args.get("mapping_path", None)
        self.classifier_path = dict_args.get("classifier_path", None)
        self.ontology_path = dict_args.get("ontology_path", None)
        self.filter_label_by_count = dict_args.get("filter_label_by_count", None)

        self.mapping = {}
        if self.mapping_path is not None:
            self.mapping = read_jsonl(self.mapping_path, dict_key="id")

        self.classifier = []
        if self.classifier_path is not None:
            self.classifier = read_jsonl(self.classifier_path)

        self.ontology = []
        if self.ontology_path is not None:
            self.ontology = read_jsonl(self.ontology_path)

        self.max_traces = dict_args.get("max_traces", None)

        self.train_random_trace = dict_args.get("train_random_trace", None)
        self.train_merge_one_hot = dict_args.get("train_merge_one_hot", None)

        self.val_last_trace = dict_args.get("val_last_trace", None)
        self.val_pad_max_shape = dict_args.get("val_pad_max_shape", None)

        self.test_last_trace = dict_args.get("test_last_trace", None)
        self.test_pad_max_shape = dict_args.get("test_pad_max_shape", None)

        self.pad_id = None
        for level in self.ontology:
            if self.pad_id is None:
                self.pad_id = 0

            if self.pad_id != 0:
                assert False, "#PAD should always have the same index"

    def train(self):
        pipeline = SequencePipeline(
            [
                ConcatShufflePipeline([MsgPackPipeline(path=p) for p in self.train_path]),
                CocoDecoderPipeline(annotation=self.train_annotation),
                CocoImageTrainPreprocessingPipeline(),
                CocoAllDecoderPipeline(
                    mapping=self.mapping,
                    classifier=self.classifier,
                    ontology=self.ontology,
                    random_trace=self.train_random_trace,
                    merge_one_hot=self.train_merge_one_hot,
                    max_traces=self.max_traces,
                ),
            ]
        )

        dataloader = torch.utils.data.DataLoader(
            pipeline(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=PadCollate(
                pad_values={
                    "image": 0.0,
                    "image_mask": False,
                    "parents": "#PAD",
                    "ontology_mask": 0,
                    "ontology_target": 0,
                    "ontology_ranges": 0,
                    "ontology_trace_mask": 0,
                    "ontology_indexes": -1,
                }
            ),
        )
        return dataloader

    def val(self):
        pipeline = SequencePipeline(
            [
                ConcatPipeline([MsgPackPipeline(path=p, shuffle=False) for p in self.val_path]),
                CocoDecoderPipeline(annotation=self.val_annotation),
                CocoImageTestPreprocessingPipeline(),
                CocoAllDecoderPipeline(
                    mapping=self.mapping,
                    classifier=self.classifier,
                    ontology=self.ontology,
                    random_trace=self.train_random_trace,
                    merge_one_hot=self.train_merge_one_hot,
                    max_traces=self.max_traces,
                ),
            ]
        )

        dataloader = torch.utils.data.DataLoader(
            pipeline(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=PadCollate(
                pad_values={
                    "image": 0.0,
                    "image_mask": False,
                    "parents": "#PAD",
                    "ontology_mask": 0,
                    "ontology_target": 0,
                    "ontology_ranges": 0,
                    "ontology_trace_mask": 0,
                    "ontology_indexes": -1,
                }
            ),
        )
        return dataloader

    def test_image_pipeline(self):
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                RandomResize([self.test_size], max_size=self.max_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        return IconclassImagePreprocessingPipeline(
            transforms,
            min_size=self.val_filter_min_dim,
        )

    def test_decode_pieline(self):
        return SequencePipeline(
            [
                ConcatPipeline([MsgPackPipeline(path=p, shuffle=False) for p in self.test_path]),
                CocoDecoderPipeline(annotation=self.test_annotation),
            ]
        )

    def test(self):
        pipeline = SequencePipeline(
            [
                self.test_decode_pieline(),
                CocoAllDecoderPipeline(
                    mapping=self.mapping,
                    classifier=self.classifier,
                    ontology=self.ontology,
                    last_trace=self.test_last_trace,
                    pad_max_shape=self.test_pad_max_shape,
                    max_traces=self.max_traces,
                ),
                self.test_image_pipeline(),
            ]
        )

        dataloader = torch.utils.data.DataLoader(
            pipeline(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=PadCollate(pad_values={"image": 0.0, "image_mask": False, "parents": "#PAD"}),
        )
        return dataloader

    def infer_image_pipeline(self):
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize(size=256),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        return ImageDecodePipeline(transforms)

    def infer(self):

        pipeline = SequencePipeline([ImagePipeline(self.infer_path), self.infer_image_pipeline()])

        dataloader = torch.utils.data.DataLoader(
            pipeline(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=PadCollate(pad_values={"image": 0.0, "image_mask": False}),
        )
        return dataloader

    @classmethod
    def add_args(cls, parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--use_center_crop", action="store_true", help="verbose output")

        parser.add_argument("--train_path", nargs="+", type=str)
        parser.add_argument("--train_annotation_path", nargs="+", type=str)
        parser.add_argument("--train_filter_min_dim", type=int, default=128, help="delete images with smaller size")
        parser.add_argument("--train_sample_additional", type=float)

        parser.add_argument("--train_random_sizes", type=int, nargs="+", default=[480, 512, 544, 576, 608, 640])
        parser.add_argument("--max_size", type=int, default=800)

        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--batch_size", type=int, default=8)

        parser.add_argument("--val_path", type=str)
        parser.add_argument("--val_annotation_path", type=str)
        parser.add_argument("--val_filter_min_dim", type=int, default=128, help="delete images with smaller size")
        parser.add_argument("--val_size", type=int, default=224)

        parser.add_argument("--test_path", type=str)
        parser.add_argument("--test_annotation_path", type=str)
        parser.add_argument("--test_filter_min_dim", type=int, default=128, help="delete images with smaller size")
        parser.add_argument("--test_size", type=int, default=640)

        parser.add_argument("--infer_path", type=str)
        parser.add_argument("--infer_size", type=int, default=640)

        parser.add_argument("--mapping_path", type=str)
        parser.add_argument("--classifier_path", type=str)
        parser.add_argument("--ontology_path", type=str)
        parser.add_argument("--filter_label_by_count", type=int, default=0)

        parser.add_argument("--max_traces", type=int)

        parser.add_argument("--train_random_trace", action="store_true")
        parser.add_argument("--train_merge_one_hot", action="store_true")

        parser.add_argument("--val_last_trace", action="store_true")
        parser.add_argument("--val_pad_max_shape", action="store_true")

        parser.add_argument("--test_last_trace", action="store_true")
        parser.add_argument("--test_pad_max_shape", action="store_true")
        return parser
