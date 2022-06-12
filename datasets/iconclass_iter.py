import argparse
import logging
import random
import imageio
from datasets.pad_collate import PadCollate
import torchvision

import numpy as np
import torch

from torchdata.datapipes.iter import FileLister, FileOpener, IterDataPipe
from torchdata.datapipes import functional_datapipe

from datasets.datasets import DatasetsManager
from datasets.pipeline import (
    Pipeline,
    MapDataset,
)
from datasets.utils import read_dict_data, read_line_data
from datasets.iconclass import IconclassDataloader

from typing import Dict, Iterator, Union, Tuple, Any

from datasets.image_pipeline import RandomResize

from utils import get_node_rank


def build_level_map(mapping):
    level_map = torch.zeros(len(mapping), dtype=torch.int64)
    for m in mapping:
        level_map[m["index"]] = len(m["parents"])

    return level_map


def worker_init_fn(worker_id):
    seed = worker_id * 1000 + int(get_node_rank())

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    info = torch.utils.data.get_worker_info()
    num_workers = info.num_workers
    datapipe = info.dataset
    torch.utils.data.graph_settings.apply_sharding(datapipe, num_workers, worker_id)


@DatasetsManager.export("iconclass_iter")
class IconclassIterDataloader:
    def __init__(self, args=None, **kwargs):
        # super().__init__(args, **kwargs)
        if args is not None:
            dict_args = vars(args)
            dict_args.update(kwargs)
        else:
            dict_args = kwargs

        self.targets = dict_args.get("targets", set())
        self.labels_path = dict_args.get("labels_path", None)

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

        self.mapping_path = dict_args.get("mapping_path", None)
        self.classifier_path = dict_args.get("classifier_path", None)
        self.ontology_path = dict_args.get("ontology_path", None)

        self.filter_label_by_count = dict_args.get("filter_label_by_count", None)

        self.labels = {}
        if self.labels_path is not None:
            self.labels = read_dict_data(self.labels_path)

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

        self.train_annotation = {}
        if self.train_annotation_path is not None:
            for path in self.train_annotation_path:
                self.train_annotation.update(read_line_data(path, dict_key="id"))

        self.val_annotation = {}
        if self.val_annotation_path is not None:
            # for path in self.val_annotation_path:
            self.val_annotation.update(read_line_data(self.val_annotation_path, dict_key="id"))

        self.test_annotation = {}
        if self.test_annotation_path is not None:
            # for path in self.test_annotation_path:
            self.test_annotation.update(read_line_data(self.test_annotation_path, dict_key="id"))

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

    def train(self):
        dp = FileLister(root=self.train_path, recursive=True, masks="*.msg")
        dp = dp.cycle()
        dp = dp.shuffle(buffer_size=128)
        dp = dp.load_from_msg()
        dp = dp.shuffle(buffer_size=2048)
        dp = dp.sharding_filter()
        dp = dp.decode_iconclass(self.train_annotation)

        if "flat" in self.targets:
            dp = dp.build_flat_target(self.mapping)

        if "yolo" in self.targets:
            dp = dp.build_yolo_target(self.mapping, self.classifier)

        if "onto" in self.targets:
            dp = dp.build_ontology_target(
                mapping=self.mapping,
                classifier=self.classifier,
                filter_label_by_count=self.filter_label_by_count,
                ontology=self.ontology,
                classifier_map=self.classifier_map,
                merge_one_hot=self.train_merge_one_hot,
                random_trace=self.train_random_trace,
                max_traces=self.max_traces,
                level_map=self.level_map,
            )

        if "clip" in self.targets:
            dp = dp.iconclass_text(labels=self.labels, shuffle=True)
            dp = dp.tokenize_openclip()
        dp = dp.augment_strong_image()
        dp = dp.clean_sample(
            [
                # "name",
                # "id",
                "image_data",
                # "path",
                # "rel_path",
                "classes",
                "ids",
                "all_ids",
                "cls_ids",
                "all_cls_ids",
            ]
        )
        dp = dp.batch(self.batch_size, drop_last=True)
        dp = dp.collate(
            PadCollate(
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
            )
        )

        return torch.utils.data.DataLoader(
            dp,
            batch_size=None,
            num_workers=self.num_workers,
            shuffle=True,
            worker_init_fn=worker_init_fn,
            pin_memory=True,
        )

    def val(self):
        dp = FileLister(root=self.val_path, recursive=True, masks="*.msg")
        dp = dp.load_from_msg()
        dp = dp.sharding_filter()
        dp = dp.decode_iconclass(self.val_annotation)
        if "flat" in self.targets:
            dp = dp.build_flat_target(self.mapping)
        if "yolo" in self.targets:
            dp = dp.build_yolo_target(self.mapping, self.classifier)
        if "onto" in self.targets:
            dp = dp.build_ontology_target(
                mapping=self.mapping,
                classifier=self.classifier,
                filter_label_by_count=self.filter_label_by_count,
                ontology=self.ontology,
                classifier_map=self.classifier_map,
                merge_one_hot=self.train_merge_one_hot,
                random_trace=False,
                max_traces=self.max_traces,
                level_map=self.level_map,
            )
        dp = dp.val_image()
        dp = dp.clean_sample(
            [
                "name",
                "id",
                "image_data",
                "path",
                "rel_path",
                "classes",
                "ids",
                "all_ids",
                "cls_ids",
                "all_cls_ids",
            ]
        )
        dp = dp.batch(self.batch_size, drop_last=True)
        dp = dp.collate(
            PadCollate(
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
            )
        )

        return torch.utils.data.DataLoader(
            dp, batch_size=None, worker_init_fn=worker_init_fn, num_workers=self.num_workers, pin_memory=True
        )

    def test(self):
        pass

    @classmethod
    def add_args(cls, parent_parser):
        # parent_parser = super().add_args(parent_parser)
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")

        parser.add_argument("--use_center_crop", action="store_true", help="verbose output")

        parser.add_argument("--train_path", nargs="+", type=str)
        parser.add_argument("--train_annotation_path", nargs="+", type=str)
        parser.add_argument("--train_filter_min_dim", type=int, default=128, help="delete images with smaller size")
        parser.add_argument("--train_sample_additional", type=float)

        parser.add_argument("--train_random_sizes", type=int, nargs="+", default=[480, 512, 544, 576, 608, 640])
        parser.add_argument("--max_size", type=int, default=800)

        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--batch_size", type=int, default=32)

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

        parser.add_argument("--labels_path", type=str)
        parser.add_argument("--mapping_path", type=str)
        parser.add_argument("--classifier_path", type=str)
        parser.add_argument("--ontology_path", type=str)
        parser.add_argument("--filter_label_by_count", type=int, default=0)

        parser.add_argument("--max_traces", type=int)

        parser.add_argument("--train_random_trace", action="store_true")
        parser.add_argument("--train_merge_one_hot", action="store_true")
        parser.add_argument("--targets", choices=("flat", "yolo", "clip", "onto"), nargs="+")

        return parser
