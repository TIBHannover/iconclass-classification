import logging
import argparse

from tqdm import tqdm

from datasets.utils import read_dict_data, read_line_data

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

from torch import Tensor

from utils import world_info_from_env
from .manager import HeadsManager

from models import utils
from models.loss import FocalBCEWithLogitsLoss


def build_level_map(mapping):
    level_map = torch.zeros(len(mapping), dtype=torch.int64)
    for m in mapping:
        level_map[m["index"]] = len(m["parents"])

    return level_map


def build_parent_map(mapping):
    parent_lut = {x["id"]: x for x in mapping}
    parent_map = torch.zeros(len(mapping), dtype=torch.int64)
    for m in mapping:
        for p in m["parents"]:
            parent_map[parent_lut[p]["index"]] = 1

    return parent_map


def build_probability_chain(mapping, prediction):
    parent_lut = {x["id"]: x for x in mapping}
    weighted_prediction = torch.zeros_like(prediction)
    for m in mapping:
        if len(m["parents"]) == 0:
            weighted_prediction[:, m["index"]] = prediction[:, m["index"]]
        else:
            parent_prob = weighted_prediction[:, parent_lut[m["parents"][-1]]["index"]]
            weighted_prediction[:, m["index"]] = parent_prob * prediction[:, m["index"]]
    return weighted_prediction


@HeadsManager.export("yolo")
class YOLOHead(nn.Module):
    name = "yolo"

    def __init__(self, args=None, **kwargs):
        super().__init__()
        if args is not None:
            dict_args = vars(args)
            dict_args.update(kwargs)
        else:
            dict_args = kwargs

        self.mapping_path = dict_args.get("mapping_path", None)
        self.ontology_path = dict_args.get("ontology_path", None)
        self.using_weights = dict_args.get("using_weights", None)

        self.use_focal_loss = dict_args.get("use_focal_loss", None)
        self.focal_loss_gamma = dict_args.get("focal_loss_gamma", None)
        self.focal_loss_alpha = dict_args.get("focal_loss_alpha", None)

        self.filter_label_by_count = dict_args.get("filter_label_by_count", None)

        self.ontology_max_iter = dict_args.get("ontology_max_iter", None)
        self.use_probability_chain = dict_args.get("use_probability_chain", None)

        self.mapping = {}
        if self.mapping_path is not None:
            self.mapping = read_line_data(self.mapping_path, dict_key="id")
            self.mapping_index = read_line_data(self.mapping_path, dict_key="index")

        self.level_map = build_level_map(self.mapping.values())
        self.parent_map = build_parent_map(self.mapping.values())

        self.ontology = []
        if self.ontology_path is not None:
            self.ontology = read_line_data(self.ontology_path)

        if self.using_weights:
            logging.info("Using weighting for loss")

            # for x in self.mapping:
            # self.weights[0, x["class_id"]] = x["weight"]
            self.weights = [x["weight_pos"] for x in self.mapping]
            self.weights = torch.Tensor(self.weights)
        else:
            self.weights = torch.ones(len(self.mapping))

        # print(self.weights)
        if self.use_focal_loss:
            self.loss_fn = FocalBCEWithLogitsLoss(
                reduction="none", gamma=self.focal_loss_gamma, alpha=self.focal_loss_alpha
            )
        else:
            self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

        if self.filter_label_by_count is not None:
            self.filter_mask = torch.tensor(
                utils.gen_filter_mask(self.mapping, self.filter_label_by_count, key="count.flat")
            )
        else:
            self.filter_mask = torch.ones(len(self.mapping), dtype=torch.float32)

    def loss(self, model, targets, outputs):

        assert "yolo_target" in targets, ""
        assert "yolo_target_mask" in targets, ""
        assert "prediction" in outputs, ""
        decoder_result = outputs.get("prediction")
        target = targets.get("yolo_target")
        classes_mask = targets.get("yolo_target_mask")

        weights = self.weights.to(decoder_result.device)
        filter_mask = self.filter_mask.to(decoder_result.device)
        loss = self.loss_fn(decoder_result, target) * weights * filter_mask * classes_mask

        return torch.sum(loss) / torch.sum(filter_mask)

    def flat_prediction(self, model, targets, outputs):
        assert "prediction" in outputs, ""
        with torch.no_grad():
            prediction = torch.sigmoid(outputs.get("prediction"))
            if self.use_probability_chain:
                return build_probability_chain(self.mapping.values(), prediction)

            return prediction

    @classmethod
    def add_args(cls, parent_parser):
        # parent_parser = super().add_args(parent_parser)
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")

        parser.add_argument("--ontology_max_iter", type=int, default=20)
        # parser.add_argument("--ontology_threshold", type=float, default=0.5)

        parser.add_argument("--mapping_path", type=str)

        parser.add_argument("--use_label_smoothing", action="store_true", default=False)
        parser.add_argument("--using_weights", action="store_true", default=False)
        parser.add_argument("--use_focal_loss", action="store_true", default=False)
        parser.add_argument("--use_probability_chain", action="store_true", default=False)
        parser.add_argument("--focal_loss_gamma", type=float, default=2)
        parser.add_argument("--focal_loss_alpha", type=float, default=0.25)
        return parser
