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


def get_ontology_weights(
    subgraphs, classes, weight_type, num_classes, external_node_weight=1.0, min_node_weight=0.0, normalization=False
):

    ont_weight = [1] * num_classes

    if weight_type == "omega":
        for cls in classes.values():
            if len(cls["predecessors"]) == 0 or cls["wd_id"] in cls["subgraphs"]:
                w = external_node_weight
            else:
                w = 1 - (cls["connected_subgraphs"] - 1) / len(subgraphs.keys())

            ont_weight[cls["idx"]] = max(min_node_weight, w)

    elif weight_type == "omega_gamma":
        for cls in classes.values():
            w = 1 - (cls["connected_subgraphs"] - 1) / len(subgraphs.keys())

            # NOTE 2nd condition is only for WIDER because some classes are internal event nodes
            if len(cls["predecessors"]) == 0 or cls["wd_id"] in cls["subgraphs"]:
                y = external_node_weight
            else:
                y = 1 - np.exp(-(len(cls["predecessors"]) - 1))
            ont_weight[cls["idx"]] = max(min_node_weight, w * y)

    elif "delta" in weight_type:
        for cls in classes.values():
            sum_exnode_dist = 0
            for extnode_dist in cls["external_node_distance"].values():
                sum_exnode_dist += extnode_dist

            mean_extnode_dist = sum_exnode_dist / len(list(cls["external_node_distance"].keys()))

            if weight_type == "delta":
                delta = 1 / mean_extnode_dist
            elif weight_type == "delta-square":
                delta = 1 / (2 ** (mean_extnode_dist - 1))
            else:
                logging.error(f"Unkown weight type: {weight_type}. Exiting ...")
                exit()

            if len(cls["predecessors"]) == 0 or cls["wd_id"] in cls["subgraphs"]:
                # node is external node
                ont_weight[cls["idx"]] = external_node_weight
            else:
                # node is internal node
                ont_weight[cls["idx"]] = max(min_node_weight, delta)

    else:
        logging.error(f"Unkown weight type: {weight_type}. Exiting ...")
        exit()

    logging.info(f"Using weight vector {ont_weight[:20]} ...")
    return ont_weight
    #
    # sg_ont_weights = {}
    # for sg in subgraphs:
    #     if not normalization:
    #         sg_ont_weights[sg] = ont_weight
    #     else:
    #         # get sum of weights of all internal event nodes
    #         sum_internal_weight = 0
    #         for internal_node in sg["preorder_nodes"]:
    #             if internal_node != sg:
    #                 sum_internal_weight += ont_weight[classes[internal_node]['idx']]
    #
    #         # set weight of all external event nodes to external_node_weight * sum_internal_weight
    #         # as the whole weight vector is divided by sum_internal_weight it sets the weights to external_node_weight
    #         for cls in classes.items():
    #             if len(cls["predecessors"]) == 0 or cls["wd_id"] in cls["subgraphs"]:
    #                 ont_weight[cls["idx"]] = external_node_weight * sum_internal_weight
    #
    #         sg_ont_weights[sg] = ont_weight / sum_internal_weight
    #
    # return sg_ont_weights


class CosineOntLoss:
    def __init__(self, mapping):

        self.weights = torch.zeros(len(mapping))

        for x in mapping:
            if x["count"]["flat"] > 0:
                self.weights[x["index"]] = 0.5
            else:

                self.weights[x["index"]] = 1.0

    def __call__(self, result, target):
        weights = self.weights.to(result.device)

        target = target * weights
        result = result * weights
        return -torch.nn.functional.cosine_similarity(target, result).mean()


class SupGraphLoss:
    def __init__(self, mapping):

        self.weights = torch.zeros(len(mapping))

        for x in mapping:
            if x["count"]["flat"] > 0:
                self.weights[x["index"]] = 1
        self.loss = torch.nn.BCEWithLogitsLoss(reduction="none")

    def __call__(self, result, target):
        weights = self.weights.to(result.device)

        loss = self.loss(result, target)
        loss = torch.sum(loss * weights) / (torch.sum(weights) + 1e-10)
        return loss


@HeadsManager.export("cos_cel")
class CosCelHead(nn.Module):
    name = "cos_cel"

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

        self.cosine_loss = CosineOntLoss(self.mapping.values())
        self.sub_graph_loss = SupGraphLoss(self.mapping.values())
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
        assert "prediction" in outputs, ""
        decoder_result = outputs.get("prediction")
        target = targets.get("yolo_target")
        sub_graph_loss = self.sub_graph_loss(target, decoder_result)

        cosine_loss = self.cosine_loss(target, decoder_result)

        return cosine_loss + sub_graph_loss /10

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
