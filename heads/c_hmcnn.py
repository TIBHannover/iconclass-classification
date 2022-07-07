import logging
import argparse

from tqdm import tqdm
import numpy as np

from datasets.utils import read_dict_data, read_line_data

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

import networkx as nx
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


def generate_adjacency_matrix(mapping):

    parent_lut = {x["id"]: x for x in mapping}

    num_nodes = len(mapping)
    R = np.zeros([num_nodes, num_nodes])

    for m in mapping:
        for p in m["parents"]:
            # R[parent_lut[p]["index"], m["index"]] = 1
            R[m["index"], parent_lut[p]["index"]] = 1

    return R


# def generate_R(mapping):
#     A = generate_adjacency_matrix(mapping)
#     num_nodes = len(mapping)
#     R = np.zeros([num_nodes, num_nodes])
#     np.fill_diagonal(R, 1)
#     g = nx.DiGraph(A)  # train.A is the matrix where the direct connections are stored
#     for i in range(len(A)):
#         ancestors = list(
#             nx.descendants(g, i)
#         )  # here we need to use the function nx.descendants() because in the directed graph the edges have source from the descendant and point towards the ancestor
#         if ancestors:
#             R[i, ancestors] = 1
#     R = torch.tensor(R, dtype=torch.int8)
#     # Transpose to get the descendants for each node
#     R = R.transpose(1, 0)
#     R = R.unsqueeze(0).expand(64, num_nodes, num_nodes).to_sparse()

#     return R


# def get_constr_out(x, R):
#     """Given the output of the neural network x returns the output of MCM given the hierarchy constraint expressed in the matrix R"""


#     c_out = x
#     c_out = c_out.unsqueeze(1)
#     c_out = c_out.expand(len(x), R.shape[1], R.shape[1])
#     R_batch = R
#     # R_batch = R.expand(len(x), R.shape[1], R.shape[1])


#     result = R_batch * c_out
#     result = result.coalesce()
#     final_out, _ = torch.max(result, dim=2)
#     return final_out


def generate_R(mapping):
    A = generate_adjacency_matrix(mapping)
    num_nodes = len(mapping)
    R = np.zeros([num_nodes, num_nodes])
    np.fill_diagonal(R, 1)
    g = nx.DiGraph(A)  # train.A is the matrix where the direct connections are stored
    for i in range(len(A)):
        ancestors = list(
            nx.descendants(g, i)
        )  # here we need to use the function nx.descendants() because in the directed graph the edges have source from the descendant and point towards the ancestor
        if ancestors:
            R[i, ancestors] = 1
    R = torch.tensor(R, dtype=torch.int8)
    # Transpose to get the descendants for each node
    R = R.transpose(1, 0)
    R = R.unsqueeze(0)

    return R


def get_constr_out(x, R):
    """Given the output of the neural network x returns the output of MCM given the hierarchy constraint expressed in the matrix R"""
    c_out = x
    c_out = c_out.unsqueeze(1)
    c_out = c_out.expand(len(x), R.shape[1], R.shape[1])
    R_batch = R.expand(len(x), R.shape[1], R.shape[1])
    final_out, _ = torch.max(R_batch * c_out, dim=2)
    return final_out


@HeadsManager.export("c_hmcnn")
class CHMCNNHead(nn.Module):
    name = "c_hmcnn"

    def __init__(self, args=None, **kwargs):
        super().__init__()
        if args is not None:
            dict_args = vars(args)
            dict_args.update(kwargs)
        else:
            dict_args = kwargs

        self.mapping_path = dict_args.get("mapping_path", None)
        self.ontology_path = dict_args.get("ontology_path", None)

        self.filter_label_by_count = dict_args.get("filter_label_by_count", None)

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

        self.weights = torch.ones(len(self.mapping))

        self.sigmoid = nn.Sigmoid()
        self.loss_fn = torch.nn.BCELoss(reduction="none")

        if self.filter_label_by_count is not None:
            self.filter_mask = torch.tensor(
                utils.gen_filter_mask(self.mapping, self.filter_label_by_count, key="count.flat")
            )
        else:
            self.filter_mask = torch.ones(len(self.mapping), dtype=torch.float32)

        # Compute matrix of ancestors R
        # Given n classes, R is an (n x n) matrix where R_ij = 1 if class i is descendant of class j
        self.R = generate_R(self.mapping.values())

        self.register_buffer("R_const", self.R)
        # exit()

    def loss(self, model, targets, outputs):

        assert "yolo_target" in targets, ""
        assert "prediction" in outputs, ""
        decoder_result = outputs.get("prediction")
        target = targets.get("yolo_target")
        R = self.R_const

        weights = self.weights.to(decoder_result.device)
        filter_mask = self.filter_mask.to(decoder_result.device)

        train_output = self.sigmoid(decoder_result)

        constr_output = get_constr_out(train_output, R)

        train_output = target * train_output
        train_output = get_constr_out(train_output, R)

        train_output = (1 - target) * constr_output + target * train_output

        with torch.autocast(device_type="cuda", enabled=False):
            loss = self.loss_fn(train_output.float(), target.float()) * weights * filter_mask
        predicted = constr_output.data > 0.5

        return torch.sum(loss) / torch.sum(filter_mask)

    def flat_prediction(self, model, targets, outputs):
        assert "prediction" in outputs, ""
        with torch.no_grad():
            prediction = torch.sigmoid(outputs.get("prediction"))

            prediction = get_constr_out(prediction, self.R_const)
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
