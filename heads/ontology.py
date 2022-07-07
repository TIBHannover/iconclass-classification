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


def build_parent_child_mask(mapping):
    parent_lut = {x["id"]: x for x in mapping}
    parent_child_masks = {}
    for m in mapping:
        if len(m["parents"]) == 0:
            parent = None
        else:
            parent = parent_lut[m["parents"][-1]]["index"]

        if parent not in parent_child_masks:
            parent_child_masks[parent] = torch.zeros(len(mapping), dtype=torch.int64)
        parent_child_masks[parent][m["index"]] = 1

    return parent_child_masks


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


@HeadsManager.export("ontology")
class OntologyHead(nn.Module):
    name = "ontology"

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
        self.ontology_loss_scale = dict_args.get("ontology_loss_scale", None)

        self.filter_label_by_count = dict_args.get("filter_label_by_count", None)

        self.ontology_max_iter = dict_args.get("ontology_max_iter", None)
        self.use_probability_chain = dict_args.get("use_probability_chain", None)

        self.mapping = {}
        if self.mapping_path is not None:
            self.mapping = read_line_data(self.mapping_path, dict_key="id")
            self.mapping_index = read_line_data(self.mapping_path, dict_key="index")

        self.level_map = build_level_map(self.mapping.values())
        self.parent_map = build_parent_map(self.mapping.values())
        self.parent_child_mask = build_parent_child_mask(self.mapping.values())

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

        assert "ontology_target" in targets, ""
        assert "ontology_levels" in targets, ""
        assert "ontology_mask" in targets, ""
        assert "classifier" in outputs, ""
        assert "ontology_trace_mask" in targets, ""

        device = targets["ontology_target"].device
        # exit()

        tgt_level = utils.map_to_level_ontology(targets["ontology_target"], targets["ontology_levels"])
        mask_level = utils.map_to_level_ontology(targets["ontology_mask"], targets["ontology_levels"])

        tgt_level_with_tokens = utils.add_sequence_tokens_to_level_ontology_target(tgt_level, mask_level)
        mask_level_with_tokens = utils.add_sequence_tokens_to_level_ontology(mask_level)
        tgt_level_with_tokens = [t.reshape(-1, t.shape[-1]) for t in tgt_level_with_tokens]

        # flat traces to batch
        tgt_level_with_tokens = [t.reshape(-1, t.shape[-1]) for t in tgt_level_with_tokens]
        mask_level_with_tokens = [t.reshape(-1, t.shape[-1]) for t in mask_level_with_tokens]
        # trace_mask = trace_mask.reshape(-1)

        weights_level = utils.map_to_level_ontology(self.weights.to(device), targets["ontology_levels"])
        filter_mask_level = utils.map_to_level_ontology(self.filter_mask.to(device), targets["ontology_levels"])
        weights_level_with_tokens = utils.add_sequence_tokens_to_level_ontology(weights_level, value=1.0)
        filter_mask_level_with_tokens = utils.add_sequence_tokens_to_level_ontology(filter_mask_level, value=1.0)

        trace_mask = targets["ontology_trace_mask"]
        trace_mask = trace_mask.reshape(-1)

        losses = []
        for t, y, w, f, m in zip(
            tgt_level_with_tokens,
            outputs["classifier"],
            weights_level_with_tokens,
            filter_mask_level_with_tokens,
            mask_level_with_tokens,
        ):
            # print(torch.unsqueeze(trace_mask, dim=-1))
            # print(y.shape, f.shape, m.shape, torch.unsqueeze(trace_mask, dim=-1).shape)

            final_mask = f * m * torch.unsqueeze(trace_mask, dim=-1)
            loss = self.loss_fn(y, t) * w * final_mask

            loss = torch.sum(loss) / torch.sum(final_mask)
            losses.append(loss)

        loss = self.ontology_loss_scale*torch.mean(torch.stack(losses))

        # assert "image_features" in outputs, "image_features not in encoder outputs"
        # assert "text_features" in outputs, "text_features not in encoder outputs"
        # assert "scale" in outputs, "scale not in encoder outputs"

        return loss

    def build_new_src(self, indices):

        new_src = []
        for x in indices.cpu().squeeze().numpy().tolist():

            # print(f"index {x}")
            token_sequence = self.mapping_index[x]["parents"] + [self.mapping_index[x]["id"]]

            # print(token_sequence)
            indexes = []
            for i, t in enumerate(token_sequence):

                indexes.append(self.mapping[t]["index"])

            for _ in range(len(indexes), len(self.ontology)):
                indexes.append(-1)
            new_src.append(torch.tensor(indexes, dtype=torch.int32))
        return torch.stack(new_src, dim=0).unsqueeze(1).to(indices.device)

    def build_new_mask(self, indices):

        new_mask = []
        for x in indices.cpu().squeeze().numpy().tolist():
            if x not in self.parent_child_mask:
                new_mask.append(torch.zeros(len(self.mapping), dtype=torch.int64))
            else:
                new_mask.append(self.parent_child_mask[x])
        return torch.stack(new_mask, dim=0).to(indices.device)

    def flat_prediction(self, model, targets, outputs):
        assert "image_embedding" in outputs, ""
        assert "ontology_levels" in targets, ""
        with torch.no_grad():
            batch_size = outputs.get("image_embedding").shape[0]
            len_prediction = len(self.mapping)
            device = outputs.get("image_embedding").device
            max_level = 8

            flat_prediction = torch.zeros([batch_size, len_prediction], device=device)

            valid_path = torch.reshape(self.parent_map, [1, -1])
            valid_path = valid_path.repeat(batch_size, 1).to(device)

            # we feed nothing into the pipeline
            src = torch.ones([batch_size, 1, len(self.ontology)], dtype=torch.int32, device=device) * -1
            output_mask = torch.stack([self.parent_child_mask[None] for _ in range(batch_size)], dim=0).to(device)

            for x in range(self.ontology_max_iter):
                # print("loop")
                # print(src.shape)
                # print(src[:2])
                decoder_outputs = model.decoder({**targets, **outputs, "ontology_indexes": src})

                classifier = utils.del_sequence_tokens_from_level_ontology(decoder_outputs["classifier"])
                classifier = [torch.sigmoid(x) for x in classifier]
                flat_classifier = utils.map_to_flat_ontology(classifier, targets.get("ontology_levels"))
                # print(flat_classifier.shape)
                # print(output_mask.shape)
                flat_classifier = flat_classifier * output_mask

                flat_prediction = torch.maximum(flat_classifier, flat_prediction)

                topk = torch.topk(flat_prediction * valid_path, 1)

                #  disable this index for the next round
                valid_path[torch.arange(valid_path.shape[0]), topk.indices.squeeze()] = 0

                # print(topk.indices)
                src = self.build_new_src(topk.indices)
                output_mask = self.build_new_mask(topk.indices)
                # print(output_mask)
                # exit()

            if self.use_probability_chain:
                return build_probability_chain(self.mapping.values(), flat_prediction)
            return flat_prediction

    @classmethod
    def add_args(cls, parent_parser):
        # parent_parser = super().add_args(parent_parser)
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")

        parser.add_argument("--ontology_max_iter", type=int, default=30)
        # parser.add_argument("--ontology_threshold", type=float, default=0.5)

        parser.add_argument("--mapping_path", type=str)

        parser.add_argument("--use_label_smoothing", action="store_true", default=False)
        parser.add_argument("--use_probability_chain", action="store_true", default=False)
        parser.add_argument("--using_weights", action="store_true", default=False)
        parser.add_argument("--ontology_loss_scale", type=float, default=1.0)
        parser.add_argument("--use_focal_loss", action="store_true", default=False)
        parser.add_argument("--focal_loss_gamma", type=float, default=2)
        parser.add_argument("--focal_loss_alpha", type=float, default=0.25)
        return parser
