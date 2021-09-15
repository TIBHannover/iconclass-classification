#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 12:53:27 2021

@author: javad
"""

import argparse
import logging
import typing_extensions

import numpy as np

import torch
from torch.nn import functional as F
from torch import nn

import pytorch_lightning as pl
from models.models import ModelsManager


from models.base_model import BaseModel
from models.utils import gen_filter_mask
from datasets.utils import read_jsonl

from models.loss import FocalBCEWithLogitsLoss
from encoders import EncodersManager
from decoders import DecodersManager

from metrics import FBetaMetric, MAPMetric

from models import utils


@ModelsManager.export("encoder_hierarchical_decoder")
class EncoderHierarchicalDecoder(BaseModel):
    def __init__(self, args=None, **kwargs):
        super(EncoderHierarchicalDecoder, self).__init__(args, **kwargs)
        if args is not None:
            dict_args = vars(args)
            dict_args.update(kwargs)
        else:
            dict_args = kwargs

        self.encoder_model = dict_args.get("encoder_model", None)
        self.pretrained = dict_args.get("pretrained", None)

        self.mapping_path = dict_args.get("mapping_path", None)
        self.classifier_path = dict_args.get("classifier_path", None)
        self.ontology_path = dict_args.get("ontology_path", None)
        self.outpath_f2_per_lbs = dict_args.get("outpath_f2_per_lbs", None)

        self.use_label_smoothing = dict_args.get("use_label_smoothing", None)
        self.label_smoothing_factor = dict_args.get("label_smoothing_factor", 0.008)
        self.using_weights = dict_args.get("using_weights", None)

        self.use_focal_loss = dict_args.get("use_focal_loss", None)
        self.focal_loss_gamma = dict_args.get("focal_loss_gamma", None)
        self.focal_loss_alpha = dict_args.get("focal_loss_alpha", None)

        self.filter_label_by_count = dict_args.get("filter_label_by_count", None)

        self.best_threshold = dict_args.get("best_threshold", None)
        self.output_method = dict_args.get("output_method")

        self.mapping_config = []
        self.mask_vec = []
        if self.mapping_path is not None:
            self.mapping_config = read_jsonl(self.mapping_path)

        self.classifier = []
        if self.classifier_path is not None:
            self.classifier = read_jsonl(self.classifier_path)

        self.ontology = []
        if self.ontology_path is not None:
            self.ontology = read_jsonl(self.ontology_path)

        if self.filter_label_by_count is not None:
            self.filter_mask = torch.tensor(
                gen_filter_mask(self.mapping_config, self.filter_label_by_count, key="count.flat")
            )
        else:
            self.filter_mask = torch.ones(len(self.mapping_config), dtype=torch.float32)

        self.num_of_labels = torch.tensor(sum(self.mask_vec))

        if self.output_method == "level":
            self.level_map = utils.build_level_map(self.mapping_config)
            self.output_sizes = [2 + torch.sum(self.level_map == i).item() for i in range(len(self.ontology))]
            self.embedding_sizes = 2 + len(self.mapping_config)

        elif self.output_method == "minimal":
            self.output_sizes = [len(x["tokenizer"]) for x in self.ontology]  # get from tockenizer
            self.embedding_sizes = max(self.output_sizes)

        elif self.output_method == "global":
            self.output_sizes = [2 + len(self.mapping_config) for i in range(len(self.ontology))]
            self.embedding_sizes = 2 + len(self.mapping_config)

        # if self.use_weights is not None:
        #     self.weights = [x['weight'] for x in self.mapping_config]
        #     self.weights = np.array(self.weights)

        self.encoder = EncodersManager().build_encoder(name=args.encoder, args=args)
        self.decoder = DecodersManager().build_decoder(
            name=args.decoder,
            args=args,
            in_features=self.encoder.dim,
            embedding_size=self.embedding_sizes,
            vocabulary_sizes=self.output_sizes,
        )

        if self.using_weights:
            logging.info("Using weighting for loss")

            # for x in self.mapping_config:
            # self.weights[0, x["class_id"]] = x["weight"]
            self.weights = [x["weight_pos"] for x in self.mapping_config]
            self.weights = torch.Tensor(self.weights)
        else:
            self.weights = torch.ones(len(self.mapping_config))
        # print(self.weights)
        if self.use_focal_loss:
            self.loss = FocalBCEWithLogitsLoss(
                reduction="none", gamma=self.focal_loss_gamma, alpha=self.focal_loss_alpha
            )
        else:
            self.loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.fbeta = FBetaMetric(num_classes=len(self.mapping_config))
        self.map = MAPMetric(num_classes=len(self.mapping_config))

    def forward(self, x):
        return x

    def training_step(self, batch, batch_idx):

        image = batch["image"]
        target = batch["ontology_target"]

        flat_target = target.reshape(-1, target.shape[-1])
        trace_mask = batch["ontology_trace_mask"]

        src = batch["ontology_indexes"]
        tgt_level = utils.map_to_level_ontology(batch["ontology_target"], batch["ontology_levels"])
        mask_level = utils.map_to_level_ontology(batch["ontology_mask"], batch["ontology_levels"])

        # add pad and start
        src_level_with_tokens = utils.add_sequence_tokens_to_index(src, add_start=True)[:, :, :-1]
        tgt_level_with_tokens = utils.add_sequence_tokens_to_level_ontology_target(tgt_level, mask_level)
        mask_level_with_tokens = utils.add_sequence_tokens_to_level_ontology(mask_level)

        self.image = image
        # image = F.interpolate(image, size = (299,299), mode= 'bicubic', align_corners=False)
        # forward image
        image_embedding = self.encoder(image)

        # increase batchsize if we have more than one trace
        image_embedding = torch.repeat_interleave(image_embedding, src.shape[1], dim=0)

        # flat traces to batch
        tgt_level_with_tokens = [t.reshape(-1, t.shape[-1]) for t in tgt_level_with_tokens]
        mask_level_with_tokens = [t.reshape(-1, t.shape[-1]) for t in mask_level_with_tokens]
        src_level_with_tokens = src_level_with_tokens.reshape(-1, src_level_with_tokens.shape[-1])
        trace_mask = trace_mask.reshape(-1)

        # compute hierarchical prediction
        decoder_result = self.decoder(image_embedding, src_level_with_tokens)

        weights_level = utils.map_to_level_ontology(self.weights.to(image_embedding.device), batch["ontology_levels"])
        filter_mask_level = utils.map_to_level_ontology(
            self.filter_mask.to(image_embedding.device), batch["ontology_levels"]
        )
        weights_level_with_tokens = utils.add_sequence_tokens_to_level_ontology(weights_level, value=1.0)
        filter_mask_level_with_tokens = utils.add_sequence_tokens_to_level_ontology(filter_mask_level, value=1.0)

        losses = []
        for t, y, w, f, m in zip(
            tgt_level_with_tokens,
            decoder_result,
            weights_level_with_tokens,
            filter_mask_level_with_tokens,
            mask_level_with_tokens,
        ):
            final_mask = f * m * torch.unsqueeze(trace_mask, dim=-1)
            loss = self.loss(y, t) * w * final_mask

            loss = torch.sum(loss) / torch.sum(final_mask)
            losses.append(loss)

        loss = torch.mean(torch.stack(losses))

        self.log("train/loss", torch.mean(loss))
        return {"loss": torch.mean(loss)}

    def validation_step(self, batch, batch_idx):
        image = batch["image"]
        target = batch["ontology_target"]

        flat_target = target.reshape(-1, target.shape[-1])
        trace_mask = batch["ontology_trace_mask"]

        src = batch["ontology_indexes"]
        tgt_level = utils.map_to_level_ontology(batch["ontology_target"], batch["ontology_levels"])
        mask_level = utils.map_to_level_ontology(batch["ontology_mask"], batch["ontology_levels"])

        # add pad and start
        src_level_with_tokens = utils.add_sequence_tokens_to_index(src, add_start=True)

        src_level_with_tokens = src_level_with_tokens[:, :, :-1]
        tgt_level_with_tokens = utils.add_sequence_tokens_to_level_ontology_target(tgt_level, mask_level)
        mask_level_with_tokens = utils.add_sequence_tokens_to_level_ontology(mask_level)

        self.image = image
        # image = F.interpolate(image, size = (299,299), mode= 'bicubic', align_corners=False)
        # forward image
        image_embedding = self.encoder(image)

        # increase batchsize if we have more than one trace
        image_embedding = torch.repeat_interleave(image_embedding, src.shape[1], dim=0)

        # flat traces to batch
        tgt_level_with_tokens = [t.reshape(-1, t.shape[-1]) for t in tgt_level_with_tokens]
        mask_level_with_tokens = [t.reshape(-1, t.shape[-1]) for t in mask_level_with_tokens]
        src_level_with_tokens = src_level_with_tokens.reshape(-1, src_level_with_tokens.shape[-1])
        trace_mask = trace_mask.reshape(-1)

        # compute hierarchical prediction
        decoder_result = self.decoder(image_embedding, src_level_with_tokens)

        weights_level = utils.map_to_level_ontology(self.weights.to(image_embedding.device), batch["ontology_levels"])
        filter_mask_level = utils.map_to_level_ontology(
            self.filter_mask.to(image_embedding.device), batch["ontology_levels"]
        )
        weights_level_with_tokens = utils.add_sequence_tokens_to_level_ontology(weights_level, value=1.0)
        filter_mask_level_with_tokens = utils.add_sequence_tokens_to_level_ontology(filter_mask_level, value=1.0)

        losses = []
        for t, y, w, f, m in zip(
            tgt_level_with_tokens,
            decoder_result,
            weights_level_with_tokens,
            filter_mask_level_with_tokens,
            mask_level_with_tokens,
        ):

            final_mask = f * m * torch.unsqueeze(trace_mask, dim=-1)
            loss = self.loss(y, t) * w * final_mask

            loss = torch.sum(loss) / torch.sum(final_mask)
            losses.append(loss)

        loss = torch.mean(torch.stack(losses))

        decoder_without_tokens = utils.del_sequence_tokens_from_level_ontology(decoder_result)

        # flat output (similar to yolo)
        flat_prediction = utils.map_to_flat_ontology(decoder_without_tokens, batch["ontology_levels"])

        # delete empty traces
        # TODO Javad maybe we can merge all traces in one vector
        if True:
            trace_flat_prediction = torch.sigmoid(flat_prediction)[trace_mask == 1, ...]
            trace_flat_target = flat_target[trace_mask == 1, ...]

            # print("##############################")
            # print(trace_flat_prediction[:5, :20])
            # print(trace_flat_target[:5, :20])
            # else:
            trace_flat_prediction = torch.sum(torch.sigmoid(flat_prediction).reshape(target.shape), dim=1)
            trace_flat_prediction /= torch.sum(batch["ontology_mask"], dim=1)
            trace_flat_prediction[torch.isnan(trace_flat_prediction)] = 0
            trace_flat_prediction[torch.isinf(trace_flat_prediction)] = 0

            trace_flat_target = torch.sum(target, dim=1) / torch.sum(batch["ontology_mask"], dim=1)
            trace_flat_target[torch.isnan(trace_flat_target)] = 0

            # print(trace_flat_prediction[0, :20])
            # print(trace_flat_target[0, :20])

        self.fbeta(trace_flat_prediction, trace_flat_target)
        self.map(trace_flat_prediction, trace_flat_target)

        return {"loss": torch.mean(loss)}

    def validation_epoch_end(self, outputs):
        logging.info("EncoderHierarchicalDecoder::validation_epoch_end")

        loss = 0.0
        count = 0
        for output in outputs:
            loss += output["loss"]
            count += 1

        self.log("val/loss", loss / count, prog_bar=True)

        filter_mask = self.filter_mask
        self.log("val/filter", torch.sum(filter_mask))

        logging.info("EncoderHierarchicalDecoder::validation_epoch_end -> fbeta")
        fbeta = self.fbeta.compute()
        for thres, value in fbeta.items():
            self.log(f"val/fbeta-{thres}", self.fbeta.mean(value, filter_mask))

        logging.info("EncoderHierarchicalDecoder::validation_epoch_end -> map")
        ap_scores_per_class = self.map.compute()
        map_score = self.map.mean(ap_scores_per_class, filter_mask)
        logging.info(f"MAP score: {map_score}")
        self.log(f"val/map", map_score, prog_bar=True)

        self.fbeta.reset()
        self.map.reset()

    @classmethod
    def add_args(cls, parent_parser):
        logging.info("EncoderHierarchicalDecoder::add_args")
        parent_parser = super().add_args(parent_parser)
        parent_parser = EncodersManager.add_args(parent_parser)
        parent_parser = DecodersManager.add_args(parent_parser)
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")
        # args, _ = parser.parse_known_args()
        # if "classifier_path" not in args:
        parser.add_argument("--mapping_path", type=str)
        parser.add_argument("--classifier_path", type=str)
        parser.add_argument("--ontology_path", type=str)

        parser.add_argument("--outpath_f2_per_lbs", type=str)

        parser.add_argument("--use_diverse_beam_search", action="store_true", default=False)
        parser.add_argument("--div_beam_s_group", type=int, default=2)

        parser.add_argument("--use_label_smoothing", action="store_true", default=False)
        parser.add_argument("--using_weights", action="store_true", default=False)
        parser.add_argument("--use_focal_loss", action="store_true", default=False)
        parser.add_argument("--focal_loss_gamma", type=float, default=2)
        parser.add_argument("--focal_loss_alpha", type=float, default=0.25)

        parser.add_argument("--filter_label_by_count", type=int, default=None)
        parser.add_argument("--output_method", choices=["global", "level", "minimal"], default="level")

        parser.add_argument("--best_threshold", nargs="+", type=float, default=[0.2])
        return parser
