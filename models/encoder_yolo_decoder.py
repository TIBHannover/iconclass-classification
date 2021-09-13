#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 12:53:27 2021

@author: javad
"""

import argparse
import logging

import numpy as np

import torch
from torch.nn import functional as F
from torch import nn

import pytorch_lightning as pl
from models.models import ModelsManager


from models.base_model import BaseModel
from models.utils import gen_filter_mask
from datasets.utils import read_jsonl, read_jsonl_lb_mapping

from models.loss import FocalBCEWithLogitsLoss
from encoders import EncodersManager
from decoders import DecodersManager


import pandas as pd
import pickle

from metrics import FBetaMetric, MAPMetric


@ModelsManager.export("encoder_yolo_decoder")
class EncoderYoloDecoder(BaseModel):
    def __init__(self, args=None, **kwargs):
        super(EncoderYoloDecoder, self).__init__(args, **kwargs)
        if args is not None:
            dict_args = vars(args)
            dict_args.update(kwargs)
        else:
            dict_args = kwargs

        self.encoder_model = dict_args.get("encoder_model", None)
        self.pretrained = dict_args.get("pretrained", None)

        self.mapping_path = dict_args.get("mapping_path", None)
        self.outpath_f2_per_lbs = dict_args.get("outpath_f2_per_lbs", None)

        self.use_label_smoothing = dict_args.get("use_label_smoothing", None)
        self.label_smoothing_factor = dict_args.get("label_smoothing_factor", 0.008)
        self.using_weights = dict_args.get("using_weights", None)

        self.use_focal_loss = dict_args.get("use_focal_loss", None)
        self.focal_loss_gamma = dict_args.get("focal_loss_gamma", None)
        self.focal_loss_alpha = dict_args.get("focal_loss_alpha", None)

        self.filter_label_by_count = dict_args.get("filter_label_by_count", None)

        self.best_threshold = dict_args.get("best_threshold", None)

        self.mapping_config = []
        self.mask_vec = []
        if self.mapping_path is not None:
            self.mapping_config = read_jsonl(self.mapping_path)

        if self.filter_label_by_count is not None:
            self.filter_mask = torch.tensor(
                gen_filter_mask(self.mapping_config, self.filter_label_by_count, key="count.flat")
            )
        else:
            self.filter_mask = torch.ones(len(self.mapping_config), dtype=torch.float32)

        self.num_of_labels = torch.tensor(sum(self.mask_vec))

        # if self.use_weights is not None:
        #     self.weights = [x['weight'] for x in self.mapping_config]
        #     self.weights = np.array(self.weights)

        # self.vocabulary_size = [len(x["tokenizer"]) for x in self.classifier_config]  # get from tockenizer
        # self.max_vocab_size = max(self.vocabulary_size)

        self.encoder = EncodersManager().build_encoder(name=args.encoder, args=args, out_features=1024)
        self.decoder = DecodersManager().build_decoder(
            name=args.decoder, args=args, in_features=self.encoder.dim, out_features=len(self.mapping_config)
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
            self.loss = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=self.weights)

        self.fbeta = FBetaMetric(num_classes=len(self.mapping_config))
        self.map = MAPMetric(num_classes=len(self.mapping_config))

    def forward(self, x):
        return x

    def training_step(self, batch, batch_idx):
        image = batch["image"]
        target = batch["yolo_target"]
        classes_mask = batch["yolo_target_mask"]

        self.image = image
        # image = F.interpolate(image, size = (299,299), mode= 'bicubic', align_corners=False)
        # forward image
        image_embedding = self.encoder(image)
        decoder_result = self.decoder(image_embedding)
        logits = decoder_result

        # print(f"{logits.shape} {target.shape}")
        weights = self.weights.to(decoder_result.device)
        filter_mask = self.filter_mask.to(decoder_result.device)

        loss = self.loss(decoder_result, target) * weights * filter_mask * classes_mask

        if hasattr(self.logger.experiment, "add_histogram"):
            self.logger.experiment.add_histogram(f"train/logits", logits, self.global_step)
            self.logger.experiment.add_histogram(f"train/target", target, self.global_step)

        self.log("train/loss", torch.sum(loss) / torch.sum(filter_mask * classes_mask))
        return {"loss": torch.mean(loss)}

    def validation_step(self, batch, batch_idx):
        image = batch["image"]
        target = batch["yolo_target"]
        classes_mask = batch["yolo_target_mask"]

        self.image = image
        # image = F.interpolate(image, size = (299,299), mode= 'bicubic', align_corners=False)
        # forward image
        image_embedding = self.encoder(image)
        decoder_result = self.decoder(image_embedding)

        logits = decoder_result
        # print(f"{logits.shape} {target.shape}")

        weights = self.weights.to(decoder_result.device)
        filter_mask = self.filter_mask.to(decoder_result.device)
        loss = self.loss(decoder_result, target) * weights * filter_mask * classes_mask

        if hasattr(self.logger.experiment, "add_histogram"):
            self.logger.experiment.add_histogram(f"val/logits", logits, self.global_step)
            self.logger.experiment.add_histogram(f"val/target", target, self.global_step)

        self.fbeta(torch.sigmoid(logits), target)
        self.map(torch.sigmoid(logits), target)

        return {"loss": torch.sum(loss) / torch.sum(filter_mask)}

    def validation_epoch_end(self, outputs):
        logging.info("EncoderYoloDecoder::validation_epoch_end")

        loss = 0.0
        count = 0
        for output in outputs:
            loss += output["loss"]
            count += 1

        self.log("val/loss", loss / count, prog_bar=True)

        filter_mask = self.filter_mask
        self.log("val/filter", torch.sum(filter_mask))

        logging.info("EncoderYoloDecoder::validation_epoch_end -> fbeta")
        fbeta = self.fbeta.compute()
        for thres, value in fbeta.items():
            self.log(f"val/fbeta-{thres}", value)

        logging.info("EncoderYoloDecoder::validation_epoch_end -> map")
        ap_scores_per_class = self.map.compute()
        map_score = self.map.mean(ap_scores_per_class, filter_mask)
        logging.info(f"MAP score: {map_score}")
        self.log(f"val/map", map_score, prog_bar=True)

        self.fbeta.reset()
        self.map.reset()

    @classmethod
    def add_args(cls, parent_parser):
        logging.info("EncoderYoloDecoder::add_args")
        parent_parser = super().add_args(parent_parser)
        parent_parser = EncodersManager.add_args(parent_parser)
        parent_parser = DecodersManager.add_args(parent_parser)
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")
        # args, _ = parser.parse_known_args()
        # if "classifier_path" not in args:
        parser.add_argument("--mapping_path", type=str)
        parser.add_argument("--outpath_f2_per_lbs", type=str)

        parser.add_argument("--use_diverse_beam_search", action="store_true", default=False)
        parser.add_argument("--div_beam_s_group", type=int, default=2)

        parser.add_argument("--use_label_smoothing", action="store_true", default=False)
        parser.add_argument("--using_weights", action="store_true", default=False)
        parser.add_argument("--use_focal_loss", action="store_true", default=False)
        parser.add_argument("--focal_loss_gamma", type=float, default=2)
        parser.add_argument("--focal_loss_alpha", type=float, default=0.25)

        parser.add_argument("--filter_label_by_count", type=int, default=None)

        parser.add_argument("--best_threshold", nargs="+", type=float, default=[0.2])
        return parser
