#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 15:48:41 2021

@author: javad
"""

import re
import argparse
import logging

import numpy as np

import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule

import pytorch_lightning as pl
from models.models import ModelsManager


from encoders.clip import CLIP, convert_weights
from models.base_model import BaseModel
from models.utils import gen_filter_mask
from datasets.utils import read_line_data, read_jsonl_lb_mapping

from models.loss import FocalBCEWithLogitsLoss


from metrics import FBetaMetric, MAPMetric

import pandas as pd
import pickle


@ModelsManager.export("clip")
class CLIPModel(BaseModel):
    def __init__(self, args=None, **kwargs):
        super(CLIPModel, self).__init__(args, **kwargs)
        if args is not None:
            dict_args = vars(args)
            dict_args.update(kwargs)
        else:
            dict_args = kwargs

        self.encoder_model = dict_args.get("encoder_model", None)
        self.pretrained = dict_args.get("pretrained", None)

        self.mapping_path = dict_args.get("mapping_path", None)
        self.classifier_path = dict_args.get("classifier_path", None)
        self.label_mapping_path = dict_args.get("label_mapping_path", None)
        self.clip_vit_path = dict_args.get("clip_vit_path", None)

        self.txt_embedding_file = dict_args.get("txt_embedding_file", None)
        if self.txt_embedding_file is not None:
            with open(self.txt_embedding_file, "rb") as f:
                self.txt_mapping = pickle.load(f)
                self.txt_features = np.concatenate([x["clip"] for x in self.txt_mapping])
                self.txt_features = torch.from_numpy(self.txt_features)
        # self.outpath_f2_per_lbs = dict_args.get("outpath_f2_per_lbs", None)

        # self.use_diverse_beam_search = dict_args.get("use_diverse_beam_search", None)
        # self.div_beam_s_group = dict_args.get("div_beam_s_group", None)

        # self.use_label_smoothing = dict_args.get("use_label_smoothing", None)
        # self.LABEL_SMOOTHING_factor = 0.008  # TODO
        # self.using_weights = dict_args.get("using_weights", None)
        # self.use_focal_loss = dict_args.get("use_focal_loss", None)
        # self.focal_loss_gamma = dict_args.get("focal_loss_gamma", None)
        # self.focal_loss_alpha = dict_args.get("focal_loss_alpha", None)
        self.filter_label_by_count = dict_args.get("filter_label_by_count", None)

        # self.best_threshold = dict_args.get("best_threshold", None)

        self.mapping_config = []
        if self.mapping_path is not None:
            self.mapping_config = read_line_data(self.mapping_path)

        self.filter_mask = torch.tensor(
            gen_filter_mask(self.mapping_config, self.filter_label_by_count, key="count.flat")
        )

        self.classifier_config = {}
        if self.classifier_path is not None:
            self.classifier_config = read_line_data(self.classifier_path)

        if self.label_mapping_path is not None:
            self.label_mapping = read_jsonl_lb_mapping(self.label_mapping_path)

        # if self.use_weights is not None:
        #     self.weights = [x['weight'] for x in self.mapping_config]
        #     self.weights = np.array(self.weights)

        self.max_level = len(self.classifier_config)

        vision_width = 768
        vision_layers = 12
        vision_patch_size = 32
        grid_size = 7
        image_resolution = 224
        embed_dim = 512

        context_length = 77
        vocab_size = 49408
        transformer_width = 512
        transformer_heads = 8
        transformer_layers = 12

        self.net = CLIP(
            embed_dim,
            image_resolution,
            vision_layers,
            vision_width,
            vision_patch_size,
            context_length,
            vocab_size,
            transformer_width,
            transformer_heads,
            transformer_layers,
        )

        if self.clip_vit_path is not None:
            state_dict = torch.load(self.clip_vit_path)

            # for key in ["input_resolution", "context_length", "vocab_size"]:
            #     if key in state_dict:
            #         del state_dict[key]

            # convert_weights(self.net)
            self.net.load_state_dict(state_dict)

        self.loss_img = nn.CrossEntropyLoss()
        self.loss_txt = nn.CrossEntropyLoss()

        self.loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        # self.fbeta = FBetaMetric(num_classes=len(self.mapping_config))
        # self.map = MAPMetric(num_classes=len(self.mapping_config))

    def forward(self, x):
        return x

    def training_step(self, batch, batch_idx):  # TODO modify the train/validation code to the new version iconclass_all
        image = batch["image"]
        description = batch["token_lb"]

        # print("***")
        # print("var - {}, mean - {}".format(torch.var(image), torch.mean(image)))

        logits_per_image, logits_per_text = self.net(image, description)

        ground_truth = torch.arange(len(logits_per_image)).type_as(logits_per_image).long().to(image.device.index)

        loss = (self.loss_img(logits_per_image, ground_truth) + self.loss_txt(logits_per_text, ground_truth)).div(2)

        acc_i = (torch.argmax(logits_per_image, 1) == ground_truth).sum()
        acc_t = (torch.argmax(logits_per_text, 0) == ground_truth).sum()
        # self.image = image
        # # image = F.interpolate(image, size = (299,299), mode= 'bicubic', align_corners=False)
        # # forward image
        # image_embedding = self.encoder(image)

        return {"loss": loss, "acc": (acc_i + acc_t) / 2 / image.shape[0]}

    def training_step_end(self, outputs):

        ll = torch.mean(outputs["loss"])
        acc = outputs["loss"]
        self.log("train/loss", ll, prog_bar=True)
        self.log("train/acc_step", acc, prog_bar=True)
        # if (self.global_step % self.trainer.log_every_n_steps) == 0:
        #     for i, (pred, target) in enumerate(zip(outputs["predictions"], outputs["targets"])):
        #         self.logger.experiment.add_histogram(f"train/predict_{i}", pred, self.global_step)
        #         self.logger.experiment.add_histogram(f"train/target_{i}", target, self.global_step)

        return {"loss": ll}

    def validation_step(
        self, batch, batch_idx
    ):  # TODO modify the train/validation code to the new version iconclass_all
        image = batch["image"]
        description = batch["token_lb"]

        logits_per_image, logits_per_text = self.net(image, description)

        ground_truth = torch.arange(len(logits_per_image)).type_as(logits_per_image).long().to(image.device.index)

        loss = (self.loss_img(logits_per_image, ground_truth) + self.loss_txt(logits_per_text, ground_truth)).div(2)

        acc_i = (torch.argmax(logits_per_image, 1) == ground_truth).sum()
        acc_t = (torch.argmax(logits_per_text, 0) == ground_truth).sum()

        return {"loss": loss, "acc": (acc_i + acc_t) / 2 / image.shape[0]}

    def validation_epoch_end(self, outputs):

        loss = 0.0
        acc = 0.0
        count = 0
        for output in outputs:
            loss += output["loss"]
            acc += output["acc"]
            count += 1

        self.log("val/loss", loss / count, prog_bar=True)
        self.log("val/acc", acc / count, prog_bar=True)

    def on_test_epoch_start(self):
        self.all_predictions = []
        self.all_targets = []

    def test_step(self, batch, batch_idx):
        image = batch["image"]
        target = batch["yolo_target"]
        classes_mask = batch["yolo_target_mask"]

        image_features = self.net.encode_image(image)
        # text_features = self.net.encode_text(description)
        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = self.txt_features.to(image_features.device)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        # self.fbeta(text_probs, target) # metric may cause memory error
        # self.map(text_probs, target)

        tt = target.detach().cpu().numpy()
        pp = text_probs.detach().cpu().numpy()
        # pp = flat_prediction_norm.detach().cpu().numpy()
        # ll = total_loss.detach().cpu().numpy()

        self.all_targets.append(tt)
        self.all_predictions.append(pp)
        # self.all_losses.append(ll)

        filter_mask = self.filter_mask.to(image_features.device)

        loss = self.loss(text_probs, target) * filter_mask * classes_mask
        return {"loss": torch.mean(loss)}

    def test_epoch_end(self, outputs):

        loss = 0.0
        acc = 0.0
        count = 0
        for output in outputs:
            loss += output["loss"]
            # acc += output["acc"]
            count += 1

        self.log("test/loss", loss / count, prog_bar=True)
        # self.log("val/acc", acc / count, prog_bar=True)

        filter_mask = self.filter_mask
        self.log("test/filter", torch.sum(filter_mask))

        logging.info("EncoderCLIPDecoder::test_epoch_end -> fbeta")
        fbeta = fbeta_cpu(self.all_predictions, self.all_targets, len(self.mapping_config), mask=filter_mask)
        for thres, value in fbeta.items():
            self.log(f"test/fbeta-{thres}", value)

        logging.info("EncoderCLIPDecoder::test_epoch_end -> map")
        map_score = map_cpu(self.all_targets, self.all_predictions, mask=filter_mask.numpy())
        logging.info(f"MAP score: {map_score}")
        self.log(f"test/map", map_score, prog_bar=True)

    @classmethod
    def add_args(cls, parent_parser):
        parent_parser = super().add_args(parent_parser)
        # parent_parser = Encoder.add_args(parent_parser)
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")
        # args, _ = parser.parse_known_args()
        # if "classifier_path" not in args:
        parser.add_argument("--classifier_path", type=str)
        parser.add_argument("--mapping_path", type=str)
        parser.add_argument("--label_mapping_path", type=str)
        parser.add_argument("--clip_vit_path", type=str)
        parser.add_argument("--outpath_f2_per_lbs", type=str)
        parser.add_argument("--txt_embedding_file", type=str)

        parser.add_argument("--use_diverse_beam_search", action="store_true", default=False)
        parser.add_argument("--div_beam_s_group", type=int, default=2)

        parser.add_argument("--use_label_smoothing", action="store_true", default=False)
        parser.add_argument("--using_weights", action="store_true", default=False)
        parser.add_argument("--use_focal_loss", action="store_true", default=False)
        parser.add_argument("--focal_loss_gamma", type=float, default=2)
        parser.add_argument("--focal_loss_alpha", type=float, default=0.25)

        parser.add_argument("--filter_label_by_count", type=int, default=0)

        parser.add_argument("--best_threshold", nargs="+", type=float, default=[0.2])

        return parser
