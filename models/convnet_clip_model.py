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

from models.resnet import ResNet50

from models.base_model import BaseModel
from datasets.utils import read_jsonl, read_jsonl_lb_mapping

from models.loss import FocalBCEWithLogitsLoss

from pytorch_lightning.core.decorators import auto_move_data

from models.encoder import Encoder
from sklearn.metrics import fbeta_score


import pandas as pd
import pickle

from models.clip import CLIP, convert_weights

@ModelsManager.export("convnet_clip_model")
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
        self.mask_vec =[]
        if self.mapping_path is not None:
            self.mapping_config = read_jsonl(self.mapping_path)
            for x in self.mapping_config:
                    if x['count'] >self.filter_label_by_count:
                        self.mask_vec.append(1)
                    else:
                        self.mask_vec.append(0)
        
        self.num_of_labels = torch.tensor(sum(self.mask_vec))

        self.mapping_lut = {}
        for m in self.mapping_config:

            if len(m["parents"]) < 1:
                p = None
            else:
                p = m["parents"][-1]

            if p not in self.mapping_lut:
                self.mapping_lut[p] = {}
            self.mapping_lut[p][m["token_id_sequence"][-1]] = m["index"]

        self.classifier_config = {}
        if self.classifier_path is not None:
            self.classifier_config = read_jsonl(self.classifier_path)

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
        
        # embed_dim = state_dict["text_projection"].shape[1]
        # context_length = state_dict["positional_embedding"].shape[0]
        # vocab_size = state_dict["token_embedding.weight"].shape[0]
        # transformer_width = state_dict["ln_final.weight"].shape[0]
        # transformer_heads = transformer_width // 64
        # transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
        
        context_length=77
        vocab_size=49408
        transformer_width=512
        transformer_heads=8
        transformer_layers=12
        
        self.net = CLIP(embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)
        
        if self.clip_vit_path is not None:
            state_dict =torch.load(self.clip_vit_path)
        
            # for key in ["input_resolution", "context_length", "vocab_size"]:
            #     if key in state_dict:
            #         del state_dict[key]
    
            # convert_weights(self.net)
            self.net.load_state_dict(state_dict)

                
        self.loss_img = nn.CrossEntropyLoss()
        self.loss_txt = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.Adam(self.net.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
        
        # self.vocabulary_size = [len(x["tokenizer"]) for x in self.classifier_config]  # get from tockenizer
        # self.max_vocab_size = max(self.vocabulary_size)
        # self.embedding_dim = 768#256
        # self.attention_dim = 128
        # self.embedding_dim = 256
        # self.max_vocab_size = max(self.vocabulary_size)
        # self.encoder = Encoder(args, embedding_dim=self.embedding_dim, flatten_embedding=True)
        # self.encoder = Encoder(args, embedding_dim=None,flatten_embedding=False )
        # self.decoder = Decoder(
        #     self.vocabulary_size, self.embedding_dim, self.attention_dim, self.embedding_dim, self.max_vocab_size
        # )

        # if self.using_weights:
        #     logging.info("Using weighting for loss")

            # for x in self.mapping_config:
            # self.weights[0, x["class_id"]] = x["weight"]
            # self.weights = [x["weight_pos"] for x in self.mapping_config]
            # self.weights = torch.Tensor(self.weights)
        # else:
        #     self.weights = torch.ones(len(self.mapping_config))
        # print(self.weights)
        # if self.use_focal_loss:
        #     self.loss = FocalBCEWithLogitsLoss(
        #         reduction="none", gamma=self.focal_loss_gamma, alpha=self.focal_loss_alpha
        #     )
        # else:
        #     self.loss = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=self.weights)

        # self.all_predictions = []
        # self.all_targets = []
        # self.all_losses = []

    def forward(self, x):
        return x

    def training_step(self, batch, batch_idx):
        image = batch["image"]
        description = batch["token_lb"]
        
        # print("***")
        # print("var - {}, mean - {}".format(torch.var(image), torch.mean(image)))
        
        logits_per_image, logits_per_text = self.net(image, description)
        
        ground_truth = torch.arange(len(logits_per_image)).type_as(
            logits_per_image).long().to(image.device.index)
        
        loss = (self.loss_img(logits_per_image,ground_truth) + 
                      self.loss_txt(logits_per_text,ground_truth)).div(2)
        
        acc_i = (torch.argmax(logits_per_image, 1) == ground_truth).sum()
        acc_t = (torch.argmax(logits_per_text, 0) == ground_truth).sum()
        # self.image = image
        # # image = F.interpolate(image, size = (299,299), mode= 'bicubic', align_corners=False)
        # # forward image
        # image_embedding = self.encoder(image)
        

        
        return {"loss": loss, "acc": (acc_i+acc_t)/2/image.shape[0]}

    def training_step_end(self, outputs):

        ll = torch.mean(outputs["loss"])
        acc = outputs["loss"]
        self.log("train/loss",ll , prog_bar=True)
        self.log("train/acc_step",acc , prog_bar=True)
        # if (self.global_step % self.trainer.log_every_n_steps) == 0:
        #     for i, (pred, target) in enumerate(zip(outputs["predictions"], outputs["targets"])):
        #         self.logger.experiment.add_histogram(f"train/predict_{i}", pred, self.global_step)
        #         self.logger.experiment.add_histogram(f"train/target_{i}", target, self.global_step)

        return {"loss": ll}
    def validation_step(self, batch, batch_idx):
        image = batch["image"]
        description = batch["token_lb"]
 
        logits_per_image, logits_per_text = self.net(image, description)
        
        ground_truth = torch.arange(len(logits_per_image)).type_as(
            logits_per_image).long().to(image.device.index)
        
        loss = (self.loss_img(logits_per_image,ground_truth) + 
                      self.loss_txt(logits_per_text,ground_truth)).div(2)
        
        acc_i = (torch.argmax(logits_per_image, 1) == ground_truth).sum()
        acc_t = (torch.argmax(logits_per_text, 0) == ground_truth).sum()
        
        return {"loss": loss, "acc": (acc_i+acc_t)/2/image.shape[0]}
    
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

