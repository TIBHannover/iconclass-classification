import logging
import re
import torch
import torch.nn.functional as F
import torchvision
import argparse
from torch import nn
from typing import Dict, List


from encoders.clip import VisualTransformer
from encoders.encoders import EncodersManager


@EncodersManager.export("vit")
class VitEncoder(nn.Module):
    def __init__(self, args=None, out_features=None, **kwargs):
        super(VitEncoder, self).__init__()
        if args is not None:
            dict_args = vars(args)
            dict_args.update(kwargs)
        else:
            dict_args = kwargs

        self.out_features = out_features

        self.clip_vit_path = dict_args.get("clip_vit_path", None)
        self.use_clip_attention = dict_args.get("using_clip_attention", False)

        vision_width = 768
        vision_layers = 12
        vision_patch_size = 32
        grid_size = 7
        image_resolution = 224
        embed_dim = 512

        vision_heads = vision_width // 64
        self.net = VisualTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            attention_flag=self.use_clip_attention,
        )
        self.dim = 512
        if self.use_clip_attention:
            self.dim = 768

        if self.out_features is not None and self.out_features > 0:
            self.output_fc = nn.Linear(self.dim, out_features)
            self.dim = out_features

        if self.clip_vit_path is not None:
            self.load_pretrained_clip_vit(self.clip_vit_path)

    def forward(self, x):
        x = self.net(x)
        if not self.use_clip_attention:
            x = torch.unsqueeze(x, dim=1)

        if self.out_features is not None and self.out_features > 0:
            x = self.output_fc(x)

        return x

    @classmethod
    def add_args(cls, parent_parser):
        logging.info("Add VitEncoder args")
        # parent_parser = super().add_args(parent_parser)
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")
        # args, _ = parser.parse_known_args()
        # if "classifier_path" not in args:
        parser.add_argument("--clip_vit_path", type=str)
        parser.add_argument("--using_clip_attention", action="store_true", default=False)

        return parser

    def load_pretrained_clip_vit(self, path_checkpoint):
        self.net.load_state_dict(torch.load(path_checkpoint))