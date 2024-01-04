import os
import logging
import argparse
from copy import deepcopy

from collections import OrderedDict
from typing import Dict, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .encoders import EncodersManager

from open_clip import CLIP
from open_clip.factory import download_pretrained, _MODEL_CONFIGS, load_state_dict, get_pretrained_url

from datasets.utils import unflat_dict, flat_dict, get_element


@EncodersManager.export("open_clip")
class OpenClip(CLIP):
    def __init__(self, args=None, returned_layers=None, average_pooling=None, **kwargs):
        if args is not None:
            dict_args = vars(args)
            dict_args.update(kwargs)
        else:
            dict_args = kwargs

        self.returned_layers = returned_layers

        self.pretrained = dict_args.get("pretrained", None)
        self.skip_loading_pretrained = dict_args.get("skip_loading_pretrained", None)
        self.visual_model = dict_args.get("visual_model", None)
        self.force_quick_gelu = dict_args.get("force_quick_gelu", None)
        self.load_clip_from_checkpoint = dict_args.get("load_clip_from_checkpoint", None)

        self.flip_ration = dict_args.get("flip_ration", 0.0)
        self.new_context_length = dict_args.get("context_length", None)
        # print(dict_args.get("context_length", None))
        # exit()
        self.clip_delete_text_tower = dict_args.get("clip_delete_text_tower", None)
        model_name = self.visual_model

        #  copy from open_clip
        model_name = model_name.replace("/", "-")  # for callers using old naming with / in ViT names

        if model_name in _MODEL_CONFIGS:
            logging.info(f"Loading {model_name} model config.")
            model_cfg = deepcopy(_MODEL_CONFIGS[model_name])
        else:
            logging.error(f"Model config for {model_name} not found; available models {list_models()}.")
            raise RuntimeError(f"Model config for {model_name} not found.")

        if self.force_quick_gelu:
            # override for use of QuickGELU on non-OpenAI transformer models
            model_cfg["quick_gelu"] = True

        super(OpenClip, self).__init__(**model_cfg)

        if self.pretrained and not self.skip_loading_pretrained and not self.load_clip_from_checkpoint:
            checkpoint_path = ""
            url = get_pretrained_url(model_name, self.pretrained)
            if url:
                checkpoint_path = download_pretrained(url)
            elif os.path.exists(self.pretrained):
                checkpoint_path = self.pretrained

            if checkpoint_path:
                logging.info(f"Loading pretrained {model_name} weights ({self.pretrained}).")
                self.load_state_dict(load_state_dict(checkpoint_path))
            else:
                logging.warning(f"Pretrained weights ({self.pretrained}) not found for model {model_name}.")
                raise RuntimeError(f"Pretrained weights ({self.pretrained}) not found for model {model_name}.")

        # model.to(device=device)
        # if precision == "fp16":
        #     assert device.type != 'cpu'
        #     convert_weights_to_fp16(model)
        self.dim = model_cfg["embed_dim"]

        if self.new_context_length > self.positional_embedding.shape[0]:
            logging.warning(f"Try to increase position embedding for context_length {self.new_context_length}")
            self.additional_positional_embedding = torch.empty(
                self.new_context_length - self.positional_embedding.shape[0], self.positional_embedding.shape[1]
            )
            nn.init.normal_(self.additional_positional_embedding, std=0.01)
            self.positional_embedding = nn.Parameter(
                torch.concat(
                    [
                        self.positional_embedding,
                        self.additional_positional_embedding,
                    ],
                    axis=0,
                ),
            )
            logging.warning(f"New positional_embedding embedding size {self.positional_embedding.shape}")

            mask = torch.empty(self.new_context_length, self.new_context_length)
            mask.fill_(float("-inf"))
            mask.triu_(1)  # zero out the lower diagonal

            self.register_buffer("attn_mask", mask, persistent=False)
            # self.text.register_buffer('attn_mask', mask, persistent=False)
        # print(self.new_context_length, self.positional_embedding.shape[0])
        # exit()#
        if self.load_clip_from_checkpoint:
            state_dict = torch.load(self.load_clip_from_checkpoint, map_location="cpu")["state_dict"]
            # for x in state_dict:
            #     print(x)
            # state_dict = {k: v for k, v in state_dict}
            state_dict = flat_dict(unflat_dict(state_dict)["encoder"])
            # print(state_dict.keys())
            self.load_state_dict(state_dict)
            # exit()

        if self.clip_delete_text_tower:
            logging.warning("Delete text encoder")
            delattr(self, "transformer")
            delattr(self, "token_embedding")
            delattr(self, "positional_embedding")
            delattr(self, "ln_final")
            delattr(self, "text_projection")

    def apply_flip(self, patch_embeddings, drop_rate=0.5):
        BATCH = 0
        CHANNEL = 2
        GRID = 1

        s = torch.rand([patch_embeddings.shape[BATCH], patch_embeddings.shape[GRID]]).to(patch_embeddings.device)
        ids_shuffle = torch.argsort(s, dim=1)
        len_keep = int(patch_embeddings.shape[GRID] * drop_rate)
        ids_keep = ids_shuffle[:, :len_keep]

        result = torch.gather(
            patch_embeddings, GRID, ids_keep.unsqueeze(CHANNEL).expand(-1, -1, patch_embeddings.shape[CHANNEL])
        )

        return result

    def encode_image(self, image):
        x = self.visual.conv1(image)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.visual.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.visual.positional_embedding.to(x.dtype)

        # print(f"{ self.flip_ration} { self.training}", flush=True)
        if self.flip_ration > 0 and self.training:
            # print(f"111 {x.shape}", flush=True)
            x = self.apply_flip(x)
            # print(f"222 {x.shape}", flush=True)
        x = self.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.visual.ln_post(x)

        if self.visual.proj is not None:
            image_embedding = x[:, 0, :] @ self.visual.proj

        return image_embedding, x[:, 1:, :]

    def forward(self, inputs: Dict) -> Dict:
        image = inputs.get("image", None)
        text = inputs.get("text", None)

        assert image is not None or text is not None, "At least one of image or text should be defined"
        image_features = None
        image_embedding = None
        if image is not None:
            image_features, image_embedding = self.encode_image(image)
            image_features = F.normalize(image_features, dim=-1)

        text_features = None
        if text is not None and not self.clip_delete_text_tower:
            # print(f"####### {text.shape}", flush=True)
            text_features = self.encode_text(text)
            text_features = F.normalize(text_features, dim=-1)

        return {
            "image_features": image_features,
            "text_features": text_features,
            "scale": self.logit_scale.exp(),
            "image_embedding": image_embedding,
        }

    @classmethod
    def add_args(cls, parent_parser):
        # parent_parser = super().add_args(parent_parser)
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")
        # args, _ = parser.parse_known_args()
        # if "classifier_path" not in args:
        parser.add_argument("--pretrained", type=str, default="laion400m_e32")
        parser.add_argument("--skip_loading_pretrained", action="store_true")
        parser.add_argument("--visual_model", type=str, default="ViT-B-16")
        parser.add_argument("--force_quick_gelu", action="store_true")
        parser.add_argument("--load_clip_from_checkpoint")

        parser.add_argument("--flip_ration", type=float, default=0.0)
        parser.add_argument("--context_length", type=int, default=77)

        parser.add_argument("--clip_delete_text_tower", action="store_true")
        return parser
