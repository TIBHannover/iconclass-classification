import re
import torch
import torch.nn.functional as F
import torchvision
import argparse
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List


from torchvision.models import resnet50, resnet152, densenet161, inception_v3


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class Encoder(nn.Module):
    def __init__(self, args=None, embedding_dim=None, flatten_embedding=None, returned_layers=None, **kwargs):
        super(Encoder, self).__init__()
        if args is not None:
            dict_args = vars(args)
            dict_args.update(kwargs)
        else:
            dict_args = kwargs
        self.flatten_embedding = flatten_embedding
        self.embedding_dim = embedding_dim
        self.returned_layers = returned_layers

        self.encoder_model = dict_args.get("encoder_model", None)
        self.pretrained = dict_args.get("pretrained", None)
        self.byol_embedding_path = dict_args.get("byol_embedding_path", None)
        self.use_frozen_batch_norm = dict_args.get("use_frozen_batch_norm", None)
        self.encoder_finetune = dict_args.get("encoder_finetune", None)

        norm_layer = None
        if self.use_frozen_batch_norm:
            norm_layer = FrozenBatchNorm2d

        if self.encoder_model == "resnet152":
            self.net = resnet152(pretrained=self.pretrained, progress=False, norm_layer=norm_layer)
            self.net = nn.Sequential(*list(self.net.children())[:-2])
            self.dim = 2048
        elif self.encoder_model == "densenet161":
            self.net = densenet161(pretrained=self.pretrained, progress=False, norm_layer=norm_layer)
            self.net = nn.Sequential(*list(list(self.net.children())[0])[:-2])
            self.dim = 1920
        elif self.encoder_model == "resnet50":
            self.net = resnet50(pretrained=self.pretrained, progress=False, norm_layer=norm_layer)
            self.net = nn.Sequential(*list(self.net.children())[:-2])
            self.dim = 2048
        elif self.encoder_model == "inceptionv3":  # TODO:: fix the input dimension of images
            self.net = inception_v3(pretrained=self.pretrained, progress=False, norm_layer=norm_layer)
            self.net = nn.Sequential(*list(self.net.children())[:-2])
            self.dim = 2048

        # for name, parameter in self.net.named_parameters():

        for name, parameter in self.net.named_parameters():
            if not self.encoder_finetune or "layer2" not in name and "layer3" not in name and "layer4" not in name:
                parameter.requires_grad_(False)
            # print(name)

        print("###################")

        # self.body = IntermediateLayerGetter(self.net, return_layers={"layer4": "0"})

        self._fc = nn.Linear(self.dim, self.embedding_dim)  # Todo add layers

        if self.embedding_dim is not None:
            self._conv1 = torch.nn.Conv2d(self.dim, self.embedding_dim, kernel_size=[1, 1])

        if self.byol_embedding_path is not None:
            self.load_pretrained_byol(self.byol_embedding_path)

    def forward(self, x):
        # x = self.body(x)["0"]
        x = self.net(x)
        if self.embedding_dim is not None:
            x = self._conv1(x)
        if self.flatten_embedding:
            x = x.permute(0, 2, 3, 1)
            x = x.view(x.size(0), -1, x.size(-1))
        return x

    @classmethod
    def add_args(cls, parent_parser):
        # parent_parser = super().add_args(parent_parser)
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")
        # args, _ = parser.parse_known_args()
        # if "classifier_path" not in args:
        parser.add_argument("--pretrained", action="store_true")
        parser.add_argument("--encoder_finetune", action="store_true")

        parser.add_argument(
            "--encoder_model", choices=("resnet152", "densenet161", "resnet50", "inceptionv3"), default="resnet50"
        )
        parser.add_argument("--byol_embedding_path", type=str)
        parser.add_argument("--use_frozen_batch_norm", action="store_true")

        return parser

    def load_pretrained_byol(self, path_checkpoint):
        assert self.encoder_model == "resnet50", "BYOL currently working with renset50"
        data = torch.load(path_checkpoint)["state_dict"]

        load_dict = {}
        for name, var in data.items():
            if "model.target_net.0._features" in name:
                new_name = re.sub("^model.target_net.0._features.", "", name)
                load_dict[new_name] = var
        # TODO move this to the encoder
        self.net.load_state_dict(load_dict)
