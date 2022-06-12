import re
import torch
import torch.nn.functional as F
import torchvision
import argparse
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List


from torchvision.models import resnet50, resnet152, densenet161, inception_v3

from encoders.encoders import EncodersManager

from encoders.position_encoding import PositionEmbeddingSine


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


@EncodersManager.export("resnet")
class ResnetEncoder(nn.Module):
    def __init__(self, args=None, returned_layers=None, average_pooling=None, **kwargs):
        super(ResnetEncoder, self).__init__()
        if args is not None:
            dict_args = vars(args)
            dict_args.update(kwargs)
        else:
            dict_args = kwargs

        self.returned_layers = returned_layers

        self.pretrained = dict_args.get("pretrained", None)
        self.byol_embedding_path = dict_args.get("byol_embedding_path", None)
        self.use_frozen_batch_norm = dict_args.get("use_frozen_batch_norm", None)
        self.encoder_finetune = dict_args.get("encoder_finetune", None)

        self.layers_returned = dict_args.get("layers_returned", ["layer4"])
        self.resnet_depth = dict_args.get("resnet_depth", "50")

        self.out_features = dict_args.get("resnet_output_depth")

        norm_layer = None
        if self.use_frozen_batch_norm:
            norm_layer = FrozenBatchNorm2d

        if int(self.resnet_depth) == 152:
            self.net = resnet152(pretrained=self.pretrained, progress=False, norm_layer=norm_layer)
            self.dim_3 = 1024
            self.dim_4 = 2048
            self.dim = 2048
        elif int(self.resnet_depth) == 50:
            self.net = resnet50(pretrained=self.pretrained, progress=False, norm_layer=norm_layer)
            self.dim_3 = 1024
            self.dim_4 = 2048
            self.dim = 2048

        if self.layers_returned is not None:
            self.body = IntermediateLayerGetter(
                self.net, return_layers={x: str(i) for i, x in enumerate(self.layers_returned)}
            )

            if self.out_features is not None:
                embedding_layers = []
                for i, x in enumerate(self.layers_returned):
                    if x == "layer4":
                        embedding_layers.append(torch.nn.Conv2d(self.dim_4, self.out_features, kernel_size=[1, 1]))
                    elif x == "layer3":
                        embedding_layers.append(torch.nn.Conv2d(self.dim_3, self.out_features, kernel_size=[1, 1]))

                # several outputs
                if len(embedding_layers) > 0:
                    self.feature_emb = torch.nn.ModuleList(embedding_layers)
                else:
                    self.feature_emb = torch.nn.ModuleList(
                        [torch.nn.Conv2d(self.dim_4, self.out_features, kernel_size=[1, 1])]
                    )
                self.dim = self.out_features
        else:
            if self.out_features is not None:
                self.feature_emb = torch.nn.ModuleList(
                    [torch.nn.Conv2d(self.dim[0], self.out_features, kernel_size=[1, 1])]
                )
                self.dim = self.out_features

        if self.byol_embedding_path is not None:
            self.load_pretrained_byol(self.byol_embedding_path)

        self.average_pooling = average_pooling
        if self.average_pooling:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        if self.layers_returned is not None:
            x = self.body(x)
            # x = self.net(x)
            if self.out_features is not None:
                x = [self.feature_emb[i](x[str(i)]) for i in range(len(self.layers_returned))]
            else:
                x = [x[str(i)] for i in range(len(self.layers_returned))]

        else:
            x = [self.net(x)]

            if self.out_features is not None:
                x = [self.feature_emb[0](x[0])]

        if self.average_pooling:
            x = [self.avgpool(y) for y in x]

        x = [y.permute(0, 2, 3, 1).reshape(y.size(0), -1, y.size(1)) for y in x]

        return torch.cat(x, dim=1)

    @classmethod
    def add_args(cls, parent_parser):
        # parent_parser = super().add_args(parent_parser)
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")
        # args, _ = parser.parse_known_args()
        # if "classifier_path" not in args:
        parser.add_argument("--pretrained", action="store_true")

        parser.add_argument("--byol_embedding_path", type=str)
        parser.add_argument("--use_frozen_batch_norm", action="store_true")
        parser.add_argument("--layers_returned", nargs="+", default=["layer4"])
        parser.add_argument("--resnet_depth", choices=["50", "152"], default="50")

        parser.add_argument("--resnet_output_depth", type=int, default=None)

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


@EncodersManager.export("resnet_position")
class ResnetPositionEncoder(nn.Module):
    def __init__(self, args=None, returned_layers=None, average_pooling=None, **kwargs):
        super(ResnetPositionEncoder, self).__init__()
        if args is not None:
            dict_args = vars(args)
            dict_args.update(kwargs)
        else:
            dict_args = kwargs
        self.returned_layers = returned_layers

        self.pretrained = dict_args.get("pretrained", None)
        self.byol_embedding_path = dict_args.get("byol_embedding_path", None)
        self.use_frozen_batch_norm = dict_args.get("use_frozen_batch_norm", None)
        self.encoder_finetune = dict_args.get("encoder_finetune", None)

        self.layers_returned = dict_args.get("layers_returned", ["layer4"])
        self.resnet_depth = dict_args.get("resnet_depth", "50")
        self.out_features = dict_args.get("resnet_output_depth")

        norm_layer = None
        if self.use_frozen_batch_norm:
            norm_layer = FrozenBatchNorm2d

        if int(self.resnet_depth) == 152:
            self.net = resnet152(pretrained=self.pretrained, progress=False, norm_layer=norm_layer)
            self.dim_3 = 1024
            self.dim_4 = 2048
            self.dim = 2048
        elif int(self.resnet_depth) == 50:
            self.net = resnet50(pretrained=self.pretrained, progress=False, norm_layer=norm_layer)
            self.dim_3 = 1024
            self.dim_4 = 2048
            self.dim = 2048

        if self.layers_returned is not None:
            self.body = IntermediateLayerGetter(
                self.net, return_layers={x: str(i) for i, x in enumerate(self.layers_returned)}
            )

            self.pos_embedder = []

            if self.out_features is not None:
                embedding_layers = []
                for i, x in enumerate(self.layers_returned):
                    if x == "layer4":
                        embedding_layers.append(torch.nn.Conv2d(self.dim_4, self.out_features, kernel_size=[1, 1]))
                        self.pos_embedder.append(PositionEmbeddingSine(self.out_features // 2, normalize=True))
                    elif x == "layer3":
                        embedding_layers.append(torch.nn.Conv2d(self.dim_3, self.out_features, kernel_size=[1, 1]))
                        self.pos_embedder.append(PositionEmbeddingSine(self.out_features // 2, normalize=True))

                # several outputs
                if len(embedding_layers) > 0:
                    self.feature_emb = torch.nn.ModuleList(embedding_layers)
                else:
                    self.feature_emb = torch.nn.ModuleList(
                        [torch.nn.Conv2d(self.dim_4, self.out_features, kernel_size=[1, 1])]
                    )
                self.dim = self.out_features
        else:
            if self.out_features is not None:
                self.feature_emb = torch.nn.ModuleList(
                    [torch.nn.Conv2d(self.dim[0], self.out_features, kernel_size=[1, 1])]
                )
                self.dim = self.out_features
                self.pos_embedder.append(PositionEmbeddingSine(self.out_features // 2, normalize=True))

        if self.byol_embedding_path is not None:
            self.load_pretrained_byol(self.byol_embedding_path)

        self.average_pooling = average_pooling
        if self.average_pooling:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        if self.layers_returned is not None:
            x = self.body(x)
            # x = self.net(x)
            if self.out_features is not None:
                x = [self.feature_emb[i](x[str(i)]) for i in range(len(self.layers_returned))]

                pos = [
                    self.pos_embedder[i](x[i], torch.ones_like(x[i])[:, 0, ...])
                    for i in range(len(self.layers_returned))
                ]
            else:
                x = [x[str(i)] for i in range(len(self.layers_returned))]

        else:
            x = [self.net(x)]

            if self.out_features is not None:
                x = [self.feature_emb[0](x[0])]
                pos = [self.pos_embedder[0](x[0], torch.ones_like(x[0]))]

        if self.average_pooling:
            x = [self.avgpool(y) for y in x]

        x = [y.permute(0, 2, 3, 1).reshape(y.size(0), -1, y.size(1)) for y in x]

        pos = [y.permute(0, 2, 3, 1).reshape(y.size(0), -1, y.size(1)) for y in pos]

        return torch.cat(x, dim=1) + torch.cat(pos, dim=1)

    @classmethod
    def add_args(cls, parent_parser):
        # parent_parser = super().add_args(parent_parser)
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")
        # args, _ = parser.parse_known_args()
        # if "classifier_path" not in args:
        parser.add_argument("--pretrained", action="store_true")
        parser.add_argument("--encoder_finetune", nargs="*", default=None)

        parser.add_argument("--byol_embedding_path", type=str)
        parser.add_argument("--use_frozen_batch_norm", action="store_true")
        parser.add_argument("--layers_returned", nargs="+", default=["layer4"])
        parser.add_argument("--resnet_depth", choices=["50", "152"], default="50")

        parser.add_argument("--resnet_output_depth", type=int, default=None)

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
