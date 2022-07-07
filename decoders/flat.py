import logging
import re
import torch
import torch.nn.functional as F
import torchvision
import argparse
from torch import nn
from typing import Dict, List

from decoders.decoders import DecodersManager


@DecodersManager.export("flat")
class FlatDecoder(nn.Module):
    def __init__(self, args=None, in_features=None, out_features=None, **kwargs):
        super(FlatDecoder, self).__init__()
        if args is not None:
            dict_args = vars(args)
            dict_args.update(kwargs)
        else:
            dict_args = kwargs

        # TODO
        print(in_features, out_features)
        in_features = 512
        print(in_features, out_features)
        self.decoder_dropout = dict_args.get("decoder_dropout", 0.5)
        self.decoder_hidden_dim = dict_args.get("decoder_hidden_dim", 512)

        self.in_features = in_features
        self.out_features = out_features

        self.dropout1 = torch.nn.Dropout(self.decoder_dropout)
        if self.decoder_hidden_dim is not None and self.decoder_hidden_dim > 0:
            self.fc = torch.nn.Linear(self.in_features, self.decoder_hidden_dim)
            self.dropout2 = torch.nn.Dropout(self.decoder_dropout)
            self.classifier = torch.nn.Linear(self.decoder_hidden_dim, self.out_features)
        else:
            self.classifier = torch.nn.Linear(self.in_features, self.out_features)

    def forward(self, inputs):

        assert "image_features" in inputs, ""
        x = inputs.get("image_features")

        if self.decoder_hidden_dim is not None and self.decoder_hidden_dim > 0:
            x = self.fc(x)
            x = self.dropout2(x)
            x = self.classifier(x)
        else:
            x = self.classifier(x)

        return {"prediction": x}

    @classmethod
    def add_args(cls, parent_parser):
        logging.info("Add FlatDecoder args")
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")
        parser.add_argument("--decoder_dropout", type=float, default=0.5)
        parser.add_argument("--decoder_hidden_dim", type=int, default=512)

        return parser
