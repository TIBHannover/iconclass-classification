import logging
import re
import torch
import torch.nn.functional as F
import torchvision
import argparse
from torch import nn
from typing import Dict, List

from decoders.decoders import DecodersManager


@DecodersManager.export("dummy")
class DummyDecoder(nn.Module):
    def __init__(self, args=None, in_features=None, out_features=None, **kwargs):
        super(DummyDecoder, self).__init__()
        if args is not None:
            dict_args = vars(args)
            dict_args.update(kwargs)
        else:
            dict_args = kwargs

        # TODO

    def forward(self, inputs):
        return {}

    @classmethod
    def add_args(cls, parent_parser):
        logging.info("Add DummyDecoder args")
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")

        return parser
