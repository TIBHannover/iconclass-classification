import os
import sys
import re
import argparse


import pytorch_lightning as pl
import torch

from datasets.iconclass import IconClassDataloader


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    return 0


if __name__ == "__main__":
    sys.exit(main())
