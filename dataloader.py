import os
import sys
import re
import argparse
import logging

import imageio


import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import torch

from callbacks import (
    ModelCheckpoint,
    ProgressPrinter,
    TensorBoardLogImageCallback,
    WandbLogImageCallback,
)  # , LogImageCallback
from datasets import DatasetsManager
from models import ModelsManager
from packaging import version

import json

try:
    import yaml
except:
    yaml = None


def parse_args():
    parser = argparse.ArgumentParser(description="", conflict_handler="resolve")

    # set logging
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-o", "--output_path", help="output path")

    args = parser.parse_known_args(["-v", "--verbose"])
    level = logging.ERROR
    if args[0].verbose:
        level = logging.INFO

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=level)

    # add arguments
    parser = DatasetsManager.add_args(parser)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    pl.seed_everything(42)

    dataset = DatasetsManager().build_dataset(name=args.dataset, args=args)

    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
    train_iter = iter(dataset.train())
    batch = next(train_iter)
    if batch.get("image") is not None:
        print(batch.get("image").shape)
        if args.output_path:
            for i, image in enumerate(torch.split(batch.get("image"), 1)):
                print(batch.get("txt"))
                print(batch.get("clip_embedding"))
                image = torch.permute(torch.squeeze(image), (1, 2, 0))
                image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))
                image = image.numpy()
                imageio.imwrite(os.path.join(args.output_path, f"image_{i}.jpg"), image)
                print(image.shape)
                print(f"{i} {batch.get('id')[i]} {batch.get('path')[i]}")
    batch = next(train_iter)
    if batch.get("image") is not None:
        print(batch.get("image").shape)
        if args.output_path:
            for i, image in enumerate(torch.split(batch.get("image"), 1)):
                print(batch.get("txt"))
                print(batch.get("clip_embedding"))
                image = torch.permute(torch.squeeze(image), (1, 2, 0))
                image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))
                image = image.numpy()
                imageio.imwrite(os.path.join(args.output_path, f"image_{i}_2.jpg"), image)
                print(image.shape)
                print(f"{i} {batch.get('id')[i]} {batch.get('path')[i]}")
    # print(batch.keys())
    # batch = next(iter(dataset.val()))
    # print(batch.get("image").shape)
    # print(batch.keys())

    return 0


if __name__ == "__main__":
    sys.exit(main())
