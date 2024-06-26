import os
import sys
import re
import argparse
import logging


import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from callbacks import ModelCheckpoint, ProgressPrinter, LogImageCallback  # , LogImageCallback
from datasets import DatasetsManager
from models import ModelsManager

from pytorch_lightning.utilities.cloud_io import load as pl_load


def parse_args():
    parser = argparse.ArgumentParser(description="", conflict_handler="resolve")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")

    parser = pl.Trainer.add_argparse_args(parser)
    parser = DatasetsManager.add_args(parser)
    parser = ModelsManager.add_args(parser)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    level = logging.ERROR
    if args.verbose:
        level = logging.INFO

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=level)

    dataset = DatasetsManager().build_dataset(name=args.dataset, args=args)

    # for x in dataset.test():
    #     print(x)
    # exit()

    model = ModelsManager().build_model(name=args.model, args=args)

    trainer = pl.Trainer.from_argparse_args(args)

    checkpoint_data = pl_load(args.resume_from_checkpoint, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint_data["state_dict"])
    model.freeze()
    model.eval()
    # model.to("cuda:0")

    for batch in dataset.infer():

        for prediction in model.infer_step(batch, k=20):
            print(f"{prediction['id']} {prediction['txt']}")
            # print(prediction)

    return 0


if __name__ == "__main__":
    sys.exit(main())
