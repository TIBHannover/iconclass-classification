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


def parse_args():
    parser = argparse.ArgumentParser(description="", conflict_handler="resolve")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")

    parser.add_argument("--output_path", help="verbose output")
    parser.add_argument("--progress_refresh_rate", type=int, default=100, help="verbose output")
    parser.add_argument("--checkpoint_save_interval", type=int, default=100, help="verbose output")

    parser = DatasetsManager.add_args(parser)
    parser = ModelsManager.add_args(parser)

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    level = logging.ERROR
    if args.verbose:
        level = logging.INFO

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=level)

    dataset = DatasetsManager().build_dataset(name=args.dataset, args=args)

    model = ModelsManager().build_model(name=args.model, args=args)

    if args.output_path is not None:
        os.makedirs(args.output_path, exist_ok=True)
        logger = TensorBoardLogger(save_dir=args.output_path, name="summary")
    else:
        logger = None

    callbacks = [
        # ProgressPrinter(refresh_rate=args.progress_refresh_rate),
        #pl.callbacks.LearningRateMonitor(),
        LogImageCallback(),
    ]

    checkpoint_callback = ModelCheckpoint(
        checkpoint_save_interval=args.checkpoint_save_interval,
        dirpath=args.output_path,
        filename="model_{global_step:06d}",
        save_top_k=-1,
        verbose=True,
        # monitor="val_loss",
        period=1,
    )

    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=callbacks, logger=logger, checkpoint_callback=checkpoint_callback,
    )

    trainer.fit(model, train_dataloader=dataset.train(), val_dataloaders=dataset.val())

    return 0


if __name__ == "__main__":
    sys.exit(main())
