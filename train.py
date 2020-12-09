import os
import sys
import re
import argparse


import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from callbacks import ModelCheckpoint
from datasets import DatasetsManager
from models import ModelsManager


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")

    parser.add_argument("--output_path", help="verbose output")

    parser = DatasetsManager.add_args(parser)
    parser = ModelsManager.add_args(parser)

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    dataset = DatasetsManager().build_dataset(name=args.dataset, args=args)

    model = ModelsManager().build_model(name=args.model, args=args)

    if args.output_path is not None:
        logger = TensorBoardLogger(
            save_dir=args.output_path, name="summary"
        )  # save_dir=config.to_args().trainer.output_path, name="summary")
    else:
        logger = None

    callbacks = [
        # ProgressPrinter(refresh_rate=config.to_args().trainer.progress_refresh_rate),
        # pl.callbacks.LearningRateLogger(),
        # LogImageCallback(config.to_args().trainer.log_save_interval),
    ]

    checkpoint_callback = ModelCheckpoint(
        # checkpoint_save_interval=config.to_args().trainer.checkpoint_save_interval,
        # filepath=os.path.join(config.to_args().trainer.output_path, "model_{global_step:06d}"),
        # save_top_k=-1,
        # verbose=True,
        # # monitor="val_loss",
        # period=0,
    )

    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=callbacks, logger=logger, checkpoint_callback=checkpoint_callback,
    )

    for x in dataset.train():
        # print(x)

        exit()
    trainer.fit(model, train_dataloader=dataset.train(), val_dataloaders=dataset.val())

    return 0


if __name__ == "__main__":
    sys.exit(main())
