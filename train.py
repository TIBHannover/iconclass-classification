import os
import sys
import re
import argparse
import logging

import uuid


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


def parse_args():
    parser = argparse.ArgumentParser(description="", conflict_handler="resolve")

    # set logging
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    args = parser.parse_known_args(["-v", "--verbose"])
    print(args)
    level = logging.ERROR
    if args[0].verbose:
        level = logging.INFO

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=level)

    parser.add_argument("--output_path", help="verbose output")
    parser.add_argument("--use_wandb", action="store_true", help="verbose output")
    parser.add_argument("--progress_refresh_rate", type=int, default=100, help="verbose output")
    parser.add_argument("--wand_name", help="verbose output")
    parser.add_argument("--checkpoint_save_interval", type=int, default=100, help="verbose output")
    parser = pl.Trainer.add_argparse_args(parser)
    parser = DatasetsManager.add_args(parser)
    # parser = EncodersManager.add_args(parser)
    # parser = DecodersManager.add_args(parser)
    print("ModelsManager")
    parser = ModelsManager.add_args(parser)
    args = parser.parse_args()
    print(args)

    return args


def main():
    args = parse_args()

    pl.seed_everything(42)

    dataset = DatasetsManager().build_dataset(name=args.dataset, args=args)

    model = ModelsManager().build_model(name=args.model, args=args)

    if args.output_path is not None:
        os.makedirs(args.output_path, exist_ok=True)

    callbacks = [
        ProgressPrinter(refresh_rate=args.progress_refresh_rate),
        pl.callbacks.LearningRateMonitor(),
        # LogImageCallback(),
        # checkpoint_callback,
    ]

    if args.output_path is not None and not args.use_wandb:
        logger = TensorBoardLogger(save_dir=args.output_path, name="summary")

        callbacks.extend([TensorBoardLogImageCallback])
    elif args.use_wandb:
        name = f"{args.model}"
        if hasattr(args, "encoder"):
            name += f"-{args.encoder}"
        if hasattr(args, "decoder"):
            name += f"-{args.decoder}"
        name += f"-{uuid.uuid4().hex[:4]}"

        if args.wand_name is not None:
            name = args.wand_name
        logger = WandbLogger(project="iart_hierarchical", log_model=False, name=name)
        logger.watch(model)
        # callbacks.extend([WandbLogImageCallback()])
    else:
        logging.warning("No logger available")
        logger = None

    if version.parse(pl.__version__) < version.parse("1.4"):
        checkpoint_callback = ModelCheckpoint(
            checkpoint_save_interval=args.checkpoint_save_interval,
            dirpath=args.output_path,
            filename="model_{step:06d}",
            save_top_k=-1,
            verbose=True,
            period=1,
        )
    else:
        checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
            dirpath=args.output_path, save_top_k=-1, every_n_train_steps=args.checkpoint_save_interval
        )

    callbacks.extend([checkpoint_callback])

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        logger=logger,
        # checkpoint_callback=checkpoint_callback,
    )

    trainer.fit(model, train_dataloader=dataset.train(), val_dataloaders=dataset.val())

    return 0


if __name__ == "__main__":
    sys.exit(main())
