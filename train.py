import os
from subprocess import call
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

import json

try:
    import yaml
except:
    yaml = None

from utils import get_node_rank

# logger = logging.getLogger()
# logger = logging.LoggerAdapter(logger, {"rank": get_node_rank()})


def parse_args():
    parser = argparse.ArgumentParser(description="", conflict_handler="resolve")

    # set logging
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    args = parser.parse_known_args(["-v", "--verbose"])
    level = logging.ERROR
    if args[0].verbose:
        level = logging.INFO
    logging.basicConfig(
        format=f"%(asctime)s rank:{get_node_rank()}  %(filename)s::[%(name)s::%(funcName)s] -> %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
        level=level,
    )

    # parse config
    parser.add_argument("-c", "--config_path", help="verbose output")

    args = parser.parse_known_args()
    if args[0].config_path:
        if re.match("^.*?\.(yml|yaml)$", args[0].config_path):
            with open(args[0].config_path, "r") as f:
                data_dict = yaml.safe_load(f)
                parser.set_defaults(**data_dict)

        if re.match("^.*?\.(json)$", args[0].config_path):
            with open(args[0].config_path, "r") as f:
                data_dict = json.load(f)
                parser.set_defaults(**data_dict)

    # add arguments
    parser.add_argument("--output_path", help="verbose output")
    parser.add_argument("--use_wandb", action="store_true", help="verbose output")
    parser.add_argument("--progress_refresh_rate", type=int, default=100, help="verbose output")
    parser.add_argument("--wandb_project", default="iart_iconclass", help="verbose output")
    parser.add_argument("--wandb_name", help="verbose output")
    parser.add_argument("--load_checkpoint", help="verbose output")
    parser.add_argument("--checkpoint_save_interval", type=int, default=2000, help="verbose output")
    parser = pl.Trainer.add_argparse_args(parser)
    parser = DatasetsManager.add_args(parser)
    parser = ModelsManager.add_args(parser)
    args = parser.parse_args()

    # write results

    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
        if yaml is not None:
            with open(os.path.join(args.output_path, "config.yaml"), "w") as f:
                yaml.dump(vars(args), f, indent=4)

        with open(os.path.join(args.output_path, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

    return args


def main():
    args = parse_args()

    pl.seed_everything(42)

    dataset = DatasetsManager().build_dataset(name=args.dataset, args=args)
    model = ModelsManager().build_model(name=args.model, args=args)

    if args.load_checkpoint:
        logging.info("load_checkpoint")
        checkpoint = torch.load(args.load_checkpoint, map_location="cpu")["state_dict"]
        model.load_state_dict(checkpoint)

    if args.output_path is not None:
        os.makedirs(args.output_path, exist_ok=True)

    callbacks = [
        ProgressPrinter(refresh_rate=args.progress_refresh_rate),
    ]

    # if args.output_path is not None and not args.use_wandb:
    #     logger = TensorBoardLogger(save_dir=args.output_path, name="summary")

    #     callbacks.extend([TensorBoardLogImageCallback])

    logger = None
    if args.use_wandb:
        name = f"{args.model}"
        if hasattr(args, "encoder"):
            name += f"-{args.encoder}"
        if hasattr(args, "decoder"):
            name += f"-{args.decoder}"
        name += f"-{uuid.uuid4().hex[:4]}"

        if args.wandb_name is not None:
            name = args.wandb_name
        # logger = WandbLogger(project=args.wandb_project, log_model=False, name=name)
        # logger.watch(model)
        # callbacks.extend([pl.callbacks.LearningRateMonitor()])
        # callbacks.extend([WandbLogImageCallback()])
    else:
        logging.warning("No logger available")

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
        # callbacks=[ProgressPrinter(refresh_rate=args.progress_refresh_rate)]
        callbacks=callbacks,
        logger=logger,
        enable_checkpointing=checkpoint_callback,
        # enable_progress_bar=False,
        auto_select_gpus=False,
    )

    logging.info(f"Start training {args.output_path}")
    trainer.fit(model, train_dataloaders=dataset.train(), val_dataloaders=dataset.val())

    return 0


if __name__ == "__main__":
    sys.exit(main())
