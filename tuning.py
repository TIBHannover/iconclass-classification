import os
import sys
import re
import argparse
import logging

import uuid

import ray
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback


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

# solve some issues with slurm
os.environ["SLURM_JOB_NAME"] = "bash"


def parse_args():
    parser = argparse.ArgumentParser(description="", conflict_handler="resolve")

    # set logging
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    args = parser.parse_known_args(["-v", "--verbose"])
    level = logging.ERROR
    if args[0].verbose:
        level = logging.INFO

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=level)

    parser.add_argument("--output_path", help="verbose output")
    parser.add_argument("--use_wandb", action="store_true", help="verbose output")
    parser.add_argument("--progress_refresh_rate", type=int, default=100, help="verbose output")
    parser.add_argument("--checkpoint_save_interval", type=int, default=100, help="verbose output")
    parser = pl.Trainer.add_argparse_args(parser)
    parser = DatasetsManager.add_args(parser)
    parser = ModelsManager.add_args(parser)
    args = parser.parse_args()

    return args


def hyper_config(args):
    parameters = {
        "lr": tune.choice([1e-2, 1e-3, 1e-4]),
        "batch_size": tune.choice([32, 16]),
        "use_focal_loss": tune.choice([True, False]),
        "optimizer": tune.choice(["SGD", "ADAM"]),
        "sched_type": tune.choice(["cosine", None]),
        "nesterov": tune.choice([True, False]),
        "weight_decay": tune.choice([1e-2, 1e-3, 1e-4, 1e-5]),
    }

    if args.decoder == "transformer_level_wise":
        parameters.update(
            {
                "transformer_d_model": tune.choice([256, 512]),
                "transformer_nhead": tune.choice([4, 8]),
                "transformer_num_encoder_layers": tune.choice([1, 3, 6]),
                "transformer_num_decoder_layers": tune.choice([1, 3, 6]),
                "transformer_dim_feedforward": tune.choice([1024, 2048]),
                "transformer_dropout": tune.choice([0.1]),
            }
        )

    if args.decoder == "attn_rnn_level_wise":
        parameters.update(
            {
                "decoder_dropout": tune.choice([0.1, 0.3, 0.5, 0.7, 0.9]),
                "decoder_embedding_dim": tune.choice([128, 256, 512]),
                "decoder_attention_dim": tune.choice([128, 256, 512]),
            }
        )
    return parameters


def train_tune(config, num_steps=20000, num_gpus=-1):
    args = config["args"]
    del config["args"]

    if "transformer_d_model" in config:
        if hasattr(args, "resnet_output_depth"):
            config["resnet_output_depth"] = config["transformer_d_model"]

    if "decoder_embedding_dim" in config:
        if hasattr(args, "resnet_output_depth"):
            config["resnet_output_depth"] = config["decoder_embedding_dim"]
    for k, v in config.items():
        print(f"{k}:{v}")
        setattr(args, k, v)

    level = logging.INFO

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=level)

    args.lr = config["lr"]
    if "use_focal_loss" in config and config["use_focal_loss"]:
        args.use_focal_loss = True

    dataset = DatasetsManager().build_dataset(name=args.dataset, args=args)

    model = ModelsManager().build_model(name=args.model, args=args)

    callbacks = [
        TuneReportCallback({"loss": "val/loss", "map": "val/map"}, on="validation_end"),
        ProgressPrinter(refresh_rate=1),
    ]

    trainer = pl.Trainer(
        max_steps=num_steps,
        # If fractional GPUs passed in, convert to int.
        gpus=num_gpus,
        logger=TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version="."),
        progress_bar_refresh_rate=0,
        callbacks=callbacks,
        val_check_interval=int(num_steps / 10),
        precision=16,
    )
    trainer.fit(model, train_dataloader=dataset.train(), val_dataloaders=dataset.val())


def main():
    args = parse_args()

    # ray.init(num_gpus=1)

    pl.seed_everything(42)

    config = hyper_config(args)

    config["args"] = args

    scheduler = tune.schedulers.ASHAScheduler(
        metric="map",
        mode="max",
    )
    analysis = tune.run(
        train_tune,
        config=config,
        num_samples=100,
        scheduler=scheduler,
        name="tune_hierarchical",
        resources_per_trial={"gpu": 1},
    )

    print(analysis.best_config)
    # callbacks = [
    #     ProgressPrinter(refresh_rate=args.progress_refresh_rate),
    #     pl.callbacks.LearningRateMonitor(),
    #     # LogImageCallback(),
    #     # checkpoint_callback,
    # ]

    # if args.output_path is not None and not args.use_wandb:
    #     logger = TensorBoardLogger(save_dir=args.output_path, name="summary")

    #     callbacks.extend([TensorBoardLogImageCallback])
    # elif args.use_wandb:
    #     name = f"{args.model}"
    #     if hasattr(args, "encoder"):
    #         name += f"-{args.encoder}"
    #     if hasattr(args, "decoder"):
    #         name += f"-{args.decoder}"
    #     name += f"-{uuid.uuid4().hex[:4]}"
    #     logger = WandbLogger(project="iart_hierarchical", log_model=False, name=name)
    #     logger.watch(model)
    #     # callbacks.extend([WandbLogImageCallback()])
    # else:
    #     logging.warning("No logger available")
    #     logger = None

    # if version.parse(pl.__version__) < version.parse("1.4"):
    #     checkpoint_callback = ModelCheckpoint(
    #         checkpoint_save_interval=args.checkpoint_save_interval,
    #         dirpath=args.output_path,
    #         filename="model_{step:06d}",
    #         save_top_k=-1,
    #         verbose=True,
    #         period=1,
    #     )
    # else:
    #     checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
    #         dirpath=args.output_path, save_top_k=-1, every_n_train_steps=args.checkpoint_save_interval
    #     )

    # callbacks.extend([checkpoint_callback])

    # # callbacks = [
    # #     ProgressPrinter(refresh_rate=args.progress_refresh_rate),
    # #     pl.callbacks.LearningRateMonitor(),
    # #     LogImageCallback(),
    # #     checkpoint_callback,
    # # ]
    # print(callbacks)
    # trainer = pl.Trainer.from_argparse_args(
    #     args,
    #     callbacks=callbacks,
    #     logger=logger,
    #     # checkpoint_callback=checkpoint_callback,
    # )

    # trainer.fit(model, train_dataloader=dataset.train(), val_dataloaders=dataset.val())

    return 0


if __name__ == "__main__":
    sys.exit(main())
