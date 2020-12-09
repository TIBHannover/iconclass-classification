import os
import sys
import re
import argparse


import pytorch_lightning as pl
import torch

from datasets import DatasetsManager


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")

    parser.add_argument("--dataset_config", help="verbose output")

    parser.add_argument("--output_path", help="verbose output")

    return parser


def main():
    args_parser = parse_args()

    args_parser = DatasetsManager.add_args(args_parser)

    args = args_parser.parse_args()

    dataset = DatasetsManager().dataset(name=args.dataset)

    train_dataloader = torch.utils.data.DataLoader(dataset.train())

    trainer = Trainer(
        # early_stop_callback=early_stop_callback,
        gpus=-1,
        logger=logger,
        weights_save_path=args.output_path,
        checkpoint_callback=checkpoint_callback,
        val_check_interval=config.to_args().trainer.val_check_interval,
        distributed_backend="ddp",
        log_save_interval=config.to_args().trainer.log_save_interval,
        gradient_clip_val=config.to_args().trainer.gradient_clip_val,
        precision=config.to_args().trainer.precision,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
