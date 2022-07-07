import os
import sys
import re
import argparse
import logging


import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from callbacks import ModelCheckpoint, ProgressPrinter
from datasets import DatasetsManager
from models import ModelsManager

from pytorch_lightning.utilities.cloud_io import load as pl_load

from utils import move_to_device
import json
import numpy as np
import h5py

try:
    import yaml
except:
    yaml = None


def parse_args():
    parser = argparse.ArgumentParser(description="", conflict_handler="resolve")

    # set logging
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    args = parser.parse_known_args(["-v", "--verbose"])
    level = logging.ERROR
    if args[0].verbose:
        level = logging.INFO

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=level)

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
    parser.add_argument("--wand_name", help="verbose output")
    parser.add_argument("--checkpoint_save_interval", type=int, default=2000, help="verbose output")
    parser.add_argument("--prediction_path", help="verbose output")
    parser = pl.Trainer.add_argparse_args(parser)
    parser = DatasetsManager.add_args(parser)
    parser = ModelsManager.add_args(parser)
    args = parser.parse_args()

    # write results

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
    #     exit()
    model = ModelsManager().build_model(name=args.model, args=args)

    trainer = pl.Trainer.from_argparse_args(args)

    checkpoint_data = pl_load(args.resume_from_checkpoint, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint_data["state_dict"])
    model.freeze()
    model.eval()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    model.to(device)

    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)

    prediction_file = None
    if args.prediction_path:
        prediction_file = h5py.File(args.prediction_path, "w")

    h5_datasets = {}
    test_dataloader = dataset.test()
    index = 0
    model.on_test_epoch_start()
    for i, sample in enumerate(test_dataloader):
        sample_gpu = move_to_device(sample, device)
        prediction = model.test_step(sample_gpu, i)
        prediction = move_to_device(prediction, device)
        batch_size = sample["yolo_target"].shape[0]
        num_classes = sample["yolo_target"].shape[1]
        # print(prediction.keys())
        # print(sample.keys())

        # print(sample["ontology_target"].shape)
        # print(sample["yolo_target"].shape)
        # print(np.abs(sample["ontology_target"][0] - sample["yolo_target"][0]))
        # print(sample["ontology_trace_mask"])
        if prediction_file:
            if i == 0:
                h5_dataset = prediction_file.create_dataset(
                    "target", (batch_size, num_classes), maxshape=(None, num_classes)
                )
                h5_datasets["target"] = h5_dataset
                h5_dataset = prediction_file.create_dataset(
                    "id", (batch_size,), maxshape=(None,), dtype=h5py.special_dtype(vlen=str)
                )
                h5_datasets["id"] = h5_dataset

            h5_datasets["target"].resize((index + batch_size, num_classes))
            h5_datasets["target"][index : index + batch_size] = sample["yolo_target"]
            h5_datasets["id"].resize((index + batch_size,))
            h5_datasets["id"][index : index + batch_size] = sample["id"]

        for h, v in prediction["flat_prediction"].items():
            print(h)
            if prediction_file:
                if i == 0:
                    h5_dataset = prediction_file.create_dataset(
                        h, (batch_size, num_classes), maxshape=(None, num_classes)
                    )
                    h5_datasets[h] = h5_dataset
                # print(h)
                # print(v.shape)
                # print(v)

                h5_datasets[h].resize((index + batch_size, num_classes))
                h5_datasets[h][index : index + batch_size] = v.cpu().detach().numpy()

        index += batch_size
        # exit()

    return 0


if __name__ == "__main__":
    sys.exit(main())
