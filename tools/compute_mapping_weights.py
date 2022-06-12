import os
import sys
import re
import argparse
import logging


import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import numpy as np
import json

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

from datasets import DatasetsManager
from datasets.utils import read_line_data


def parse_args():
    parser = argparse.ArgumentParser(description="", conflict_handler="resolve")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")

    parser.add_argument("--output_path", required=True, help="verbose output")
    parser.add_argument("--input_mapping_path", required=True, help="verbose output")

    parser = DatasetsManager.add_args(parser)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    level = logging.ERROR
    if args.verbose:
        level = logging.INFO

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=level)

    dataset = DatasetsManager().build_dataset(name=args.dataset, args=args)

    mapping_list = read_line_data(args.input_mapping_path)
    print(mapping_list[100])
    with open(args.output_path, "w") as f:
        label_sum_yolo = None
        label_sum_flat = None
        count = 0

        for i, d in enumerate(dataset.train()):
            # print(d.keys())
            count += d["yolo_target"].shape[0]
            if label_sum_yolo is None:
                label_sum_yolo = d["yolo_target"]
            else:
                label_sum_yolo = np.sum(
                    np.concatenate([d["yolo_target"], label_sum_yolo], axis=0), axis=0, keepdims=True
                )

            if label_sum_flat is None:
                label_sum_flat = d["flat_target"]
            else:
                label_sum_flat = np.sum(
                    np.concatenate([d["flat_target"], label_sum_flat], axis=0), axis=0, keepdims=True
                )

            # print(label_sum_flat.shape)
            # print(label_sum_flat[0, :10])
            # print(label_sum_yolo.shape)
            # print(label_sum_yolo[0, :10])
            # exit()
            # print(d["target"].shape)
            max_value = np.amax(np.asarray(label_sum_yolo))
            weights = count / (label_sum_yolo.shape[1] * label_sum_yolo)

            max_weights = np.amax(np.asarray(weights))
            # max_weights =
            # print(f"{label_sum_yolo[:6]} {weights[:6]} {max_value} {max_weights}")
            # print(d)
            # exit()
            if i % 1000 == 0:
                print(f"{i} {count} {max_value} {max_weights} {label_sum_yolo.shape}")

        weights[np.isinf(weights)] = 1.0
        for x in mapping_list:
            f.write(
                json.dumps(
                    {
                        **x,
                        "count": {
                            "yolo": label_sum_yolo[0, x["index"]].item(),
                            "flat": label_sum_flat[0, x["index"]].item(),
                        },
                        # "weight": {"yolo": {weights[0, x["index"]].item()}, "flat": {}},
                    }
                )
                + "\n"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
