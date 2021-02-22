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
from datasets.utils import read_jsonl


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

    mapping_list = read_jsonl(args.input_mapping_path)
    print(mapping_list[100])
    with open(args.output_path, "w") as f:
        label_sum = None
        count = 0
        for d in dataset.train():
            # print(d.keys())
            count += d["ids_vec"].shape[0]
            if label_sum is None:
                label_sum = d["ids_vec"]
            else:
                label_sum = np.sum(np.concatenate([d["ids_vec"], label_sum], axis=0), axis=0, keepdims=True)

            # print(d["target"].shape)
            max_value = np.amax(np.asarray(label_sum))
            weights = count / (label_sum.shape[1] * label_sum)

            max_weights = np.amax(np.asarray(weights))
            # max_weights =
            # print(f"{label_sum[:6]} {weights[:6]} {max_value} {max_weights}")
            # print(d)
            # exit()
        weights[np.isinf(weights)] = 1.0
        for x in mapping_list:
            f.write(
                json.dumps({**x, "count": label_sum[0, x["index"]].item(), "weight": weights[0, x["index"]].item()})
                + "\n"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
