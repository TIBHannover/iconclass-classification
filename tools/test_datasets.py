#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 21:05:38 2021

@author: javad
"""

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

    # parser.add_argument("--output_path", required=True, help="verbose output")
    # parser.add_argument("--input_mapping_path", required=True, help="verbose output")

    # parser.add_argument("--filter_label_by_count", required=True, type=int, help="verbose output")

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

    # mapping_list = read_jsonl(args.input_mapping_path)
    # print(mapping_list[100])
    # filtered_lbs_inds = []
    # for ind, x in enumerate(mapping_list):
    # if x["count"] < args.filter_label_by_count:
    # filtered_lbs_inds.append(ind)

    for d in dataset.train():
        print(d)
        exit()

    with open(args.output_path, "w") as f:
        label_sum = None
        count = 0
        for d in dataset.train():
            # print(d.keys())
            d_filtered = np.delete(d["ids_vec"], filtered_lbs_inds, axis=1)
            count += d_filtered.shape[0]
            if label_sum is None:
                label_sum = d_filtered
            else:
                label_sum = np.sum(np.concatenate([d_filtered, label_sum], axis=0), axis=0, keepdims=True)

                aa = label_sum.T
            # print(d["target"].shape)
            max_value = np.amax(np.asarray(label_sum))
            weights = count / (label_sum.shape[1] * label_sum)

            max_weights = np.amax(np.asarray(weights))
            # max_weights =
            # print(f"{label_sum[:6]} {weights[:6]} {max_value} {max_weights}")
            # print(d)
            # exit()
        print(count)
        weights[np.isinf(weights)] = 1.0
        weight_positive = (count - label_sum) / label_sum
        weight_positive[np.isinf(weight_positive)] = 1.0
        non_filter_inds = [i for i in range(weights.shape[1])]
        print("writing in jsonl file!")
        cnt = 0
        for x in mapping_list:
            if x["count"] >= args.filter_label_by_count:
                # print(x)
                f.write(
                    json.dumps(
                        {
                            **x,
                            "count": label_sum[0, cnt].item(),
                            "weight": weights[0, cnt].item(),
                            "weight_pos": weight_positive[0, cnt].item(),
                        }
                    )
                    + "\n"
                )
                cnt += 1
            else:
                f.write(json.dumps({**x, "count": 0, "weight": 0, "weight_pos": 0}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
