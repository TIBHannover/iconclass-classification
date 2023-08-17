import os
import sys
import re
import argparse
import logging
import numpy as np
import json

from hierarchical.datasets.datapipes import *
from hierarchical.datasets.utils import read_dict_data, read_line_data
from typing import Dict, Iterator, Union, Tuple, Any
from torchdata.datapipes.iter import FileLister, FileOpener, IterDataPipe


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("--mapping_path")
    parser.add_argument("--classifier_path")
    parser.add_argument("--output_path")
    parser.add_argument("--annotation_path")
    args = parser.parse_args()
    return args


class IconclassDecoderPipeline(IterDataPipe):
    def __init__(self, annotation=None):
        self.annotation = annotation

    def __iter__(self) -> Iterator[Dict]:
        for id, anno in self.annotation.items():
            out_sample = {
                "image_data": "",
                "id": id,
                "path": "",
            }

            yield {**out_sample, **anno}


def main():
    args = parse_args()
    mapping = read_line_data(args.mapping_path, dict_key="id")
    mapping_list = read_line_data(args.mapping_path)
    classifier = read_line_data(args.classifier_path)
    annotation = read_line_data(args.annotation_path, dict_key="id")

    dp = IconclassDecoderPipeline(annotation)
    dp = dp.build_flat_target(mapping)
    dp = dp.build_yolo_target(mapping, classifier)

    label_sum_yolo = None
    label_sum_flat = None
    count = 0
    for d in dp:
        count += 1
        if label_sum_yolo is None:
            label_sum_yolo = d["yolo_target"]
        else:
            label_sum_yolo = np.sum(np.stack([d["yolo_target"], label_sum_yolo], axis=0), axis=0)

        if label_sum_flat is None:
            label_sum_flat = d["flat_target"]
        else:
            label_sum_flat = np.sum(np.stack([d["flat_target"], label_sum_flat], axis=0), axis=0)
        print(count)
    with open(args.output_path, "w") as f:
        for x in mapping_list:
            f.write(
                json.dumps(
                    {
                        **x,
                        "count": {
                            "yolo": label_sum_yolo[x["index"]].item(),
                            "flat": label_sum_flat[x["index"]].item(),
                        },
                        # "weight": {"yolo": {weights[0, x["index"]].item()}, "flat": {}},
                    }
                )
                + "\n"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
