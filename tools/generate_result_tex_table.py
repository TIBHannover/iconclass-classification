import os
import sys
import re
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-i", "--input_paths", nargs="+", help="verbose output")
    parser.add_argument("-l", "--labels", nargs="+", help="verbose output")
    parser.add_argument("-f", "--filter", nargs="+", default=[0, 10, 100, 1000], type=int, help="verbose output")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    assert len(args.labels) == len(args.input_paths)

    for l, p in zip(args.labels, args.input_paths):
        line_strings = [f"{l}"]

        with open(p) as f:
            for line in f:
                d = json.loads(line)
                if d.get("min_train_label", 0) in args.filter and d.get("level") is None:

                    line_strings.append(f"\\num{{{round(d['ap'], 4)}}}")
        print(" & ".join(line_strings))
    return 0


if __name__ == "__main__":
    sys.exit(main())