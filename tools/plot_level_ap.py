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
    parser.add_argument("-f", "--filter", default=0, type=int, help="verbose output")
    parser.add_argument("-o", "--output_path", help="verbose output")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    fig, ax = plt.subplots(figsize=(6, 4))

    assert len(args.labels) == len(args.input_paths)

    for l, p in zip(args.labels, args.input_paths):
        data = []
        print(l, p)
        with open(p) as f:
            for line in f:
                d = json.loads(line)
                if d.get("min_train_label", 0) == args.filter and d.get("level") is not None:
                    data.append(d)
        data = sorted(data, key=lambda x: x["level"])

        levels = [x["level"] for x in data]
        aps = [x["ap"] for x in data]

        ax.plot(levels, aps, label=l)

    ax.set_xlabel("Level of Granularity")
    ax.set_ylabel("mAP")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(args.output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
