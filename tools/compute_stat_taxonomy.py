import os
import sys
import re
import argparse

from hierarchical.tools.utils import read_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-a", "--mapping_path", help="verbose output")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    yolo_count = 0
    flat_count = 0
    unique = 0
    data = read_jsonl(args.mapping_path)
    stat = {}

    for line in data:
        p = len(line["parents"])
        if p not in stat:
            stat[p] = 0
        stat[p] += 1

    print(stat)
    return 0


if __name__ == "__main__":
    sys.exit(main())
