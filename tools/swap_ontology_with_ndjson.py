import os
import sys
import re
import argparse
import json
from utils.notation import Notation


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-i", "--input_path", help="path to the ndjson")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    return 0


if __name__ == "__main__":
    sys.exit(main())
