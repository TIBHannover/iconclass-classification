import os
import sys
import re
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("--mapping_path")
    parser.add_argument("--output_mapping_path")
    parser.add_argument("--output_classifier_path")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # split parent paths in single tokens
    max_level = 0
    data = []
    with open(args.mapping_path, "r") as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            data.append(
                {"id": d["id"],}
            )

    tokenizer_level = set()

    for d in data:
        tokenizer_level.add(d["id"])

    # for i, t in enumerate(tokenizer_level):
    #     t = list(sorted(t))
    tokenizer_level = list(sorted(tokenizer_level))

    if args.output_mapping_path is not None:
        with open(args.output_mapping_path, "w") as f:
            for d in data:
                f.write(json.dumps({**d, "class_id": tokenizer_level.index(d["id"])}) + "\n")

    if args.output_classifier_path is not None:

        with open(args.output_classifier_path, "w") as f:
            for i, t in enumerate([tokenizer_level]):
                f.write(json.dumps({"index": i, "depth": i, "tokenizer": list(t)}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
