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
            parents = d["p"]
            data.append({"id": d["id"], "parents": parents})

            if max_level < len(parents) + 1:
                max_level = len(parents) + 1

    tokenizer_level = [set() for x in range(max_level)]

    for d in data:
        tokenizer_level[len(d["parents"])].add(d["id"])

    tokenizer = []
    tokenizer_level_list = []
    for i, t in enumerate(tokenizer_level):
        t = list(sorted(t))
        tokenizer_level_list.append(t)

        tokenizer.extend(t)

    print(tokenizer[:20])
    # tokenizer = list(sorted(tokenizer))

    if args.output_mapping_path is not None:
        with open(args.output_mapping_path, "w") as f:
            for d in data:
                f.write(json.dumps({"id": d["id"], "class_id": tokenizer.index(d["id"])}) + "\n")

    if args.output_classifier_path is not None:

        with open(args.output_classifier_path, "w") as f:
            start = 0
            for i, t in enumerate(tokenizer_level_list):
                print(t[:5])
                f.write(
                    json.dumps({"index": i, "range": [start, start + len(t)], "depth": i, "tokenizer": list(t)}) + "\n"
                )
                start += len(t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
