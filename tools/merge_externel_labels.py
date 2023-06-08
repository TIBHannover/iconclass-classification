import os
import sys
import re
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-i", "--input_paths", nargs="+")
    parser.add_argument("-o", "--output_path")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    id_descriptions = {}
    for path in args.input_paths:
        print(path)
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                # print(d)
                if d["id"] not in id_descriptions:
                    id_descriptions[d["id"]] = []
                id_descriptions[d["id"]].extend(d["descriptions"])
    print(len(id_descriptions))

    with open(args.output_path, "w") as f:
        for id, descriptions in id_descriptions.items():
            f.write(json.dumps({"id": id, "descriptions": descriptions}) + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
