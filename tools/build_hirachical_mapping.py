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
    parser.add_argument("--max_level")
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
            level_id = d["id"]
            parents = d["p"]
            parent_id_without_prefix = []

            if len(parents) > 0:
                parent_id_without_prefix = [parents[0]]
                for parent_index in range(1, len(parents)):
                    parent_id = parents[parent_index]
                    for parent in parent_id_without_prefix:
                        parent_id = re.sub(f"^{re.escape(parent)}", "", parent_id)
                    parent_id_without_prefix.append(parent_id)

            for parent in parent_id_without_prefix:
                level_id = re.sub(f"^{re.escape(parent)}", "", level_id)

            data.append(
                {
                    "id": d["id"],
                    "level_id": level_id,
                    "parents": parents,
                    "parents_without_prefix": parent_id_without_prefix,
                    "token_sequnce": parent_id_without_prefix + [level_id],
                }
            )

            if max_level < len(parents) + 1:
                max_level = len(parents) + 1

    tokenizer_level = [set() for x in range(max_level)]

    for d in data:
        parents = d["parents_without_prefix"]
        for i, p in enumerate(parents):
            tokenizer_level[i].add(p)

        tokenizer_level[len(parents)].add(d["level_id"])

    for i, t in enumerate(tokenizer_level):
        t = ["#PAD", "#START"] + list(sorted(t))
        tokenizer_level[i] = t

    if args.output_mapping_path is not None:
        with open(args.output_mapping_path, "w") as f:
            for d in data:
                token_id_sequnce = [tokenizer_level[i].index(x) for i, x in enumerate(d["token_sequnce"])]
                # print({**d, "token_id_sequnce": token_id_sequnce})
                f.write(json.dumps({**d, "token_id_sequnce": token_id_sequnce}) + "\n")

    if args.output_classifier_path is not None:

        with open(args.output_classifier_path, "w") as f:
            for i, t in enumerate(tokenizer_level):
                f.write(json.dumps({"index": i, "depth": i, "tokenizer": t}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
