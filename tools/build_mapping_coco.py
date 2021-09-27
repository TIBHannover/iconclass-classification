import imageio

import os
import sys
import re
import argparse

import numpy as np
import time

import json


import logging


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")

    parser.add_argument("-a", "--annotation", help="verbose output")
    parser.add_argument("-o", "--output", help="path to dir of images")

    parser.add_argument("-t", "--tree", help="verbose output")
    parser.add_argument("-n", "--names", help="verbose output")
    parser.add_argument("-l", "--labels", help="verbose output")

    args = parser.parse_args()
    return args


def read_coco_annotations(annotation_path):
    data = {}
    logging.info(f"Read annotation: {annotation_path}")
    with open(annotation_path, "r") as f:
        d = json.loads(f.read())

        logging.info(f"Read images")
        for sample in d["images"]:
            data[sample["file_name"]] = {"id": sample["id"], "path": sample["file_name"]}
        data_lut = {v["id"]: k for k, v in data.items()}

        logging.info(f"Read categories")
        categories_lut = {x["id"]: x["name"] for x in d["categories"]}

        logging.info(f"Read annotations")
        for anno in d["annotations"]:
            image_path = data_lut[anno["image_id"]]
            if "bbox" not in data[image_path]:
                data[image_path]["bbox"] = []
                data[image_path]["classes"] = []
                data[image_path]["classes_label"] = []

            data[image_path]["classes"].append(anno["category_id"])
            data[image_path]["classes_label"].append(categories_lut[anno["category_id"]])
            data[image_path]["bbox"].append(anno["bbox"])

    return data


def read_yolo_tree(tree_path, names_path, labels_path):
    tree = {}

    names_lut = []
    with open(names_path, "r") as f:
        for line in f:
            names_lut.append(line.strip())

    with open(tree_path, "r") as f:
        tree_list = []
        for i, line in enumerate(f):
            tree_list.append([line.split()[0], int(line.split()[1]), i])

        for i, t in enumerate(tree_list):

            if t[1] < 0:
                parent = None
            else:
                parent = tree_list[t[1]][0]

            parents = []
            p = parent
            while p is not None:
                parents.append(p)

                if p is None or p not in tree:
                    p = None
                else:
                    p = tree[p]["parent"]

            tree[t[0]] = {
                "id": t[0],
                "index": t[2],
                "parent": parent,
                "name": names_lut[t[2]],
                "parents": parents[::-1],
            }

    return tree


def main():
    args = parse_args()

    level = logging.ERROR
    if args.verbose:
        level = logging.INFO

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=level)

    tree = read_yolo_tree(args.tree, args.names, args.labels)

    classifier_lut = {}
    for i, node in tree.items():
        parent = node["parent"]
        if parent not in classifier_lut:
            classifier_lut[parent] = []
        classifier_lut[parent].append(node)

    classifiers = []
    classifier_lut_2 = {}
    for i, (label, concepts) in enumerate(classifier_lut.items()):
        indexes = [x["index"] for x in concepts]

        start = min(indexes)
        stop = max(indexes) + 1

        if (stop - start) != len(set(indexes)):
            print(name)
        if label is None:
            name = None
            id = None
            parent = {"name": None, "index": None}
        else:
            name = tree[label]["name"]
            id = tree[label]["id"]
            parent = tree[label]["parent"]
            parent = classifier_lut_2[parent]
            parent = {"name": parent["name"], "index": parent["index"]}

        d = {
            "index": i,
            "range": [start, stop],
            "name": name,
            "id": id,
            "parent": parent,
            "depth": len(concepts[0]["parents"]),
        }

        classifier_lut_2[id] = d
        classifiers.append(d)

    for c in classifiers[0:10]:
        print(c)

    ontology = {}

    for c in classifiers:
        if c["depth"] not in ontology:
            ontology[c["depth"]] = {"index": c["depth"], "depth": c["depth"], "tokenizer": []}

    for c in tree.values():
        ontology[len(c["parents"])]["tokenizer"].append(c["name"])

    os.makedirs(args.output, exist_ok=True)

    with open(os.path.join(args.output, "classifiers.jsonl"), "w") as f:
        for x in classifiers:
            f.write(json.dumps(x) + "\n")

    with open(os.path.join(args.output, "mapping.jsonl"), "w") as f:
        for x in tree.values():
            f.write(json.dumps(x) + "\n")

    with open(os.path.join(args.output, "ontology.jsonl"), "w") as f:
        for x in ontology.values():
            f.write(json.dumps(x) + "\n")

    print(ontology)
    return 0


if __name__ == "__main__":
    sys.exit(main())
