import os
import sys
import re
import argparse

import random

import json
import logging

from utils import read_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-a", "--annotation", help="verbose output")
    parser.add_argument("-o", "--output", help="path to dir of images")
    parser.add_argument("-m", "--mapping", help="path to dir of images")
    parser.add_argument("-c", "--classifiers", help="path to dir of images")
    parser.add_argument("-y", "--yolo_coco_map", help="path to dir of images")
    parser.add_argument("-n", "--yolo_coco_names", help="path to dir of images")

    parser.add_argument("-r", "--slice", type=str, default="::1", help="data slice to stored")

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

    result = {}
    for k, v in data.items():
        if "bbox" not in v:
            continue
        result[k] = v

    return result


def main():
    args = parse_args()

    random.seed(1337)

    data = read_coco_annotations(args.annotation)

    data_slice = slice(*[int(x) if x else None for x in args.slice.split(":")])
    print(len(data))
    data = {k: v for k, v in [(k, v) for k, v in data.items()][data_slice]}
    print(len(data))

    mapping = read_jsonl(args.mapping, dict_key="index")
    classifiers = read_jsonl(args.classifiers, dict_key="id")

    yolo_coco_map = {}
    with open(args.yolo_coco_map, "r") as f:
        for i, line in enumerate(f):
            yolo_coco_map[i] = int(line)

    yolo_coco_names = {}
    with open(args.yolo_coco_names, "r") as f:
        for i, line in enumerate(f):
            yolo_coco_names[line.strip()] = i

    annotations = []
    for _, x in data.items():
        if "bbox" not in x:
            continue

        classes = []
        for b, l, c in zip(x["bbox"], x["classes_label"], x["classes"]):
            y_coco_index = yolo_coco_names[l]
            y_9000_index = yolo_coco_map[y_coco_index]
            m = mapping[y_9000_index]

            classes.append({"bbox": b, "class": m["index"], "name": m["name"], "id": m["id"], "index": m["index"]})

        annotations.append({"id": x["id"], "path": x["path"], "classes": classes})

    with open(args.output, "w") as f:
        for anno in annotations:
            f.write(json.dumps(anno) + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
