import imageio

import os
import sys
import re
import argparse

import numpy as np
import time

import json

import multiprocessing as mp
import random

import logging

from utils import MsgPackWriter
from utils import ImageDataloader


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-i", "--image_path", help="verbose output")
    parser.add_argument("-a", "--annotation", help="verbose output")
    parser.add_argument("-o", "--output", help="path to dir of images")
    parser.add_argument("-p", "--process", type=int, default=8)
    parser.add_argument("-c", "--chunck", type=int, default=1024, help="Images per file")
    parser.add_argument("-r", "--slice", type=str, default="::1", help="data slice to stored")

    parser.add_argument("-s", "--shuffle", action="store_true", default=True, help="verbose output")

    args = parser.parse_args()
    return args


def read_image(path):
    try:
        image = imageio.imread(path)
    except:
        return None

    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)

    if image.shape[-1] == 4:
        image = image[:, :, 0:3]

    return image


def read_image_process(args):
    try:
        path = args["path"]
        id = args["id"]
        rel_path = args["rel_path"]

        image = read_image(path)
        result = {
            "id": id,
            "image": imageio.imwrite(imageio.RETURN_BYTES, image, format="jpg", quality=95),
            "path": rel_path,
        }

        return result
    except KeyboardInterrupt as e:
        raise
    except Exception as e:
        logging.error(e)

        return None


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

    level = logging.ERROR
    if args.verbose:
        level = logging.INFO

    random.seed(1337)

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=level)

    image_loader = ImageDataloader(args.image_path, shuffle=args.shuffle)
    whitelist = None
    if args.annotation:
        whitelist = read_coco_annotations(args.annotation)
        data_slice = slice(*[int(x) if x else None for x in args.slice.split(":")])
        print(len(whitelist))
        whitelist = {k: v for k, v in [(k, v) for k, v in whitelist.items()][data_slice]}
        print(len(whitelist))

        logging.info(len(image_loader))

        def filter_white(a):
            match = re.match(r"^((.*?/)*([^_]*))(.*)[\.]*(\..*?)$", a["rel_path"])
            if not match:
                return False
            if match[1] + match[5] in whitelist:
                return True
            return False

        image_loader = list(filter(filter_white, image_loader))

        for i, x in enumerate(image_loader):
            match = re.match(r"^((.*?/)*([^_]*))(.*)[\.]*(\..*?)$", x["rel_path"])
            if whitelist[match[1] + match[5]]["id"] == 404462:
                print(whitelist[match[1] + match[5]])
            image_loader[i]["id"] = whitelist[match[1] + match[5]]["id"]

        logging.info(len(image_loader))

    # exit()
    count = 0
    with mp.Pool(args.process) as pool:
        with MsgPackWriter(args.output) as f:
            start = time.time()
            image_loader = [{**x} for x in image_loader]
            for i, x in enumerate(pool.imap(read_image_process, image_loader)):
                if x is None:
                    continue
                # print(i)
                f.handle(x)
                count += 1

                # f.write(json.dumps({'key': x['path'], 'path': x['path'], 'index': i}) + '\n')
                if i % 1000 == 0:
                    # txn.commit()
                    end = time.time()
                    logging.info(f"{i}: {1000/(end - start):.2f} image/s")
                    start = end

    print(count)

    return 0


if __name__ == "__main__":
    sys.exit(main())
