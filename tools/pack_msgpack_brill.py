import imageio

import os
import sys
import re
import argparse

import numpy as np
import cv2
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
    parser.add_argument("-p", "--process", type=int, default=16)
    parser.add_argument("-c", "--chunck", type=int, default=1024, help="Images per file")

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
        result = {"id": id, "image": imageio.imwrite(imageio.RETURN_BYTES, image, format="jpg"), "path": rel_path}

        return result
    except KeyboardInterrupt as e:
        raise
    except Exception as e:
        logging.error(e)

        return None


def read_annotations(annotation_path):
    data = {}
    with open(annotation_path, "r") as f:
        for line in f:
            d = json.loads(line)
            data[d["name"]] = d
    return data


def main():
    args = parse_args()

    level = logging.ERROR
    if args.verbose:
        level = logging.INFO

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=level)

    annotation = read_annotations(args.annotation)
    # print(annotation)
    # exit()
    images_path = []
    for root, dirs, files in os.walk(args.image_path):
        for f in files:
            file_path = os.path.join(root, f)
            # name = os.path.splitext(f)[0]
            if f in annotation:
                images_path.append(
                    {
                        "path": file_path,
                        "rel_path": os.path.relpath(file_path, os.path.commonpath([args.image_path, file_path])),
                        "name": f,
                        "id": annotation[f]["id"],
                    }
                )

    with mp.Pool(args.process) as pool:
        with MsgPackWriter(args.output) as f:
            start = time.time()
            for i, x in enumerate(pool.imap(read_image_process, images_path)):
                if x is None:
                    continue
                # print(i)
                f.handle(x)

                # f.write(json.dumps({'key': x['path'], 'path': x['path'], 'index': i}) + '\n')
                if i % 1000 == 0:
                    # txn.commit()
                    end = time.time()
                    logging.info(f"{i}: {1000/(end - start):.2f} image/s")
                    start = end

    return 0


if __name__ == "__main__":
    sys.exit(main())
