import msgpack

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


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-i", "--image_path", help="verbose output")
    parser.add_argument("-a", "--annotation", help="verbose output")
    parser.add_argument("-o", "--output", help="path to dir of images")
    parser.add_argument("-p", "--process", type=int, default=8)
    parser.add_argument("-c", "--chunck", type=int, default=1024, help="Images per file")

    parser.add_argument("-s", "--shuffle", action="store_true", default=True, help="verbose output")

    args = parser.parse_args()
    return args


class MsgPackWriter:
    def __init__(self, msg_path, chunck_size=1024, **kwargs):

        self.path = os.path.abspath(msg_path)
        self.chunck_size = chunck_size

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        shards_re = r"shard_(\d+).msg"

        self.shards_index = [
            int(re.match(shards_re, x).group(1)) for x in os.listdir(self.path) if re.match(shards_re, x)
        ]
        self.shard_open = None
        self.current_shard = 0

    def open_next(self):

        if len(self.shards_index) == 0:
            next_index = 0
        else:
            next_index = sorted(self.shards_index)[-1] + 1
        self.shards_index.append(next_index)
        self.current_shard = next_index

        if self.shard_open is not None and not self.shard_open.closed:
            self.shard_open.close()
        self.count = 0
        self.shard_open = open(os.path.join(self.path, f"shard_{next_index}.msg"), "wb")
        logging.info(f"OPEN NEXT {self.path}  {self.count}  {self.chunck_size}")

    def __enter__(self):
        self.open_next()
        return self

    def __exit__(self, type, value, tb):
        self.shard_open.close()

    def handle(self, data_dict):
        # print(f"{self.path}  {self.count}  {self.chunck_size}")
        if self.count >= self.chunck_size:
            self.open_next()

        self.shard_open.write(msgpack.packb(data_dict))
        self.count += 1


class ImageDataloader:
    def __init__(self, path, filter=None, shuffle=True):
        image_re = re.compile(r".*?\.(png|tif|tiff|jpg|jpeg|gif)", re.IGNORECASE)
        self.path = path

        if os.path.isdir(path):
            self.paths = [{"path": os.path.join(p, x)} for p, _, f in os.walk(path) for x in f if image_re.match(x)]
            if filter is not None:
                filter_re = re.compile(filter)
                self.paths = [x for x in self.paths if filter_re.match(x["path"])]
        else:
            self.paths = [{"path": path}]

        if shuffle:
            logging.info("Shuffle paths")
            random.shuffle(self.paths)

    def __len__(self):
        return len(self.paths)

    def __iter__(self):
        for path in self.paths:
            yield {**path, "rel_path": os.path.relpath(path["path"], os.path.commonpath([self.path, path["path"]]))}


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
            data[d["rel_path"]] = d
    return data


def main():
    args = parse_args()

    level = logging.ERROR
    if args.verbose:
        level = logging.INFO

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=level)

    image_loader = ImageDataloader(args.image_path, shuffle=args.shuffle)

    whitelist = None
    if args.annotation:
        whitelist = read_annotations(args.annotation)

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
            image_loader[i]["id"] = whitelist[match[1] + match[5]]["id"]

        logging.info(len(image_loader))

    with mp.Pool(args.process) as pool:
        with MsgPackWriter(args.output) as f:
            start = time.time()
            image_loader = [{**x} for x in image_loader]
            for i, x in enumerate(pool.imap(read_image_process, image_loader)):
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

