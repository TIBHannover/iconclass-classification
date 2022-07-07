import re
import os
import logging
import random

import msgpack
import json


def get_element(data_dict: dict, path: str, split_element="."):
    if path is None:
        return data_dict

    if callable(path):
        elem = path(data_dict)

    if isinstance(path, str):
        elem = data_dict
        try:
            for x in path.strip(split_element).split(split_element):
                try:
                    x = int(x)
                    elem = elem[x]
                except ValueError:
                    elem = elem.get(x)
        except:
            pass

    if isinstance(path, (list, set)):
        elem = [get_element(data_dict, x) for x in path]

    return elem


def read_jsonl(path, dict_key=None, keep_keys=None):
    data = []
    with open(path, "r") as f:
        for line in f:
            d = json.loads(line)
            if keep_keys is not None:
                d = {k: get_element(d, v) for k, v in keep_keys.items()}
            data.append(d)

    if dict_key is not None:
        data = {get_element(x, dict_key): x for x in data}

    return data


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
            yield {
                **path,
                "rel_path": os.path.relpath(path["path"], os.path.commonpath([self.path, path["path"]])),
                "name": os.path.splitext(os.path.basename(path["path"]))[0],
            }
