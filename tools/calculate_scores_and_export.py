import os
import sys
import re
import argparse

import numpy as np
import h5py
from tqdm import tqdm
from sklearn.metrics import average_precision_score

import logging
import torch
import json
import msgpack
import multiprocessing as mp


def split_dict(data_dict, splits):
    data = flat_dict(data_dict)
    result = [{}] * len(splits)
    for k, v in data.items():
        result_list = torch.split(v, splits)
        for i in range(len(splits)):
            result[i][k] = result_list[i]

    return [unflat_dict(x) for x in result]


def unflat_dict(data_dict, parse_json=False):
    result_map = {}
    if parse_json:
        data_dict_new = {}
        for k, v in data_dict.items():
            try:
                data = json.loads(v)
                data_dict_new[k] = data
            except:
                data_dict_new[k] = v
        data_dict = data_dict_new
    for k, v in data_dict.items():
        path = k.split(".")
        prev = result_map
        for p in path[:-1]:
            if p not in prev:
                prev[p] = {}
            prev = prev[p]
        prev[path[-1]] = v
    return result_map


def flat_dict(data_dict, parse_json=False):
    result_map = {}
    for k, v in data_dict.items():
        if isinstance(v, dict):
            embedded = flat_dict(v)
            for s_k, s_v in embedded.items():
                s_k = f"{k}.{s_k}"
                if s_k in result_map:
                    logging.error(f"flat_dict: {s_k} alread exist in output dict")

                result_map[s_k] = s_v
            continue

        if k not in result_map:
            result_map[k] = []
        result_map[k] = v
    return result_map


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


def read_dict_data(path: str, dict_key: str = None, keep_keys: str = None):
    file_data = []
    if path.endswith(".jsonl"):
        with open(path, "r") as f:
            return json.loads(f.read)

    if path.endswith(".msg"):

        with open(path, "rb") as f:
            return msgpack.unpackb(f.read())

    data = []
    for d in file_data:
        if keep_keys is not None:
            d = {k: get_element(d, v) for k, v in keep_keys.items()}
        data.append(d)

    if dict_key is not None:
        data = {get_element(x, dict_key): x for x in data}

    return data


def read_line_data(path: str, dict_key: str = None, keep_keys: str = None):
    file_data = []
    if path.endswith(".jsonl"):
        with open(path, "r") as f:
            for line in f:
                d = json.loads(line)
                file_data.append(d)

    if path.endswith(".msg"):

        with open(path, "rb") as f:
            u = msgpack.Unpacker(f)
            for line in u:
                d = json.loads(line)
                file_data.append(d)
    data = []
    for d in file_data:
        if keep_keys is not None:
            d = {k: get_element(d, v) for k, v in keep_keys.items()}
        data.append(d)

    if dict_key is not None:
        data = {get_element(x, dict_key): x for x in data}

    return data


def write_jsonl_lb_mapping(path, outpath):
    # read icon meta file and write the mapping file
    data = read_line_data(path)
    with open(outpath, mode="a") as f:
        for i in data:
            json_rec = json.dumps({i["id"]: i["kw"]})
            f.write(json_rec + "\n")


def read_jsonl_lb_mapping(path):
    # read label mapping file
    # output a dictionary key:id value:labels
    data = dict()
    with open(path, "r") as f:
        for line in f:
            d = json.loads(line)
            data.update(d)

    return data


def gen_filter_mask(mappings, threshold, key="count.yolo"):
    mask = np.zeros(len(mappings), dtype=np.float32)
    for m in mappings:
        # print(m)
        count = int(get_element(m, key))
        if count > threshold:
            mask[m["index"]] = 1

    return mask


def build_level_map(mapping):
    level_map = np.zeros(len(mapping), dtype=np.int64)
    for m in mapping:
        level_map[m["index"]] = len(m["parents"])

    return level_map


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-p", "--prediction_path", required=True, help="path to the prediction h5 file")
    parser.add_argument("-o", "--output_path", required=True, help="path to the prediction h5 file")
    parser.add_argument("-f", "--filter", nargs="+", type=int, default=[0], help="path to the prediction h5 file")
    parser.add_argument("--mapping_path", type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    mapping_config = []
    if args.mapping_path is not None:
        mapping_config = read_line_data(args.mapping_path)

    level_map = build_level_map(mapping_config)
    max_level = np.max(level_map)
    # exit()
    index_mapping_lut = {x["index"]:x for x in mapping_config}
    prediction_file = h5py.File(args.prediction_path, "r")
    # print(prediction_file.keys())
    with open(args.output_path, "w") as file:
        possible_predictions = ["ontology", "clip", "yolo", "flat"]
        for possible_prediction in possible_predictions:
            if possible_prediction not in prediction_file:
                continue
            # print(possible_prediction)
            prediction = prediction_file[possible_prediction]
            target = prediction_file["target"]

            target_sum = np.sum(target, axis=0)
            # print(target_sum.shape)

            ap = average_precision_score(target, prediction, average=None)
            print(ap)

            ap_mask = target_sum > 0
            for x in range(prediction.shape[1]):
                if target_sum[x] > 0:
                    # print(mapping_config[x])
                    ap_mask[x] = 1
            score = np.sum(np.nan_to_num(ap) * ap_mask) / np.sum(ap_mask)
            num_classes = np.sum(ap_mask)
            file.write(
                json.dumps(
                    {
                        "head": possible_prediction,
                        "ap": score.item(),
                        "num_classes": int(num_classes.item()),
                        "min_train_label": None,
                        "level": None,
                    }
                )
                + "\n"
            )
            # print(f"f: {score} {num_classes}")

            for level in range(max_level):
                level_mask = (level_map == level).astype(np.int64)

                scores = np.nan_to_num(ap) * ap_mask * level_mask
                num_classes = np.sum(level_mask * ap_mask)
                for x in np.argsort(-scores):
                    if scores[x] > 0.0:
                        m = index_mapping_lut[x]
                        # print()
                        # print(x)
                        # print(f"\tf: {f}, {score} {num_classes}")
                        file.write(
                            json.dumps(
                                {
                                    "id": m["id"],
                                    "score": scores[x].item(),
                                    "level": level,
                                }
                            )
                            + "\n"
                        )

    return 0


if __name__ == "__main__":
    sys.exit(main())