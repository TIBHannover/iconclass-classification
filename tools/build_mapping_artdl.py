from genericpath import exists
import os
import sys
import re
import argparse

import json
import torch
import msgpack
import logging
import uuid


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


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")

    parser.add_argument("-a", "--annotation_path", help="verbose output")
    parser.add_argument("-m", "--mapping_path", help="verbose output")
    parser.add_argument("-c", "--classifier_path", help="verbose output")
    parser.add_argument("-o", "--output_path", help="verbose output")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    annotation_dict = {"train": {}, "val": {}, "test": {}}

    mapping = read_line_data(args.mapping_path, dict_key="id")
    classifier = read_line_data(args.classifier_path, dict_key="name")

    class_map = {
        "11F(MARY)": "11F",
        # "11HH(CATHERINE)": "11H(CATHERINE)",
        # "11HH(MARY MAGDALENE)": "11H(MARY MAGDALENE)",
        # "11HH(BARBARA)": "11H(BARBARA)",
    }

    for root, dirs, files in os.walk(args.annotation_path):
        for f in files:
            file_path = os.path.join(root, f)
            m = re.match(r"^(.*?)_(.*?)\..*?$", f)

            if not m:
                # print(f)
                continue
            set_name = m.group(2)
            class_name = m.group(1)

            with open(file_path) as f:
                for line in f:
                    filename, is_shown = line.split()
                    is_shown = int(is_shown)

                    if filename not in annotation_dict[set_name]:
                        annotation_dict[set_name][filename] = []

                    if is_shown <= 0:
                        continue
                    if class_name in class_map:
                        annotation_dict[set_name][filename].append(class_map[class_name])
                    else:
                        annotation_dict[set_name][filename].append(class_name)

    os.makedirs(args.output_path, exist_ok=True)
    for image_set, v in annotation_dict.items():
        output_path = os.path.join(args.output_path, f"{image_set}.jsonl")
        with open(output_path, "w") as f:
            for filename, classes in v.items():
                filtered_classes = [cls for cls in classes if cls in mapping]
                # all parents
                all_parents = []
                for x in [mapping[c]["parents"] for c in filtered_classes]:
                    all_parents.extend(x)

                all_parents = list(set(all_parents))

                all_cls_ids = [0] + [classifier[p]["index"] for p in all_parents]

                # only the last parent
                last_parents = [mapping[c]["parents"][-1] for c in filtered_classes]
                last_parents = list(set(last_parents))

                cls_ids = [classifier[p]["index"] for p in last_parents]

                result = {
                    "id": uuid.uuid4().hex,
                    "classes": filtered_classes,
                    "name": filename,
                    "ids": [mapping[c]["index"] for c in filtered_classes],
                    "all_ids": [mapping[c]["index"] for c in all_parents]
                    + [mapping[c]["index"] for c in filtered_classes],
                    "cls_ids": cls_ids,
                    "all_cls_ids": all_cls_ids,
                }
                f.write(json.dumps(result) + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
