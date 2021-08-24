# pip install git+https://github.com/openai/CLIP.git

import os
import sys
import re
import argparse

import clip
import json
import pickle
import logging

import torch
import numpy as np
from scipy.special import softmax

from PIL import Image

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


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


def get_model_name():
    model_name = "ViT-B/32"

    return model_name


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    return device


def get_model():
    model, _ = clip.load(get_model_name(), device=get_device())

    return model


def get_preprocess():
    size = get_model().input_resolution.item()

    transform = Compose(
        [
            Resize(size, interpolation=Image.BICUBIC),
            CenterCrop(size),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )

    return transform


def tokenize(cls):
    if isinstance(cls, str):
        cls = [cls]

    return [clip.tokenize(cl) for cl in cls]


def get_text_features(model, cls, normalize=True):
    text = torch.cat(tokenize(cls)).to(get_device())

    with torch.no_grad():
        features = model.encode_text(text)

    if normalize:
        features /= features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy()


def get_text_features_batch(data, batch_size=128):
    for i in range(0, len(data), batch_size):
        inputs = data[i : i + batch_size]

        yield get_text_features(inputs), inputs


def get_top_k(img_features, txt_features):
    if isinstance(img_features, list):
        img_features = np.array(img_features)

    features = img_features @ txt_features.T

    result = softmax(100.0 * features, axis=1)
    result = torch.from_numpy(result)

    top_k = torch.topk(result, k=25, axis=-1)

    values = top_k.values.tolist()
    indices = top_k.indices.tolist()

    return values, indices


# def get_top_k_batch(index_path, text, n=5000):
#     index = HNSWIndex().load(index_path)

#     txt_features = get_text_features_batch(text)
#     txt_features = list(zip(*txt_features))[0]
#     txt_features = np.vstack(txt_features)

#     paths, text = np.array(index.idx), np.array(text)
#     file_name = index_path.split(".")[0] + "_keywords"

#     with open(f"{file_name}.ndjson", "w") as file_obj:
#         for i in tqdm(range(0, len(index), n)):
#             idx = range(i, min(i + n, len(index)))
#             img_features = index.get(list(idx))

#             batch = get_top_k(img_features, txt_features)

#             for j, value in enumerate(zip(batch[0], batch[1])):
#                 result = {
#                     "path": paths[idx[j]],
#                     "annotations": {"values": list(value[0]), "labels": list(text[value[1]])},
#                 }

#                 file_obj.write(ujson.dumps(result) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-t", "--txt_path", help="verbose output")
    parser.add_argument("-o", "--output_path", help="verbose output")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model = get_model()
    data = read_jsonl(args.txt_path)
    with open(args.output_path, "wb") as f:
        text_mapping = []
        for i, x in enumerate(data):
            text_data = ", ".join(x["kw"]["en"]) + ", " + x["txt"]["en"]
            text_vec = get_text_features(model, text_data[:77])

            text_mapping.append({**x, "clip": text_vec})
            print(i)
            if i % 100 == 0:
                logging.info(i)

        pickle.dump(text_mapping, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
