#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copy from openai repo
"""

import torch
from typing import Any, Union, List
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

# _tokenizer = _Tokenizer()


def tokenize(
    tokenizer, texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False
) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)
    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all CLIP models use 77 as the context length
    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length
    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    # move from global variable to argument
    _tokenizer = tokenizer

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, : len(tokens)] = torch.tensor(tokens)

    return result


# aa=" invidia present accusing irreverence biasimo honouring detrattione abstract betraying deceit maledicenza mocking conduct truth vitioso calumny ears accused remorse apelles treachery personification calunnia repenting confronted ass della derisione morality ignorance suspicion ignorantia innocent bad idea throne judge envy allegory dispreggio slander ripa virtù judgement"
# aa = aa.replace(", ", " ")
# print(aa.count(" ")+1)
# # bb = tokenize(aa)
# cc = [49406, 512, 603, 3357, 2881, 41474, 582, 3115, 1553, 717, 3596, 2080, 37754, 561, 635, 49265, 637, 10197, 18436, 910, 3534, 585, 1662, 25184, 25239, 37870, 11647, 4319, 85, 760, 21285, 1198, 1622, 1350, 9861, 9434, 515, 25282, 688, 24431, 46005, 1692, 3510, 35755, 1198, 569, 3098, 515, 1501, 694, 20033, 775, 5301, 22986, 839, 7233, 637, 34133, 22735, 34910, 7653, 3120, 320, 11241, 2103, 3612, 15999, 5656, 21017, 7456, 1985, 11073, 5472, 8050, 82, 9666, 553, 2217, 1790, 83, 127, 373, 25246, 49407]
# for i in cc:
#     print(_tokenizer.decode([i]))
# import json

# def get_element(data_dict: dict, path: str, split_element="."):
#     if path is None:
#         return data_dict

#     if callable(path):
#         elem = path(data_dict)

#     if isinstance(path, str):
#         elem = data_dict
#         try:
#             for x in path.strip(split_element).split(split_element):
#                 try:
#                     x = int(x)
#                     elem = elem[x]
#                 except ValueError:
#                     elem = elem.get(x)
#         except:
#             pass

#     if isinstance(path, (list, set)):
#         elem = [get_element(data_dict, x) for x in path]

#     return elem

# def read_jsonl(path, dict_key=None, keep_keys=None):
#     data = []
#     with open(path, "r") as f:
#         for line in f:
#             d = json.loads(line)
#             if keep_keys is not None:
#                 d = {k: get_element(d, v) for k, v in keep_keys.items()}
#             data.append(d)

#     if dict_key is not None:
#         data = {get_element(x, dict_key): x for x in data}

#     return data
# import numpy as np
# path = "/home/javad/Desktop/Dataset/iconclass/last_version/labels.jsonl"

# lb = read_jsonl(path, dict_key="id")
# lb_list = {}
# for k,v in lb.items():
#     vv = ", ".join(v["kw"]["en"]) + ", " + v["txt"]["en"]
#     lb_list[k] = vv

# # for i in lb_list.values():
# #     tokenize(i)
# aa = ["page","chelsea",  "death of St. Agatha"]
# bb = tokenize(["This is " + desc for desc in aa]).numpy()
# print(_tokenizer.decode(bb[0,:]))
