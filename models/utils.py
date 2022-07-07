from os import posix_fadvise
import numpy as np
import torch
import argparse
from datasets.utils import get_element

from typing import List, Union

from torch import Tensor, LongTensor


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    # assert current >= 0 and rampup_length >= 0, 'linear_rampup'
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length, "cosine_rampdown"
    return float(0.5 * (np.cos(np.pi * current / rampdown_length) + 1))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def gen_filter_mask(mappings, threshold, key="count.yolo"):
    mask = np.zeros(len(mappings), dtype=np.float32)
    for m in mappings:
        count = int(get_element(m, key))
        if count > threshold:
            mask[m["index"]] = 1

    return mask


def map_to_level_ontology(ontology_target, ontology_levels, max_level=8):
    assert ontology_target.shape[-1] == ontology_target.shape[-1]
    targets = []
    for x in range(max_level):
        target = ontology_target[..., ontology_levels[0, ...] == x].reshape(*ontology_target.shape[:-1], -1)
        targets.append(target)

    return targets


def map_classifier_to_flat_ontology(ontology_target, ontology_levels, level=0):
    mapping = torch.arange(ontology_levels.shape[-1])

    target = torch.zeros(
        *ontology_target.shape[:-1],
        ontology_levels.shape[-1],
        dtype=ontology_target.dtype,
        device=ontology_target.device,
    )
    m = mapping[ontology_levels[0, ...] == level]
    target[..., m] = ontology_target

    return target


def map_to_flat_ontology(ontology_target, ontology_levels, max_level=8):
    mapping = torch.arange(ontology_levels.shape[-1])
    target = None
    for i, t in enumerate(ontology_target):
        if target is None:
            target = torch.zeros(*t.shape[:-1], ontology_levels.shape[-1], dtype=t.dtype, device=t.device)
        m = mapping[ontology_levels[0, ...] == i]
        target[..., m] = t

    return target


def add_sequence_tokens_to_index(seq, add_start=True, add_stop=False, pad_token=-1, dim=-1):
    # move everything to right to add space for start and stop
    seq += 2
    seq[seq == (pad_token + 2)] = 0

    # pad values
    if add_start:
        padding_values = torch.ones(seq.shape[:-1] + torch.Size([1]), dtype=torch.int32, device=seq.device)
        seq = torch.cat([padding_values, seq], dim=dim)

    if add_stop:
        padding_values = torch.zeros(seq.shape[:-1] + torch.Size([1]), dtype=torch.int32, device=seq.device)
        seq = torch.cat([seq, padding_values], dim=dim)
    return seq


def add_sequence_tokens_to_level_ontology_target(ontology_target, ontology_mask):
    result_target = []
    for i, x in enumerate(ontology_target):
        padding_values = (torch.sum(ontology_mask[i], dim=-1) == 0).type(torch.int32)
        padding_values = torch.unsqueeze(padding_values, -1)

        x = torch.cat([padding_values, torch.zeros_like(padding_values), x], dim=-1)
        result_target.append(x)

    return result_target


def add_sequence_tokens_to_level_ontology(ontology_target, value=1):
    result_target = []
    for i, x in enumerate(ontology_target):
        padding_values = torch.ones(*x.shape[:-1], 2, dtype=x.dtype, device=x.device) * value

        target = torch.cat([padding_values, ontology_target[i]], dim=-1)
        result_target.append(target)
    return result_target


def del_sequence_tokens_from_level_ontology(ontology_target):
    result_target = []
    for t in ontology_target:
        result_target.append(t[..., 2:])
    return result_target


def build_level_map(mapping):
    level_map = torch.zeros(len(mapping), dtype=torch.int64)
    for m in mapping:
        level_map[m["index"]] = len(m["parents"])

    return level_map


def local_to_global_mapping_config(mapping):
    # map local indexes in each level(path) to global indexes
    mapto_global_id = dict()
    for v in mapping:
        local_index = tuple(v["token_id_sequence"])
        mapto_global_id[local_index] = v["index"] + 2
    return mapto_global_id


def mapping_local_global_idx(mapping, seq):
    next_seq_global = []
    for s in seq:
        next_seq_global.append(mapping[tuple(s.tolist())])
    return next_seq_global


def check_seq_path(token_id, seq, mapping_global_indexes):
    # checking the [seq,token_id] sequence exists in the mapping_global_indexes dict

    flag = True
    try:
        seq = seq.tolist()
        seq.append(token_id.item())
        mapping_global_indexes[(tuple(seq))]
    except KeyError:
        flag = False
    return flag


class HierarchicalLevelMapper:
    def __init__(self, mapping: List, classifier: List) -> None:
        self.level_map = build_level_map(mapping)
        # print(classifier)
        self.mapping_map = {x["index"]: x for x in mapping}
        self.classifier_map = {x["index"]: x for x in classifier}

    def to_flat(self, ontology: Tensor, level: int, remove_tokens: bool = True) -> Tensor:
        # print("####")
        # print(ontology.shape)
        if remove_tokens:
            ontology = ontology[..., 2:]
        # print(ontology.shape)
        # print(f"level_map {self.level_map.shape}")
        # print(self.level_map.shape[-1])
        mapping = torch.arange(self.level_map.shape[-1])

        m = mapping[self.level_map == level]
        # print(f"mapping")
        # print(mapping.shape)
        # print(mapping)
        # print(m)
        # print(m.shape)
        # print(m)
        flat = torch.zeros(*ontology.shape[:-1], self.level_map.shape[-1], dtype=ontology.dtype, device=ontology.device)
        # print(flat.shape)
        flat[..., m] = ontology

        return flat

    def classifier_mask_from_parent(self, classifiers: Union[List[int], LongTensor], remove_tokens: bool = True):
        # print(classifiers)
        if isinstance(classifiers, Tensor):
            classifiers = classifiers.cpu().detach().numpy().tolist()
        result = []
        for x in classifiers:
            mask = torch.zeros(len(self.mapping_map))
            if remove_tokens:
                if x == 1:
                    x = None
                else:
                    x = x - 2

            if x is None:
                classifier = self.classifier_map[x]
                mask[classifier["range"][0] : classifier["range"][1]] = 1
            else:
                mapping = self.mapping_map[x]
                if mapping["id"] in self.classifier_map:
                    classifier = self.classifier_map[mapping["id"]]
                    mask[classifier["range"][0] : classifier["range"][1]] = 1

            self.mapping_map
            result.append(mask)

        return torch.stack(result, dim=0)

        # one_hot = torch.zeros(len(self.mapping))
        # mask = torch.zeros(len(self.mapping))

        #     classifier = self.classifier_map[p]
        #     ranges.append([classifier["range"][0], classifier["range"][1]])
        #     mask[classifier["range"][0] : classifier["range"][1]] = 1

    def pad_id(self):
        return 0


class HierarchicalUnroller:
    def __init__(self):
        pass

    def __call__(self):
        pass