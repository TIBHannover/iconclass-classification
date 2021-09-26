from os import posix_fadvise
import numpy as np
import torch
import argparse
from datasets.utils import get_element


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