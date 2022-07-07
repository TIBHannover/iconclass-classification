from .strategy import *
from .log import *

import torch


def move_to_device(data, device):
    if isinstance(data, dict):
        result_dict = {}
        for k, v in data.items():
            result_dict[k] = move_to_device(v, device)
        return result_dict

    elif isinstance(data, (list, set)):
        result_list = []
        for v in data:
            result_list.append(move_to_device(v, device))
        return result_list

    elif isinstance(data, torch.Tensor):
        return data.to(device)

    else:
        return data