import torch
import torch.nn.functional as F
import numpy as np

import re

try:
    from torch._six import container_abcs, string_classes, int_classes
except:
    import collections.abc as container_abcs

    int_classes = int
    string_classes = str


np_str_obj_array_pattern = re.compile(r"[SaUO]")


default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, " "dicts or lists; found {}"
)


def pad_tensor(input, max_shape, value=None):
    if value is None:
        value = 0
    max_shape = torch.tensor(max_shape, dtype=torch.int64)
    if len(max_shape) == 0:
        return input

    padding_size = torch.LongTensor(max_shape - torch.tensor(input.shape))

    padding_size = torch.LongTensor([[0, x] for x in padding_size.tolist()[::-1]]).view(2 * max_shape.shape[0]).tolist()
    result = F.pad(input, padding_size, value=value)

    return result


def pad_collate(batch, pad_values=None):
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        max_shape = np.amax(list(map(lambda x: list(x.shape) if getattr(x, "shape", None) else [], batch)), axis=0)

        batch = list(map(lambda x: pad_tensor(x, max_shape, pad_values), batch))
        out = None

        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == "numpy" and elem_type.__name__ != "str_" and elem_type.__name__ != "string_":
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return pad_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):

        result_dict = {}
        for key in elem:
            padded_sub_dict = pad_collate([d[key] for d in batch], pad_values[key] if key in pad_values else None)
            result_dict.update({key: padded_sub_dict})
        return result_dict
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(pad_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # TODO find a better way
        if pad_values is not None:
            # max_len = 0
            # for x in batch:
            #     batch_len = len(x)
            #     if batch_len > max_len:
            #         max_len = batch_len
            return batch
        else:
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError("each element in list of batch should be of equal size")
            transposed = zip(*batch)
            return [pad_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


class PadCollate:
    def __init__(self, pad_values=None):
        self.pad_values = pad_values

    def __call__(self, batch):
        return pad_collate(batch, self.pad_values)
