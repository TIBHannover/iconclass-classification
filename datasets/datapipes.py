from sklearn.utils import shuffle
import torch
import logging
import random

from torchdata.datapipes.iter import FileLister, FileOpener, IterDataPipe
from torchdata.datapipes import functional_datapipe

from typing import Dict, Iterator, Union, Tuple, Any

import torchvision
import imageio
import numpy as np


@functional_datapipe("iconclass_text")
class IconclassTextGenerator(IterDataPipe):
    def __init__(
        self,
        dp,
        labels: Dict = None,
        shuffle: bool = False,
        join: str = None,
        num_labels: int = None,
        input_field: str = None,
        output_field: str = None,
    ):
        self.dp = dp
        self.labels = labels
        self.num_labels = num_labels
        self.input_field = input_field if input_field else "classes"
        self.output_field = output_field if output_field else "txt"
        self.shuffle = shuffle
        self.join = join if join else "+ "

    def __iter__(self):

        for sample in self.dp:
            classes = sample.get(self.input_field, None)

            if classes is None:
                logging.warning(f"Input field {self.input_field} is missing")
                continue

            if not isinstance(classes, (list, set)):
                classes = list(classes)

            if self.num_labels and self.num_labels > 0:
                if self.shuffle:
                    classes = random.sample(classes, min(self.num_labels, len(classes)))
                else:
                    classes = classes[: min(self.num_labels, len(classes))]

            class_texts = []
            for c in classes:
                class_text = self.labels.get(c, None)
                if class_text:
                    class_texts.append(class_text)

            classes_text = self.join.join(class_texts)

            yield {**sample, self.output_field: classes_text}


@functional_datapipe("iconclass_externel_text")
class IconclassExternalTextGenerator(IterDataPipe):
    def __init__(
        self,
        dp,
        labels: Dict = None,
        shuffle: bool = False,
        input_field: str = None,
        output_field: str = None,
    ):
        self.dp = dp
        self.labels = labels
        self.input_field = input_field if input_field else "id"
        self.output_field = output_field if output_field else "txt"
        self.shuffle = shuffle

    def __iter__(self):

        for sample in self.dp:
            sample_key = sample.get(self.input_field, None)

            class_texts = self.labels.get(sample_key)

            if class_texts is None:
                class_texts = [""]
                # print(sample_key, flush=True)
                # print(sample, flush=True)
                # exit()
            if shuffle:
                txt = random.choice(class_texts)
            else:
                txt = class_texts[0]
            # print(txt)

            yield {**sample, self.output_field: txt}


@functional_datapipe("tokenize_openclip")
class TokenizeOpenClip(IterDataPipe):
    def __init__(self, dp, input_field: str = None, output_field: str = None) -> None:
        self.dp = dp
        self.input_field = input_field if input_field else "txt"
        self.output_field = output_field if output_field else "clip_embedding"

    def __iter__(self) -> Iterator[Dict]:
        import open_clip

        for sample in self.dp:
            embedding = torch.squeeze(open_clip.tokenize([sample.get(self.input_field)]))
            yield {**sample, self.output_field: embedding}


@functional_datapipe("load_from_msg")
class MsgPackLoader(IterDataPipe):
    def __init__(self, dp) -> None:
        self.dp = dp

    def __iter__(self) -> Iterator[Dict]:
        import msgpack

        for path in self.dp:
            with open(path, "rb") as f:

                unpacker = msgpack.Unpacker(f, max_buffer_size=1024 * 1024 * 1024, raw=True)

                for x in unpacker:
                    yield x


@functional_datapipe("decode_iconclass")
class IconclassDecoderPipeline(IterDataPipe):
    def __init__(self, dp, annotation=None):
        self.dp = dp
        self.annotation = annotation

    def __iter__(self) -> Iterator[Dict]:

        for sample in self.dp:
            out_sample = {
                "image_data": sample[b"image"],
                "id": sample[b"id"].decode("utf-8"),
                "path": sample[b"path"].decode("utf-8"),
            }

            if b"additional" in sample:
                out_sample.update({"additional": sample[b"additional"]})

            if out_sample["id"] not in self.annotation:
                logging.info(f"Dataset: {out_sample['id']} not in annotation")
                return None
            else:
                anno = self.annotation[out_sample["id"]]

            yield {**out_sample, **anno}


@functional_datapipe("build_flat_target")
class FlatTargetBuilder(IterDataPipe):
    def __init__(self, dp, mapping=None):
        self.dp = dp
        self.mapping = mapping

    def __iter__(self) -> Iterator[Dict]:

        for sample in self.dp:
            y_onehot_flat = torch.zeros(len(self.mapping))

            for c in sample["classes"]:
                if c in self.mapping:
                    y_onehot_flat.scatter_(0, torch.tensor(self.mapping[c]["index"]), 1)
            yield {**sample, "flat_target": y_onehot_flat}


@functional_datapipe("build_yolo_target")
class YOLOTargetBuilder(IterDataPipe):
    def __init__(self, dp, mapping=None, classifier=None):
        self.dp = dp
        self.mapping = mapping
        self.classifier = classifier

    def __iter__(self) -> Iterator[Dict]:

        for sample in self.dp:
            result = {}
            y_onehot_yolo_labels = torch.zeros(len(self.mapping))
            for x in sample["all_ids"]:
                y_onehot_yolo_labels.scatter_(0, torch.tensor(x), 1)
            result["yolo_target"] = y_onehot_yolo_labels

            y_onehot_yolo_classes = torch.zeros(len(self.classifier))
            for x in sample["all_cls_ids"]:
                y_onehot_yolo_classes.scatter_(0, torch.tensor(x), 1)
            result["yolo_classes"] = y_onehot_yolo_classes

            y_onehot_yolo_labels_weights = torch.zeros(len(self.mapping))
            for x in sample["all_cls_ids"]:
                for y in range(self.classifier[x]["range"][0], self.classifier[x]["range"][1]):
                    # print(y)
                    y_onehot_yolo_labels_weights.scatter_(0, torch.tensor(y), 1)
            result["yolo_target_mask"] = y_onehot_yolo_labels_weights

            yield {**sample, **result}


@functional_datapipe("build_ontology_target")
class OntologyTargetBuilder(IterDataPipe):
    def __init__(
        self,
        dp,
        mapping=None,
        classifier=None,
        filter_label_by_count=None,
        ontology=None,
        classifier_map=None,
        merge_one_hot=None,
        random_trace=None,
        max_traces=None,
        level_map=None,
    ):
        self.dp = dp
        self.mapping = mapping
        self.classifier = classifier
        self.filter_label_by_count = filter_label_by_count
        self.ontology = ontology
        self.classifier_map = classifier_map
        self.merge_one_hot = merge_one_hot
        self.random_trace = random_trace
        self.max_traces = max_traces
        self.level_map = level_map

    def __iter__(self) -> Iterator[Dict]:

        for sample in self.dp:
            if len(sample["classes"]) == 0:
                # continue
                # ontology_mask torch.Size([64, 4, 21484])
                # ontology_target torch.Size([64, 4, 21484])
                # ontology_indexes torch.Size([64, 4, 8])
                # ontology_ranges torch.Size([64, 4, 8, 2])
                # ontology_trace_mask torch.Size([64, 4])
                # ontology_levels torch.Size([64, 21484])

                yield {
                    **sample,
                    "ontology_mask": torch.zeros([0, len(self.mapping)], dtype=torch.int32),
                    "ontology_target": torch.zeros([0, len(self.mapping)]),
                    "ontology_indexes": torch.zeros([0, 8], dtype=torch.int32),
                    "ontology_ranges": torch.zeros([0, 8, 2], dtype=torch.int32),
                    "ontology_trace_mask": torch.zeros([0], dtype=torch.int32),
                    "ontology_levels": self.level_map,
                }
                continue
            ontology_target = []
            ontology_mask = []
            ontology_trace_mask = []
            ontology_ranges = []
            ontology_indexes = []
            for c in sample["classes"]:
                token_id_sequence = self.mapping[c]["token_id_sequence"]

                token_sequence = self.mapping[c]["parents"] + [c]
                if self.filter_label_by_count is not None and self.filter_label_by_count > 0:
                    counts = [self.mapping[x]["count"]["yolo"] for x in token_sequence]
                    ignore_labels = [False if x > self.filter_label_by_count else True for x in counts]
                else:
                    ignore_labels = [False for x in token_sequence]

                # build vectors and masks
                parents = [None] + self.mapping[c]["parents"]
                ranges = []
                one_hot = torch.zeros(len(self.mapping))
                mask = torch.zeros(len(self.mapping))
                for i, p in enumerate(parents):
                    if not ignore_labels[i]:
                        classifier = self.classifier_map[p]
                        ranges.append([classifier["range"][0], classifier["range"][1]])
                        mask[classifier["range"][0] : classifier["range"][1]] = 1
                        one_hot[self.mapping[token_sequence[i]]["index"]] = 1

                for _ in range(len(ranges), len(self.ontology)):
                    ranges.append([0, 0])

                ontology_mask.append(mask)
                ontology_target.append(one_hot)
                ontology_ranges.append(torch.tensor(ranges, dtype=torch.int32))
                ontology_trace_mask.append(torch.ones([], dtype=torch.int32))

                # build indexes sequence
                indexes = []
                for i, t in enumerate(token_sequence):
                    if not ignore_labels[i]:
                        indexes.append(self.mapping[t]["index"])

                for _ in range(len(indexes), len(self.ontology)):
                    indexes.append(-1)
                ontology_indexes.append(torch.tensor(indexes, dtype=torch.int32))

            # add ones form other traces
            if self.merge_one_hot:
                for i, m in enumerate(ontology_mask):
                    target = ontology_target[i]
                    for j, m2 in enumerate(ontology_mask):
                        target = target + m * ontology_mask[j] * ontology_target[j]
                    target[target > 0] = 1
                    ontology_target[i] = target

            # randomly shuffle everything
            if self.random_trace:
                ontology_mask, ontology_target, ontology_indexes, ontology_ranges, ontology_trace_mask = zip(
                    *random.sample(
                        list(
                            zip(ontology_mask, ontology_target, ontology_indexes, ontology_ranges, ontology_trace_mask)
                        ),
                        k=len(ontology_mask),
                    )
                )

            if self.max_traces is not None and self.max_traces > 0:
                ontology_mask = ontology_mask[: self.max_traces]
                ontology_target = ontology_target[: self.max_traces]
                ontology_indexes = ontology_indexes[: self.max_traces]
                ontology_ranges = ontology_ranges[: self.max_traces]
                ontology_trace_mask = ontology_trace_mask[: self.max_traces]
            yield {
                **sample,
                "ontology_mask": torch.stack(ontology_mask, dim=0),
                "ontology_target": torch.stack(ontology_target, dim=0),
                "ontology_indexes": torch.stack(ontology_indexes, dim=0),
                "ontology_ranges": torch.stack(ontology_ranges, dim=0),
                "ontology_trace_mask": torch.stack(ontology_trace_mask, dim=0),
                "ontology_levels": self.level_map,
            }


import torchvision.transforms.functional as F


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, "constant")


@functional_datapipe("augment_strong_image")
class SrongImageAugmenter(IterDataPipe):
    def __init__(self, dp, output_size=384, min_size=None):
        self.dp = dp
        self.output_size = output_size
        self.min_size = min_size
        self.transformation = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                SquarePad(),
                torchvision.transforms.Resize(512),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandAugment(),
                torchvision.transforms.RandomResizedCrop(output_size, scale=[0.7, 1.0]),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __iter__(self) -> Iterator[Dict]:

        for sample in self.dp:
            image = imageio.imread(sample["image_data"])

            if self.min_size is not None and self.min_size > 0:

                if image.shape[0] < self.min_size or image.shape[1] < self.min_size:

                    continue
            image = self.transformation(image)
            yield {**sample, "image": image}


@functional_datapipe("augment_weak_image")
class WeakImageAugmenter(IterDataPipe):
    def __init__(self, dp, output_size=384, min_size=None):
        self.dp = dp
        self.output_size = output_size
        self.min_size = min_size
        self.transformation = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                SquarePad(),
                torchvision.transforms.Resize(512),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomResizedCrop(output_size, scale=[0.7, 1.0]),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __iter__(self) -> Iterator[Dict]:

        for sample in self.dp:
            image = imageio.imread(sample["image_data"])

            if self.min_size is not None and self.min_size > 0:

                if image.shape[0] < self.min_size or image.shape[1] < self.min_size:

                    continue
            image = self.transformation(image)
            yield {**sample, "image": image}


@functional_datapipe("val_image")
class ValImage(IterDataPipe):
    def __init__(self, dp, output_size=384, min_size=None):
        self.dp = dp
        self.output_size = output_size
        self.min_size = min_size
        self.transformation = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                SquarePad(),
                torchvision.transforms.Resize(output_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __iter__(self) -> Iterator[Dict]:

        for sample in self.dp:
            image = imageio.imread(sample["image_data"])

            if self.min_size is not None and self.min_size > 0:

                if image.shape[0] < self.min_size or image.shape[1] < self.min_size:

                    continue
            image = self.transformation(image)
            yield {**sample, "image": image}


@functional_datapipe("clean_sample")
class SampleCleaner(IterDataPipe):
    def __init__(self, dp, keys=None):
        self.dp = dp
        self.keys = keys

    def __iter__(self) -> Iterator[Dict]:

        for sample in self.dp:
            for key in self.keys:
                if key in sample:
                    del sample[key]
            yield {**sample}