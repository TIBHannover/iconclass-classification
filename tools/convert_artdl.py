import os
import sys
import re
import json
import uuid
import argparse
import random

from hierarchical.utils.notation import Notation, generate_iconclass

from hierarchical.utils.read_data import read_line_data


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("--brill_path")

    parser.add_argument("--mapping_path")
    parser.add_argument("--classifier_path")

    parser.add_argument("--test_frac", type=float)
    parser.add_argument("--val_frac", type=float)

    parser.add_argument("--train_path")
    parser.add_argument("--val_path")
    parser.add_argument("--test_path")
    args = parser.parse_args()
    return args


def read_brill(brill_path):
    samples = {}
    iconclasses = []
    with open(brill_path) as f:
        data = json.load(f)
        for img, annotations in data.items():
            # print(img, annotations)
            samples[img] = annotations
            for anno in annotations:
                iconclasses.append(anno)

    return samples, iconclasses


def main():
    args = parse_args()

    mapping = read_line_data(args.mapping_path, dict_key="id")

    brill_samples, brill_classes = read_brill(args.brill_path)
    dataset_len = len(brill_samples)
    # exit()
    new_samples = []
    for sample, annotations in brill_samples.items():
        new_labels = []
        for label in annotations:
            sample_iconclasses = generate_iconclass(label)
            for s in sample_iconclasses:
                if s is None:
                    continue

                if s in mapping:
                    new_labels.append(s)
                else:
                    try:
                        for x in Notation(s).strip_key().get_parents_until()[::-1]:
                            if x in mapping:
                                new_labels.append(x)
                                break
                    except:
                        print("###############")
                        print(Notation(label))

        new_labels = list(set(new_labels))

        ids = []
        all_ids = []

        cls_ids = []
        all_cls_ids = []
        classes = []
        for annotation in new_labels:
            m = mapping[annotation]
            classes.append(m["id"])

            ids.append(m["index"])
            all_ids.append(m["index"])

            cls_ids.append(m["classifier_index"])
            all_cls_ids.append(m["classifier_index"])

            # tree_paths.append(m['tree_path'])
            # all_tree_paths.append(m['tree_path'])
            # print(m)
            # exit()
            p = m["parent"]
            while p != None:
                # print(p)
                k = mapping[p]
                p = k["parent"]

                all_ids.append(k["index"])
                all_cls_ids.append(k["classifier_index"])
                # all_tree_paths.append(k['tree_path'])
            all_ids = list(set(all_ids))
            all_cls_ids = list(set(all_cls_ids))

        # print(image_annotation)
        # print(ids)
        # print(all_ids)
        # print(cls_ids)
        # print(all_cls_ids)
        # exit()
        new_samples.append(
            {
                "name": sample,
                "id": uuid.uuid4().hex,
                "classes": classes,
                "ids": ids,
                "all_ids": all_ids,
                "cls_ids": cls_ids,
                "all_cls_ids": all_cls_ids,
            }
        )
    # if args.random:
    random.seed(1337)
    random.shuffle(new_samples)
    test_size = int(round(args.test_frac * dataset_len))
    val_size = int(round(args.val_frac * dataset_len))
    with open(args.test_path, "w") as f:
        test_split = new_samples[:test_size]
        print(len(test_split))
        for x in test_split:
            f.write(json.dumps(x) + "\n")

    with open(args.val_path, "w") as f:
        val_split = new_samples[test_size : test_size + val_size]
        print(len(val_split))
        for x in val_split:
            f.write(json.dumps(x) + "\n")

    with open(args.train_path, "w") as f:
        train_split = new_samples[test_size + val_size :]
        print(len(train_split))
        for x in train_split:
            f.write(json.dumps(x) + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
