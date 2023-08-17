import os
import sys
import re
import argparse
import json
import logging
from hierarchical.utils.notation import Notation


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-i", "--input_path", help="path to the ndjson")
    parser.add_argument("-o", "--output_path", help="path for all result files")
    parser.add_argument("--iconclass_path")
    parser.add_argument("--train_path")
    parser.add_argument("--val_path")
    parser.add_argument("--test_path")
    args = parser.parse_args()
    return args


def find_child(tree, id):
    for i, c in enumerate(tree["child"]):
        if c["id"] == id:
            return i
    return None


def parse_tree(tree_node: dict, class_map: dict, parent: dict = None):
    classifiers = []
    mappings = []
    classifier_index = 0
    class_index = 0
    range_index = 0

    if len(tree_node["child"]) < 1:
        return classifiers, mappings

    if parent is None:
        parent = {"name": None, "index": None}

    classifiers.append(
        {
            "index": classifier_index,
            "range": [range_index, range_index + len(tree_node["child"])],
            "parent": parent,
            "name": tree_node["id"],
            "depth": 0,
        }
    )

    mappings.extend(
        [
            {
                "id": x["id"],
                **class_map[x["id"]],
                "index": i,
                "classifier_index": classifier_index,
                "parent": tree_node["id"],
            }
            for i, x in enumerate(tree_node["child"])
        ]
    )

    range_index += len(tree_node["child"])
    classifier_index += 1

    # for c in tree_node['child']:

    for c in tree_node["child"]:
        sub_classifiers, sub_mappings = parse_tree(
            c, class_map, parent={"name": classifiers[0]["name"], "index": classifiers[0]["index"]}
        )

        class_index += 1
        if len(sub_classifiers) < 1:
            continue

        def update_mapping(a):
            a["index"] += range_index
            a["classifier_index"] += classifier_index
            return a

        sub_mappings = list(map(update_mapping, sub_mappings))
        mappings.extend(sub_mappings)

        def update_classifiers(a):
            a["index"] += classifier_index
            a["range"][0] += range_index
            a["range"][1] += range_index
            a["depth"] += 1
            return a

        sub_classifiers = list(map(update_classifiers, sub_classifiers))

        classifiers.extend(sub_classifiers)

        class_index += 1
        classifier_index += len(sub_classifiers)
        range_index += len(sub_mappings)

    return classifiers, mappings


def build_mapping(data: list):
    tree = {"id": None, "child": []}
    class_info = {}
    for i, entry in enumerate(data):
        print(entry)
        if len(entry.keys()) != 1:
            raise ValueError()
        entry_id = list(entry.keys())[0]
        entry_data = entry[entry_id]
        depth = len(entry_data["p"])

        # Find parent node and build tree
        parent_node = tree
        if depth == 0:
            parent = None
        else:
            parent = sorted(entry_data["p"], key=lambda a: len(a))
            for p in parent:
                index = find_child(parent_node, p)
                if index is None:
                    parent_node["child"].append({"id": p, "child": []})
                parent_node = parent_node["child"][find_child(parent_node, p)]

        index = find_child(parent_node, entry_id)

        if index is not None:
            parent_node["child"][index]
        else:
            parent_node["child"].append({"id": entry_id, "child": []})

        # build class map
        class_info[entry_id] = entry[entry_id]

    classifiers, mappings = parse_tree(tree, class_info)

    index_lut = {x["name"]: x["index"] for x in classifiers}
    print(index_lut)

    def update_classifiers(a):
        if a["parent"]["name"] is not None:
            a["parent"]["index"] = index_lut[a["parent"]["name"]]
        return a

    classifiers = list(map(update_classifiers, classifiers))

    logging.info("Tree building done")

    return classifiers, mappings


def build_image_annotation(mappings: list, classifiers: list):
    image_dict = {}
    for m in mappings:
        if "ex" not in m:
            continue

        for example in m["ex"]:
            if example not in image_dict:
                image_dict[example] = []
            image_dict[example].append(m["id"])
    return image_dict


def read_annotation_file(path: str):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))

    classifiers, mappings = build_mapping(data)

    images = build_image_annotation(mappings, classifiers)

    return classifiers, mappings, images


def read_image_folder(path: str):
    result = []
    for root, dirs, files in os.walk(path):
        for f in files:
            file_path = os.path.join(root, f)
            result.append(
                {
                    "name": os.path.splitext(os.path.basename(file_path))[0],
                    "path": file_path,
                    "rel_path": os.path.relpath(file_path, os.path.commonpath([path, file_path])),
                }
            )

    return result


def write_mappings(path: str, classifiers: list, mappings: list):
    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, "mapping.jsonl"), "w") as f:
        for mapping in mappings:
            f.write(json.dumps(mapping) + "\n")

    with open(os.path.join(path, "classifiers.jsonl"), "w") as f:
        for classifier in classifiers:
            f.write(json.dumps(classifier) + "\n")


def bind_info(images: list, classifiers: list, mappings: list, images_annotation: dict):
    # tmp_mapping_index = {k['index']: k for k in mappings}
    # tmp_mapping_tree_path = {k['tree_path']: k for k in mappings}
    tmp_mappings_index = {k["id"]: k for k in mappings}

    result = []

    for image in images:
        if image["name"] not in images_annotation:
            continue

        image_annotation = images_annotation[image["name"]]

        ids = []
        all_ids = []

        cls_ids = []
        all_cls_ids = []
        classes = []
        for annotation in image_annotation:
            m = tmp_mappings_index[annotation]
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
                k = tmp_mappings_index[p]
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
        result.append(
            {
                **image,
                "classes": classes,
                "ids": ids,
                "all_ids": all_ids,
                "cls_ids": cls_ids,
                "all_cls_ids": all_cls_ids,
            }
        )
    return result


def write_image_info(path: str, train: list, val: list, test: list):
    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, "train.jsonl"), "w") as f:
        for x in train:
            f.write(json.dumps(x) + "\n")

    with open(os.path.join(path, "val.jsonl"), "w") as f:
        for x in val:
            f.write(json.dumps(x) + "\n")

    with open(os.path.join(path, "test.jsonl"), "w") as f:
        for x in test:
            f.write(json.dumps(x) + "\n")


def read_duplicate(path: str):
    result = {}
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)
            # print(data)

            result[os.path.splitext(os.path.basename(data["path"]))[0]] = [
                os.path.splitext(os.path.basename(x["path"]))[0] for x in data["similarities"]
            ]

    return result


def remove_duplicate(duplicate: dict, image_annotation: dict):
    results = {}
    removed = {}  # dict copy to original
    for i, (key, value) in enumerate(image_annotation.items()):
        if key not in duplicate:
            results[key] = set(value)

        elif key in removed:
            # Add concepts to first image
            source = removed[key]
            for x in value:
                results[source].add(x)

        elif key in duplicate:
            # New duplicate founded
            results[key] = set(value)
            if key not in removed:
                removed[key] = key
            for x in duplicate[key]:
                removed[x] = key

    return results


# def main():
#     args = parse_args()

#     level = logging.ERROR
#     if args.verbose:
#         level = logging.DEBUG

#     logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=level)

#     logging.info("Read annotation files")
#     classifiers, mappings, image_anno = read_annotation_file(args.annotation)

#     logging.info(f"Images in Annotations {len(image_anno.keys())}")
#     duplicate = {}
#     if args.duplicate is not None:
#         duplicate = read_duplicate(args.duplicate)
#         # print(duplicate)
#         image_anno = remove_duplicate(duplicate, image_anno)

#         logging.info(f"Images in Annotations after duplicate removed {len(image_anno.keys())}")
#     # exit()

#     logging.info("Read image files")
#     image_list = read_image_folder(args.image)

#     write_mappings(args.output, classifiers, mappings)

#     image_list = bind_info(image_list, classifiers, mappings, image_anno)

#     for i in range(len(image_list)):
#         image_list[i].update({"id": uuid.uuid4().hex})

#     if args.random:
#         random.seed(1337)
#         random.shuffle(image_list)

#     val_entries = int(float(args.val_split) * len(image_list))
#     val = image_list[:val_entries]

#     logging.info(f"Val set size: {len(val)}")

#     test_entries = int(float(args.test_split) * len(image_list))
#     test = image_list[val_entries : val_entries + test_entries]

#     logging.info(f"Test set size: {len(test)}")

#     train = image_list[val_entries + test_entries :]
#     logging.info(f"Train set size: {len(train)}")

#     write_image_info(args.output, train, val, test)

#     return 0


# if __name__ == "__main__":
#     sys.exit(main())


def generate_iconclass_trace(iconclass_annotation):
    results = []
    notation = Notation(iconclass_annotation)
    if not notation.is_valid():
        # check if there is a sep
        if ":" in iconclass_annotation:
            for sub_iconclass_annotation in iconclass_annotation.split(":"):
                results.extend(generate_iconclass_trace(sub_iconclass_annotation))

        return results

    results.extend([x.code for x in notation.strip_key().get_parents_until()])
    results.append(notation.strip_key().code)

    return results


def generate_iconclass(iconclass_annotation):
    results = []
    notation = Notation(iconclass_annotation)
    if not notation.is_valid():
        # check if there is a sep
        if ":" in iconclass_annotation:
            for sub_iconclass_annotation in iconclass_annotation.split(":"):
                results.extend(generate_iconclass(sub_iconclass_annotation))

        return results

    results.append(notation.strip_key().code)

    return results


def main():
    args = parse_args()

    # text_dict = {}
    # with open(args.iconclass_path) as f:
    #     for line in f:
    #         notation = line[2:]
    #         if len(notation) <= 0:
    #             continue
    #         text_dict[notation.strip()] = 1

    text_dict = {}
    for root, dirs, files in os.walk(args.iconclass_path):
        for f in files:
            file_path = os.path.join(root, f)
            if "refs" in file_path:
                continue
            with open(file_path, "r") as f:
                for line in f:
                    m = re.match(r"^(.*?)\|(.*?)$", line)
                    if not m:
                        print(line)
                        continue
                    if m.group(1) not in text_dict:
                        text_dict[m.group(1)] = m.group(2)
                    else:
                        text_dict[m.group(1)] = f"{text_dict[m.group(1)]}, {m.group(2)}"

    # print(len(text_dict))
    # exit()

    samples = {}
    iconclasses = []
    with open(args.input_path) as f:
        for line in f:
            data = json.loads(line)
            if "themes" in data:
                samples[data["id"]] = data["themes"]
                iconclasses.extend(data["themes"])

    iconclasses = list(set(iconclasses))
    all_iconclasses = []

    for iconclass in iconclasses:
        all_iconclasses.extend(generate_iconclass_trace(iconclass))

    all_iconclasses = list(set(all_iconclasses))

    # print(all_iconclasses)
    print(len(all_iconclasses))

    filtered_iconclasses = [x for x in all_iconclasses if x in text_dict]
    print(len(filtered_iconclasses))

    extend_all_parents = []

    for iconclass in filtered_iconclasses:
        extend_all_parents.append(iconclass)
        extend_all_parents.extend([x.code for x in Notation(iconclass).strip_key().get_parents_until()])

    extend_all_parents_set = set(extend_all_parents)

    extend_all_parents = list(extend_all_parents_set)

    old_annotation = []
    for iconclass in extend_all_parents:
        old_annotation.append({iconclass: {"p": [x.code for x in Notation(iconclass).strip_key().get_parents_until()]}})

    old_annotation = sorted(old_annotation, key=lambda x: list(x.keys())[0])

    classifiers, mappings = build_mapping(old_annotation)
    print(classifiers[:10])
    print(len(mappings))
    write_mappings(args.output_path, classifiers, mappings)
    # images = build_image_annotation(mappings, classifiers)

    tmp_mappings_index = {k["id"]: k for k in mappings}

    # print(filtered_iconclasses)

    # with_text = [x for x in all_iconclasses if Notation(x).has_text()]
    # print(len(with_text))
    if args.train_path:
        with open(args.train_path) as f, open(os.path.join(args.output_path, "train.jsonl"), "w") as f_out:
            for line in f:
                data = json.loads(line)
                if data["name"] not in samples:
                    logging.error(data)
                    return
                sample = samples[data["name"]]
                if not sample:
                    print(data)
                new_labels = []
                for label in sample:
                    sample_iconclasses = generate_iconclass(label)
                    for s in sample_iconclasses:
                        if s is None:
                            print(data)
                            continue

                        if s in extend_all_parents_set:
                            new_labels.append(s)
                        else:
                            try:
                                for x in Notation(s).strip_key().get_parents_until()[::-1]:
                                    if x in extend_all_parents_set:
                                        new_labels.append(x)
                                        break
                            except:
                                print("###############")
                                print(Notation(label))
                                print(data)
                ids = []
                all_ids = []

                cls_ids = []
                all_cls_ids = []
                classes = []
                for annotation in new_labels:
                    m = tmp_mappings_index[annotation]
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
                        k = tmp_mappings_index[p]
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
                f_out.write(
                    json.dumps(
                        {
                            **data,
                            "classes": classes,
                            "ids": ids,
                            "all_ids": all_ids,
                            "cls_ids": cls_ids,
                            "all_cls_ids": all_cls_ids,
                        }
                    )
                    + "\n"
                )
    if args.val_path:
        with open(args.val_path) as f, open(os.path.join(args.output_path, "val.jsonl"), "w") as f_out:
            for line in f:
                data = json.loads(line)
                if data["name"] not in samples:
                    logging.error(data)
                    return
                sample = samples[data["name"]]
                if not sample:
                    print(data)
                new_labels = []
                for label in sample:
                    sample_iconclasses = generate_iconclass(label)
                    for s in sample_iconclasses:
                        if s is None:
                            print(data)
                            continue

                        if s in extend_all_parents_set:
                            new_labels.append(s)
                        else:
                            try:
                                for x in Notation(s).strip_key().get_parents_until()[::-1]:
                                    if x in extend_all_parents_set:
                                        new_labels.append(x)
                                        break
                            except:
                                print("###############")
                                print(Notation(label))
                                print(data)
                ids = []
                all_ids = []

                cls_ids = []
                all_cls_ids = []
                classes = []
                for annotation in new_labels:
                    m = tmp_mappings_index[annotation]
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
                        k = tmp_mappings_index[p]
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
                f_out.write(
                    json.dumps(
                        {
                            **data,
                            "classes": classes,
                            "ids": ids,
                            "all_ids": all_ids,
                            "cls_ids": cls_ids,
                            "all_cls_ids": all_cls_ids,
                        }
                    )
                    + "\n"
                )
    if args.test_path:
        with open(args.test_path) as f, open(os.path.join(args.output_path, "test.jsonl"), "w") as f_out:
            for line in f:
                data = json.loads(line)
                if data["name"] not in samples:
                    logging.error(data)
                    return
                sample = samples[data["name"]]
                if not sample:
                    print(data)
                new_labels = []
                for label in sample:
                    sample_iconclasses = generate_iconclass(label)
                    for s in sample_iconclasses:
                        if s is None:
                            print(data)
                            continue

                        if s in extend_all_parents_set:
                            new_labels.append(s)
                        else:
                            try:
                                for x in Notation(s).strip_key().get_parents_until()[::-1]:
                                    if x in extend_all_parents_set:
                                        new_labels.append(x)
                                        break
                            except:
                                print("###############")
                                print(Notation(label))
                                print(data)
                ids = []
                all_ids = []

                cls_ids = []
                all_cls_ids = []
                classes = []
                for annotation in new_labels:
                    m = tmp_mappings_index[annotation]
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
                        k = tmp_mappings_index[p]
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
                f_out.write(
                    json.dumps(
                        {
                            **data,
                            "classes": classes,
                            "ids": ids,
                            "all_ids": all_ids,
                            "cls_ids": cls_ids,
                            "all_cls_ids": all_cls_ids,
                        }
                    )
                    + "\n"
                )
                # print(new_labels)
                # exit()

                # print(data)
    # print(samples)
    # exit()

    return 0


if __name__ == "__main__":
    sys.exit(main())
