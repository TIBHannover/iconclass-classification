import os
import sys
import re
import argparse
import json
import logging

from hierarchical.utils.notation import Notation


def generate_and_filter_iconclass(classes, iconclass_data):
    iconclasses = list(set(classes))
    all_iconclasses = []

    for iconclass in iconclasses:
        all_iconclasses.extend(generate_iconclass_trace(iconclass))

    all_iconclasses = list(set(all_iconclasses))

    # print(all_iconclasses)
    print(len(all_iconclasses))

    filtered_iconclasses = [x for x in all_iconclasses if x in iconclass_data]
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

    return sorted(old_annotation, key=lambda x: list(x.keys())[0])


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


def read_iconclass_data(iconclass_path):
    text_dict = {}
    for root, dirs, files in os.walk(iconclass_path):
        for f in files:
            file_path = os.path.join(root, f)
            if "refs" in file_path:
                continue
            with open(file_path, "r") as f:
                for line in f:
                    m = re.match(r"^(.*?)\|(.*?)$", line)
                    if not m:
                        continue
                    if m.group(1) not in text_dict:
                        text_dict[m.group(1)] = m.group(2)
                    else:
                        text_dict[m.group(1)] = f"{text_dict[m.group(1)]}, {m.group(2)}"
    return text_dict


def read_iconclass(iconclass_path):
    samples = {}
    iconclasses = []
    with open(iconclass_path) as f:
        for line in f:
            data = json.loads(line)
            if "themes" in data:
                samples[data["id"]] = data["themes"]
                iconclasses.extend(data["themes"])

    return samples, iconclasses


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


def read_artdl(artdl_path):
    samples = {}
    iconclasses = []
    for root, dirs, files in os.walk(artdl_path):
        for f in files:
            file_path = os.path.join(root, f)
            m = re.match(r"^(.*?)_(.*?)\..*?$", f)

            if not m:
                # print(f)
                continue
            set_name = m.group(2)
            class_name = m.group(1)
            iconclasses.append(class_name)

            with open(file_path) as f:
                for line in f:
                    filename, is_shown = line.split()
                    is_shown = int(is_shown)

                    if filename not in samples:
                        samples[filename] = []

                    if is_shown <= 0:
                        continue
                    samples[filename].append(class_name)

    return samples, iconclasses


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

    def update_classifiers(a):
        if a["parent"]["name"] is not None:
            a["parent"]["index"] = index_lut[a["parent"]["name"]]
        return a

    classifiers = list(map(update_classifiers, classifiers))

    logging.info("Tree building done")

    return classifiers, mappings


def write_mappings(path: str, classifiers: list, mappings: list):
    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, "mapping.jsonl"), "w") as f:
        for mapping in mappings:
            f.write(json.dumps(mapping) + "\n")

    with open(os.path.join(path, "classifiers.jsonl"), "w") as f:
        for classifier in classifiers:
            f.write(json.dumps(classifier) + "\n")


def write_iconclass(input_path, output_path, samples, iconclass_mapping):
    with open(input_path) as f, open(output_path, "w") as f_out:
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

                    if s in iconclass_mapping:
                        new_labels.append(s)
                    else:
                        try:
                            for x in Notation(s).strip_key().get_parents_until()[::-1]:
                                if x in iconclass_mapping:
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
                m = iconclass_mapping[annotation]
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
                    k = iconclass_mapping[p]
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


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--iconclass_data_path")
    parser.add_argument("--iconclass_train_path")
    parser.add_argument("--iconclass_val_path")
    parser.add_argument("--iconclass_test_path")
    parser.add_argument("--iconclass_path")
    parser.add_argument("--output_path")
    parser.add_argument("--brill_path")
    parser.add_argument("--artdl_path")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    iconclass_data = read_iconclass_data(args.iconclass_data_path)
    classes = []
    if args.iconclass_path:
        iconclass_samples, iconclass_classes = read_iconclass(args.iconclass_path)
        classes.extend(iconclass_classes)
    if args.brill_path:
        brill_samples, brill_classes = read_brill(args.brill_path)
        classes.extend(brill_classes)
    if args.artdl_path:
        artdl_samples, artdl_classes = read_artdl(args.artdl_path)
        classes.extend(artdl_classes)

    all_iconclasses = []

    for iconclass in list(set(classes)):
        all_iconclasses.extend(generate_iconclass_trace(iconclass))

    all_iconclasses = generate_and_filter_iconclass(all_iconclasses, iconclass_data)

    classifiers, mappings = build_mapping(all_iconclasses)
    write_mappings(args.output_path, classifiers, mappings)

    tmp_mappings_index = {k["id"]: k for k in mappings}
    print(all_iconclasses)
    print(len(all_iconclasses))

    if args.iconclass_train_path:
        iconclass_output_path = os.path.join(args.output_path, "iconclass")
        os.makedirs(iconclass_output_path, exist_ok=True)
        write_iconclass(
            args.iconclass_train_path,
            os.path.join(iconclass_output_path, "train.jsonl"),
            samples=iconclass_samples,
            iconclass_mapping=tmp_mappings_index,
        )
    if args.iconclass_val_path:
        iconclass_output_path = os.path.join(args.output_path, "iconclass")
        os.makedirs(iconclass_output_path, exist_ok=True)
        write_iconclass(
            args.iconclass_val_path,
            os.path.join(iconclass_output_path, "val.jsonl"),
            samples=iconclass_samples,
            iconclass_mapping=tmp_mappings_index,
        )
    if args.iconclass_test_path:
        iconclass_output_path = os.path.join(args.output_path, "iconclass")
        os.makedirs(iconclass_output_path, exist_ok=True)
        write_iconclass(
            args.iconclass_test_path,
            os.path.join(iconclass_output_path, "test.jsonl"),
            samples=iconclass_samples,
            iconclass_mapping=tmp_mappings_index,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
