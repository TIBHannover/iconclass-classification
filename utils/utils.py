# !/usr/bin/python
# -*- coding: utf-8 -*-

import re
import ujson
import os.path

from collections import defaultdict

SEPARATOR = ' | '


def get_ext(file_path):
    return os.path.splitext(file_path)[1]


def is_in(terms_1, terms_2):
    if not isinstance(terms_1, str) and not \
            isinstance(terms_2, str):
        if set(terms_1).intersection(terms_2):
            return True
    elif not isinstance(terms_1, str) and \
            isinstance(terms_2, str):
        if any(x in terms_2 for x in terms_1):
            return True
    else:
        return terms_1 in terms_2

    return False


def dump_json(document, line_break=True, indent=0):
    document = ujson.dumps(
        document, indent=indent, ensure_ascii=False
    )

    return document + '\n' if line_break else document


def get_reversed(data, join=True):
    reversed_data = defaultdict(list)

    for key, values in data.items():
        if join and not isinstance(values, str):
            values = [SEPARATOR.join(sorted(values))]

        for value in values:
            reversed_data[value].append(key)

    if all(len(x) == 1 for x in reversed_data.values()):
        reversed_data = {
            k: v[0] for k, v in reversed_data.items()
        }

    return reversed_data


def get_bracketed(document, get_key=False):
    def _get_content():
        stack = list()

        for i, char in enumerate(document):
            if char == '(':
                stack.append(i)
            elif char == ')' and stack:
                start = stack.pop()

                if len(stack) == 0:
                    content = document[start + 1: i]

                    if get_key and '+' in content:
                        yield content

                    if not (get_key or '+' in content):
                        yield content

    # if isinstance(document, Notation):
    #     document = document.code

    return list(_get_content())


def strip_bracketed(document):
    bracketed = get_bracketed(document, get_key=True) + \
                get_bracketed(document, get_key=False)

    for x in bracketed:
        document = document.replace('({})'.format(x), '')

    return re.sub(r'\s\s+', ' ', document).strip()
