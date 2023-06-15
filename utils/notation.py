# !/usr/bin/python
# -*- coding: utf-8 -*-

import re

from copy import deepcopy
from utils import get_bracketed

SCHEME = re.compile(
    r'^(\d{1,2})([A-IK-Z]{1,2})?(\d+)?'
    r'(\([^+)]+\))?(\d+)?(\(\+[0-9]+\))?$'
)


def is_valid(x):
    if isinstance(x, Notation) and x.is_valid():
        return True

    return False


def valid_notation(function):
    def function_wrapper(self, *args, **kwargs):
        if self.is_valid():
            return function(self, *args, **kwargs)

    return function_wrapper


def valid_notations(function):
    def function_wrapper(self, other, *args, **kwargs):
        if self.is_valid() and is_valid(other):
            return function(self, other, *args, **kwargs)

    return function_wrapper


class Notation:
    def __init__(self, code):
        self.code = code.replace('+ ', '+').strip(':,; ')
        text = get_bracketed(self.code, get_key=False)

        if text:
            text_without = text[0].replace(' (', ', ')
            text_without = text_without.replace('(', ', ')
            text_without = text_without.replace(')', '')

            self.code = self.code.replace(text[0], text_without)

        self._match = SCHEME.findall(self.code)

        if self.is_valid():
            self._match = list(self._match[0])

            self.division = int(self.code[0])
            self.depth = self._get_depth()

    def __hash__(self):
        return hash(self.code)

    @valid_notations
    def __eq__(self, other):
        if self.code == other.code:
            return True

        return False

    @valid_notations
    def __ne__(self, other):
        return not (self.code == other.code)

    @valid_notations
    def __lt__(self, other):
        return self.code < other.code

    @valid_notation
    def __len__(self):
        return self.depth

    def __repr__(self):
        return '<{} object with code {}>'.format(
            self.__class__.__name__, self.code
        )

    def is_valid(self):
        if self._match:
            return True

        return False

    @valid_notation
    def add(self, other):
        if isinstance(other, str):
            return Notation(self.code + other)

    @valid_notation
    def replace(self, old_value, new_value=''):
        if isinstance(old_value, Notation):
            old_value = old_value.code

        if isinstance(new_value, Notation):
            new_value = new_value.code

        code = self.code.replace(old_value, new_value, 1)

        return Notation(code)

    @valid_notations
    def is_child(self, other):
        if self.division == other.division:
            for x in self.get_parents_until(other.depth):
                if x == other:
                    return True

        return False

    @valid_notations
    def is_direct_child(self, other):
        if self.get_parent() == other:
            return True

        return False

    @valid_notation
    def is_basic(self):
        if self.get_basic() == self.code:
            return True

        return False

    @valid_notation
    def get_basic(self):
        division = self._match[0]
        letter = self.get_letter()

        if letter:
            return division + letter

        return division

    @valid_notation
    def get_letter(self):
        return self._match[1]

    @valid_notation
    def has_queue(self):
        if self.get_queue():
            return True

        return False

    @valid_notation
    def get_queue(self):
        return self._match[2]

    @valid_notation
    def has_text(self):
        if self.get_text():
            return True

        return False

    @valid_notation
    def get_text(self):
        return self._match[3].strip('()')

    @valid_notation
    def has_name(self):
        if self.has_text() and \
                self.get_text() != '...':
            return True

        return False

    @valid_notation
    def has_digit(self):
        if self.get_digit():
            return True

        return False

    @valid_notation
    def get_digit(self):
        return self._match[4]

    @valid_notation
    def has_key(self):
        if self.get_key():
            return True

        return False

    @valid_notation
    def get_key(self):
        key = self._match[5]

        return key.strip('()').strip('+')

    @valid_notation
    def strip_key(self):
        key = self._match[5]
        code = self.code.replace(key, '')

        return Notation(code)

    @valid_notation
    def get_anti(self):
        if len(self.get_letter()) == 2:
            code = self.code[:3] + self.code[4:]

            return Notation(code)

    @valid_notation
    def get_next(self):
        return self._change(self._add_next, True)

    @valid_notation
    def get_parent(self):
        return self._change(self._strip_last, False)

    @valid_notation
    def get_parents_until(self, depth=1):
        def _get_until(notation):
            if notation.depth > depth:
                notation = notation.get_parent()

                yield from _get_until(notation)
                yield notation

        return list(_get_until(self))

    def _get_depth(self):
        depth = len(self.get_basic()[:2])

        if self.get_letter():
            depth += 1

        if self.has_queue():
            depth += len(self.get_queue())

        if self.has_text():
            if self.get_text() != '...':
                depth += 1

            depth += 1

        if self.has_digit():
            depth += len(self.get_digit())

        if self.has_key():
            depth += len(self.get_key())

        return depth

    def _change(self, _function, is_none=True):
        index = self._get_last_index()

        if index == 5:
            value = _function(self.get_key())
        elif index == 4:
            value = _function(self.get_digit())
        elif index == 3:
            value = '(...)' if self.has_name() else ''
        elif index == 2:
            value = _function(self.get_queue())
        elif index == 1:
            value = _function(self.get_letter())
        else:
            value = _function(self.get_basic()[:2])

        if not (index == 3 and is_none):
            return self._join_code(value, index, is_none)

    def _get_last_index(self):
        for i, text in enumerate(reversed(self._match)):
            if text:
                return len(self._match) - 1 - i

    def _join_code(self, value, i, is_none):
        if not is_none or value:
            code = deepcopy(self._match)
            code[i] = value

            if code[5] and '+' not in code[5]:
                code[5] = '(+{})'.format(code[5])

            code = ''.join(str(x) for x in code if x)

            return Notation(code)

    @staticmethod
    def _add_next(value):
        if value.isdigit():
            value = int(value) + 1

            if value % 10 != 0:
                return str(value)
        else:
            values = list()

            for x in value:
                if x != 'I':
                    x = ord(x) + 1
                else:
                    x = ord(x) + 2

                values.append(chr(x))

            return ''.join(values)

    @staticmethod
    def _strip_last(value):
        if value.isdigit():
            return value[:-1]
