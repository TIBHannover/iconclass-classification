#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 21:12:48 2021

@author: javad
"""
import numpy as np
import json
import nltk

# nltk.download()
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from datasets.utils import read_line_data

stop_words_en = set(stopwords.words("english"))
stop_words_gr = set(stopwords.words("german"))
# eng_words = set(nltk.corpus.words.words())


def preprocessing_labels(label_path, output_path):
    """gets the input labels/descriptions of icon class and do the following prprocessing steps for CLIP:
        lowercase, removing digits, removing punctuations,
        removing stop words in english and german, removing duplicates description
    label_path: the path for label.jsonl file
    output_path: the path for new jsonl file with the preprocessed labels
    """
    lb = read_line_data(label_path, dict_key="id")
    lb_list = []
    for k, v in lb.items():
        vv = ", ".join(v["kw"]["en"]) + ", " + v["txt"]["en"]
        vv = word_tokenize(vv)
        vv_l = [w.lower() for w in vv]
        words = [word for word in vv_l if word.isalpha()]
        table = str.maketrans("", "", string.punctuation)
        stripped = [w.translate(table) for w in words]
        words = [word for word in stripped if word.isalpha()]
        words_en = [w for w in words if not w in stop_words_en]
        words_gr = [w for w in words_en if not w in stop_words_gr]
        words_gr = list(set(words_gr))

        lb_list.append({"id": k, "txt": words_gr})


# output_path = "/home/javad/Desktop/Dataset/iconclass/last_version/clean_clip_labels.jsonl"
# append = False
# mode = 'a+' if append else 'w'
# with open(output_path, mode, encoding='utf-8') as f:
#     for line in lb_list:
#         json_record = json.dumps(line, ensure_ascii=False)
#         f.write(json_record + '\n')

# for i in lb_list.values():
#     tokenize(i)
# aa = ["page","chelsea",  "death of St. Agatha"]
# bb = tokenize(["This is " + desc for desc in aa]).numpy()
# print(_tokenizer.decode(bb[0,:]))