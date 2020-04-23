#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File: corpus-label-classification -> word2vec_static
@IDE    : PyCharm
@Author : fengchengli
@Date   : 2020/4/2 20:46
=================================================='''
from gensim.models import Word2Vec
import pandas as pd
from jieba import analyse
import jieba
import os
cur_dir = os.path.dirname(__file__)
stop_dict_file = os.path.join(cur_dir, "/data/stopwords.dat")
analyse.set_stop_words(stop_dict_file)

data_path = os.path.join(cur_dir, '/data/')
output_path = os.path.join(cur_dir, '/model/word2vec/word2vec.model')


def get_all_data():
    common_texts = []
    for path, dirs, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(path, file)
            if file_path[-3:] == 'txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    texts = list(map(lambda x: list(jieba.cut(x.strip())), f.readlines()))
                    common_texts.extend(texts)
    return common_texts


def train_word2vec():
    common_texts = get_all_data()
    model = Word2Vec(common_texts, size=100, window=5, min_count=0, workers=12)
    model.save(output_path)


if __name__ == '__main__':
    train_word2vec()
