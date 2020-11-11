#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2020/11/08 14:02:29
@Author  :   lzh 
@Version :   1.0
@Desc    :   utils functions
'''
import numpy as np
import random
import math
from time import time
from collections import Counter


def gini(x):
    """gini score g = 1 - \sum_k p_k^2

    Args:
        x (list or np.ndarray): label list.
    Returns:
        gini score (float).
    """
    count = Counter(x)
    n = len(x)
    ps = [(x / n) ** 2 for k, x in count.items()]
    gini_score = 1 - np.sum(ps)
    return gini_score


def gini_score(y0, y1):
    """calc weight gini.

    Args:
        y (np.ndarray).
        idx0 (list [int]).
        idx1 (list [int]).
    Returns:
        weighted gini score (float).
    """
    N = len(y0) + len(y1)
    score = len(y0) / N * gini(y0) + len(y1) / N * gini(y1)
    return score


def filter_data(data, feats, feat_types, feat_map):
    """delete column which has only one value.
    """
    rm_list = []
    for idx, name in enumerate(feats):
        if feat_types[name] == "categorical" and \
           len(set(data[:, idx])) <= 1:
            rm_list.append(idx)
        elif feat_types[name] == "numerical" and len(feat_map[name]) == 0:
            rm_list.append(idx)

    keep_idx = list(filter(lambda x: x not in rm_list, range(len(feats))))
    feats = [feats[i] for i in keep_idx]
    data = data[:, keep_idx]
    feat_map = {n: feat_map[n] for n in feats}
    return data, feats, feat_map


def random_feat(feats, min_m):
    """random choose m features.

    Args:
        feats (list [str]).
        min_m (int).
    Returns:
        feat2idx (dict): {feat: idx}
    """
    feat2idx = {name: i for i, name in enumerate(feats)}
    m = math.ceil(math.sqrt(len(feats)))
    if m < min_m:
        m = min_m
    choose_idx = list(range(len(feats)))
    random.shuffle(choose_idx)
    choose_idx = choose_idx[: m]
    feat2idx = {feats[i]: feat2idx[feats[i]] for i in choose_idx}
    return feat2idx


def log(func):
    def wrapper(*args, **kw): 
        local_time = time()  
        result = func(*args, **kw)
        print(f"function {func.__name__} cost time {time() - local_time} s")
        return result
    return wrapper