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


def bootstrap_sample(L, size):
    a = np.arange(L)
    return np.random.choice(a, size=size)


def balance_sample(y):
    idxes = []
    counter = Counter(y)
    N = counter.most_common()[-1][1]
    for c in counter:
        c_idx = np.where(y == c)[0]
        L = counter[c]
        c_idx = c_idx[bootstrap_sample(L, N)].tolist()
        idxes += c_idx
    return idxes


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
    assert len(y0) > 0 and len(y1) > 0
    N = len(y0) + len(y1)
    score = len(y0) / N * gini(y0) + len(y1) / N * gini(y1)
    return score


def filter_data(data, feat_types, split_x, origin_idx):
    """delete column which has only one value.
    """
    rm_list = []
    for idx in range(data.shape[1]):
        if feat_types[idx] == "categorical" and \
           len(set(data[:, idx])) <= 1:
            rm_list.append(idx)
        elif feat_types[idx] == "numerical":
            if idx not in split_x:
                rm_list.append(idx)
                continue
            for x in split_x[idx].copy():
                cond = data[:, idx] <= x
                cond1 = np.logical_not(cond)
                if np.all(cond) or np.all(cond1):
                    # print(f"remove {x}, {idx}")
                    # print(set(data[:, idx]))
                    split_x[idx].remove(x)
            if len(split_x[idx]) == 0:
                rm_list.append(idx)

    keep_idx = list(filter(lambda x: x not in rm_list, range(data.shape[1])))
    keep_idx2idx = {k: i for i, k in enumerate(keep_idx)}
    data = data[:, keep_idx]
    feat_map = {keep_idx2idx[n]: split_x[n]
                for n in keep_idx if n in split_x}
    feat_types = [feat_types[n] for n in keep_idx]
    origin_idx = {keep_idx2idx[n]: origin_idx[n] for n in keep_idx}
    return data, feat_map, feat_types, origin_idx


def random_feat(len_feat, min_m):
    """random choose m features.

    Args:
        len_feat (int).
        min_m (int).
    Returns:
        feat_idx (list): [idx]]
    """
    m = max([math.ceil(math.sqrt(len_feat)), min_m])
    m = min(m, len_feat)
    choose_idx = list(range(len_feat))
    random.shuffle(choose_idx)
    feat_idx = choose_idx[: m]
    return feat_idx


def log(func):
    def wrapper(*args, **kw): 
        local_time = time()  
        result = func(*args, **kw)
        print(f"function {func.__name__} cost time {time() - local_time} s")
        return result
    return wrapper