#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   preprocess.py
@Time    :   2020/10/31 21:58:13
@Author  :   lzh
@Version :   1.0
@Contact :   robinlin163@163.com
@Desc    :   preprocess for bank data.
'''
# %%
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm

from utils import log


# %%
def categery_preprocess(column):
    """map the feature data to [-1, 0, ..., M-1] int categories.
    Only for categorical data.

    Args:
        column (np.ndarray): feature column [n_samples, 1].
        undefine (object): missing value expression in data.
    Returns:
        data (np.ndarray): [n samples, 1], -1 means undefine,
            0 to M-1 means valid data.
        int2feature (dict): {int: origin_feature_name}.
        feature2int (dict): {origin feature name: int}
    """
    feature2int = {}
    labels = set(column)
    feature2int.update({label: i for i, label in enumerate(labels)})
    data = np.vectorize(feature2int.__getitem__)(column)
    int2feature = {v: k for k, v in feature2int.items()}
    return data, int2feature, feature2int


def numerical_preprocess(column, y):
    """generate split point and map undefine to -1.

    Args:
        column (np.ndarray): numerical column [n_samples, 1].
        y (np.ndarray): int labels.
    Returns:
        data (np.ndarray): [n samples, 1], map float x to int label, eg:
            x = 1.0, 2th interval is (0, 1.2], so we map x to 2,
            undefine value will map to -1
        spilt_points (list [float]): split points for column.
    """
    data = np.zeros(column.shape, dtype=np.uint8)
    origin_data = column
    y2 = y

    # sort x, find potential split point

    sort_idx = np.argsort(column)
    column = column[sort_idx]
    y2 = y2[sort_idx]
    diff_x = np.diff(column)
    idxes = np.where(diff_x != 0)[0] + 1
    idxes = [0] + idxes.tolist() + [None]

    # sort y array with the same x value
    for i in range(len(idxes)-1):
        x = column[idxes[i]: idxes[i+1]]
        ys = y2[idxes[i]: idxes[i+1]]
        sort_idx = np.argsort(ys)
        column[idxes[i]: idxes[i+1]] = x[sort_idx]
        y2[idxes[i]: idxes[i+1]] = ys[sort_idx]

    # find split points
    diff_y = np.diff(y2)
    split_idx = np.where(diff_y != 0)[0] + 1
    split_idx = list(filter(lambda x: x in idxes, split_idx))

    split_x = [(column[i] + column[i-1]) / 2 for i in split_idx]
    split_x = sorted(split_x)
    iter_x = [-np.inf] + split_x
    # map x to int label
    for i in range(len(iter_x) - 1):
        select_idx = np.logical_and(origin_data > iter_x[i], origin_data <= iter_x[i + 1])
        data[select_idx] = i
    data[origin_data > iter_x[-1]] = i + 1
    # print(f"data {data}")
    # print(set(data))
    # print(f"split x {split_x}")
    return data, split_x
        

def build_inverted_index(data):
    """build a value: [idx0, idx1] inverted table.

    Args:
        data (list or np.ndarray).
    Returns:
        table (dict).
    """
    values = set(data)
    print(len(values))
    table = {v: [] for v in values}
    for i, v in enumerate(data):
        table[v].append(i)
    return table


@log
def preprocess(df, y, feature_types):
    """preprocess data into a reverted index dict for efficient split.

    Args:
        df (pandas.DataFrame): n samples x n_features.
        y (np.array): labels, n_samples x 1.
        feature_type (dict): key is feature name,
            value is in ["numerical", "categorical"].
    Returns:
        data (dict): {feat_name: {feat_value: [idx]}}.
        label_map (dict): {feat_name: [x0, x1, ...]} for numerical,
            {feat_name: {"int2feat": v, "feat2int": v}} for categorical.
    """
    for _, v in feature_types.items():
        assert v in ["numerical", "categorical"]
    data = {}
    label_map = {}
    for feat_name, feat_type in feature_types.items():
        column = df[feat_name].values
        # numerical process
        if feat_type == "numerical":
            column, split_x = numerical_preprocess(column, y)
            label_map[feat_name] = split_x
        # categorical process
        else:
            column, int2feat, feat2int = categery_preprocess(column)
            label_map[feat_name] = {"int2feat": int2feat, "feat2int": feat2int}

        data[feat_name] = build_inverted_index(column)
    return data, label_map
