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


def find_split_x(column, y):
    """generate split point.

    Args:
        column (np.ndarray): numerical column [n_samples, 1].
        y (np.ndarray): int labels.
    Returns:
        spilt_points (list [float]): split points for column.
    """

    # sort x, find potential split point
    sort_idx = np.argsort(column)
    column = column[sort_idx]
    y2 = y[sort_idx]
    diff_x = np.diff(column)
    idxes = np.where(diff_x != 0)[0] + 1
    idxes = [0] + idxes.tolist() + [None]

    last_y = None
    split_x = []
    # find split x
    for i in range(len(idxes)-1):
        ys = y2[idxes[i]: idxes[i+1]]
        if len(set(ys)) == 1:
            if last_y is not None and last_y != ys[0]:
                x0 = column[idxes[i]]
                x1 = column[idxes[i - 1]]
                split_x.append((x0 + x1) / 2)
            last_y = ys[0]
        else:
            last_y = None
            if i >= 1:
                split_x.append((column[idxes[i]] + column[idxes[i-1]]) / 2)
            if i+1 < len(idxes) - 1:
                split_x.append((column[idxes[i]] + column[idxes[i+1]]) / 2)
    split_x = sorted(list(set(split_x)))
    return split_x


@log
def preprocess(df, y, feature_types):
    """preprocess data, map categorical object to int.

    Args:
        df (pandas.DataFrame): n samples x n_features.
        y (np.array): labels, n_samples x 1.
        feature_type (dict): key is feature name,
            value is in ["numerical", "categorical"].
    Returns:
        data (numpy.ndarray): n samples x n feature.
        feat_names (list (str)): for index.
        label_map (dict): {feat_name: [x0, x1, ...]} for numerical,
            {feat_name: {"int2feat": v, "feat2int": v}} for categorical.
    """
    for _, v in feature_types.items():
        assert v in ["numerical", "categorical"]
    label_map = {}
    for feat_name, feat_type in feature_types.items():
        column = df[feat_name].values
        # numerical process
        if feat_type == "numerical":
            split_x = find_split_x(column, y)
            label_map[feat_name] = split_x
        # categorical process
        else:
            column, int2feat, feat2int = categery_preprocess(column)
            label_map[feat_name] = {"int2feat": int2feat, "feat2int": feat2int}
            df[feat_name] = column
    data = df.values
    feat_names = list(df.columns)
    return data, feat_names, label_map
