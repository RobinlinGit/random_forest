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
from numpy.core.fromnumeric import sort
import pandas as pd
from time import time


# %%

def categery_preprocess(column, undefine):
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
    feature2int = {undefine: -1}
    labels = set(column)
    if undefine in labels:
        labels.remove(undefine)
    feature2int.update({label: i for i, label in enumerate(labels)})
    data = np.vectorize(feature2int.__getitem__)(column)
    int2feature = {v: k for k, v in feature2int.items()}
    return data, int2feature, feature2int


def numerical_preprocess(column, undefine, y):
    """generate split point and map undefine to -1.

    Args:
        column (np.ndarray): numerical column [n_samples, 1].
        undefine (object): missing value expression in data.
        y (np.ndarray): labels.
    Returns:
        data (np.ndarray): [n samples, 1], map float x to int label, eg:
            x = 1.0, 2th interval is (0, 1.2], so we map x to 2,
            undefine value will map to -1
        spilt_points (list [float]): split points for column.
    """
    data = np.zeros(column.shape, dtype=np.uint8)
    origin_data = column
    undefine_idx = np.where(column == undefine)[0]
    y2 = y[column != undefine]
    column = column[column != undefine]

    # map y to int label
    y_values = set(y2)
    ylabel2int = {l: i for i, l in enumerate(y_values)}
    y2 = np.vectorize(ylabel2int.__getitem__)(y2)

    # sort x, find potential split point

    sort_idx = np.argsort(column)
    column = column[sort_idx]
    y2 = y2[sort_idx]
    diff_x = np.diff(column)
    idxes = np.where(diff_x != 0)[0] + 1
    idxes = [0] + idxes.tolist() + [-2]

    # sort y array with the same x value
    for i in range(len(idxes)-1):
        x = column[idxes[i]: idxes[i+1] + 1]
        ys = y2[idxes[i]: idxes[i+1] + 1]
        sort_idx = np.argsort(ys)
        column[idxes[i]: idxes[i+1] + 1] = x[sort_idx]
        y2[idxes[i]: idxes[i+1] + 1] = ys[sort_idx]

    # find split points
    diff_y = np.diff(y2)
    split_idx = np.where(diff_y != 0)[0] + 1
    split_idx = list(filter(lambda x: x in idxes, split_idx))

    split_x = [(column[i] + column[i-1]) / 2 for i in split_idx]
    split_x = sorted(split_x)
    iter_x = [-np.inf] + split_x
    print(iter_x)
    # map x to int label
    for i in range(len(iter_x) - 1):
        select_idx = np.logical_and(origin_data > iter_x[i], origin_data <= iter_x[i + 1])
        data[select_idx] = i
        print(i, np.sum(select_idx))
    print(iter_x[-1])
    data[origin_data > iter_x[-1]] = i + 1
    data[undefine_idx] = -1
    return data, split_x
        

# %%
df = pd.read_csv("bank-additional-full.csv")
print(df.dtypes)

# %%
column = df['emp.var.rate'].values
print(set(column))
y = df['y'].values

# %%
x, split_x = numerical_preprocess(column, 999, y)
print(x, split_x)
# %%

# %%
