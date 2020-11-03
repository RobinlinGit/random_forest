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
df = pd.read_csv("bank-additional-full.csv")
print(df.dtypes)
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
    """generate split point and map undefine to inf.

    Args:
        column (np.ndarray): numerical column [n_samples, 1].
        undefine (object): missing value expression in data.
        y (np.ndarray): labels.
    Returns:
        data (np.ndarray): [n samples, 1], only map undefine to inf.
        spilt_points (list [float]): split points for column.
    """
    column[column == undefine] = np.inf
    y2 = y[column != np.inf]
    data = column
    column = column[column != np.inf]
    sort_idx = np.argsort(column)
    y2 = y2[sort_idx]
    split_idx = np.where(np.diff(y2) != 0)[0]
    splits = (column[split_idx + 1] + column[split_idx]) / 2
    split_points = list(set(splits))
    return data, split_points





        
# %%
y = df['y'].values
y, i2y, y2i = categery_preprocess(y, 0)
print(y)
# %%
column = df['emp.var.rate'].values
print(set(column))


# %%
print(numerical_preprocess(column, 999, y))
# %%
