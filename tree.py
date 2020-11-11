#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   tree.py
@Time    :   2020/11/02 11:34:55
@Author  :   lzh
@Version :   1.0
@Contact :   robinlin163@163.com
@Desc    :   CART tree implement
'''
import numpy as np
import math
import random
from collections import Counter
from functools import reduce

from numpy.lib.shape_base import split
from preprocess import preprocess
from utils import (
    gini,
    gini_score,
    filter_data,
    random_feat,
    log
)



def tree_condition(x, x0, operator):
    """tree node condition function

    Args:
        x (int or float): data.
        x0 (int or float): condition right value.
        operator (str): support "==", "<=".
    """
    assert operator in ["==", "<="]
    if operator == "==":
        return x == x0
    else:
        return x <= x0


class CartTree(object):
    """Cart Decision Tree Warpper.

    Attributes:
        root (CartTreeNode): tree.
        min_m (int): min select feat num.
        max_depth (int): 
    """
    def __init__(self, max_depth, min_samples, min_m=2):
        self.root = CartTreeNode(0, max_depth, min_samples, min_m)
        self.min_m = min_m
        self.max_depth = max_depth
        self.min_samples = min_samples

    @log
    def fit(self, X, y, feat_type):
        """train

        Args:
            X (pd.DataFrame): n samples x n_features.
            y (np.ndarray).
            feat_type (dict): {feat_name: numerical or categorical}
        """
        data, self.feat_names, self.feat_map = preprocess(X, y, feat_type)
        origin_idx = {i: i for i in range(len(self.feat_names))}
        split_x = {
            i: self.feat_map[self.feat_names[i]]
            for i in range(len(self.feat_names))
            if feat_type[self.feat_names[i]] == "numerical"
        }
        feat_types = {i: feat_type[self.feat_names[i]] for i in origin_idx}
        print(feat_types)
        self.root.fit(data, y, split_x, feat_types, origin_idx)
        print("done")

class CartTreeNode(object):
    """Cart Decision Tree Algorithm.

    Attributes:
        depth (int): this node's depth.
        true_t, false_t (CartTreeNode): child node.
        max_depth (int): max depth for tree.
        min_samples (int): if samples < min_samples, regard node as leaf.
        x0 (int or float): condition right value.
        feat_name (int).
        data_type (str): "numerical" or "categorical".
        min_m (int): select m features when split.
        is_leaf (bool).
        output (int): if is leaf, output is label, else None.
    """
    def __init__(self, depth, max_depth, min_samples, min_m):
        self.max_depth = max_depth
        self.depth = depth
        self.min_samples = min_samples
        self.min_m = min_m
        self.data_type = None
        self.x0 = None
        self.feat_name = None
        self.is_leaf = False
    
    def fit(self, data, y, split_x, feat_type, origin_idx):
        """

        Args:
            data (np.ndarray): n samples x n feature.
            y (np.ndarray): labels.
            splix_x (dict): {col_idx: split points}
            feat_type (list): feat_name: "num" or "cate".
            origin_idx (dict): {idx: origin idx in origin feature}
        Returns:
            None
        """
        if len(set(y)) == 1 or\
           len(y) <= self.min_samples or\
           self.depth >= self.max_depth:
            counter = Counter(y)
            try:
                self.output = counter.most_common()[0][0]
                self.is_leaf = True
            except IndexError:
                print(y)
            return

        # filter data, delete column which has only 1 value
        data, feat_map, feat_type, origin_idx = filter_data(
            data, feat_type, split_x, origin_idx
        )
        # choose m feat
        feat_idx = random_feat(data.shape[1], self.min_m)
        # iter and find best feat and best value
        best_score = float('inf')
        best_feat = None
        best_value = None
        for idx in feat_idx:
            x = data[:, idx]
            if feat_type[idx] == "categorical":
                xs = list(set(x))
                if len(xs) == 2:
                    xs.pop(0)
                for v in xs:
                    true_idx = x == v
                    true_y = y[true_idx]
                    false_y = y[np.logical_not(true_idx)]
                    score = gini_score(true_y, false_y)
                    if score < best_score:
                        best_score = score
                        best_feat = idx
                        best_value = v
            else:
                xs = feat_map[idx]
                for v in xs:
                    true_idx = x <= v
                    true_y = y[true_idx]
                    false_y = y[np.logical_not(true_idx)]
                    score = gini_score(true_y, false_y)
                    if score < best_score:
                        best_score = score
                        best_feat = idx
                        best_value = v
        # update self info
        self.score = best_score
        self.feat_col = origin_idx[best_feat]
        self.x0 = best_value
        self.data_type = feat_type[best_feat]
        # split
        self.split(data, y, feat_map, feat_type, best_feat, origin_idx)

    def split(self, data, y, split_x, feat_types, col, origin_idx):
        """split
        """

        x = data[:, col]
        true_split = split_x.copy()
        false_split = split_x.copy()
        if feat_types[col] == "categorical":
            true_idx = x == self.x0
        else:
            true_idx = x <= self.x0
            x0idx = split_x[col].index(self.x0)
            true_split[col] = split_x[col][: x0idx]
            false_split[col] = split_x[col][x0idx + 1: ]
        false_idx = np.logical_not(true_idx)

        # split data into 2 parts
        true_data = data[true_idx]
        false_data = data[false_idx]
        true_y = y[true_idx]
        false_y = y[false_idx]
        

        # build new tree
        self.true_t = CartTreeNode(self.depth + 1, self.max_depth,
                                   self.min_samples, self.min_m)
        self.true_t.fit(true_data, true_y, true_split, feat_types, origin_idx)
        self.false_t = CartTreeNode(self.depth + 1, self.max_depth,
                                    self.min_samples, self.min_m)
        self.false_t.fit(false_data, false_y, false_split, feat_types, origin_idx)


