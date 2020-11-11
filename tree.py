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
from preprocess import preprocess
from utils import (
    gini,
    gini_score,
    filter_data,
    random_feat
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

    def fit(self, X, y, feat_type):
        """train

        Args:
            X (pd.DataFrame): n samples x n_features.
            y (np.ndarray).
            feat_type (dict): {feat_name: numerical or categorical}
        """
        self.data, self.feat_names, self.feat_map = preprocess(X, y, feat_type)


class CartTreeNode(object):
    """Cart Decision Tree Algorithm.

    Attributes:
        depth (int): this node's depth.
        true_t, false_t (CartTreeNode): child node.
        max_depth (int): max depth for tree.
        min_samples (int): if samples < min_samples, regard node as leaf.
        x0 (int or float): condition right value.
        feat_name (str).
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
    
    def fit(self, data, y, feats, feat_map, feat_type):
        """

        Args:
            data (np.ndarray): n samples x n feature.
            y (np.ndarray): labels.
            feats (list): indicate data's column name.
            feat_map (dict): contain split point and object to int label.
            feat_type (dict): feat_name: "num" or "cate".
        Returns:
            None
        """
        if len(set(y)) == 1 or\
           len(y) <= self.min_samples or\
           self.depth >= self.max_depth:
            counter = Counter(y)
            self.output = counter.most_common()[0][0]
            self.is_leaf = True
            return

        # filter data, delete column which has only 1 value
        data, feats, feat_map = filter_data(data, feats, feat_type, feat_map)
        # choose m feat
        choose_feats = random_feat(feats, self.min_m)
        # iter and find best feat and best value
        best_score = float('inf')
        best_feat = None
        best_value = None
        for name, idx in choose_feats.items():
            x = data[:, idx]
            if feat_type[name] == "categorical":
                xs = list(set(x))
                if len(xs) == 2:
                    xs.pop(0)
                for v in xs:
                    true_y = y[x == v]
                    false_y = y[x != v]
                    score = gini_score(true_y, false_y)
                    if score < best_score:
                        best_score = score
                        best_feat = name
                        best_value = v
            else:
                xs = feat_map[name]
                for v in xs:
                    true_y = y[x <= v]
                    false_y = y[x > y]
                    score = gini_score(true_y, false_y)
                    if score < best_score:
                        best_score = score
                        best_feat = name
                        best_value = v
        # update self info
        self.score = best_score
        self.feat_name = best_feat
        self.x0 = best_value
        self.data_type = feat_type[best_feat]
        # split
        self.split(data, y, feats, feat_map, feat_type)

    def split(self, data, y, feats, feat_map, feat_types):
        """split
        """
        idx = feats.index(self.feat_name)
        x = data[:, idx]
        if feat_types[self.feat_name] == "categorical":
            true_idx = x == self.x0
            false_idx = x != self.x0
        else:
            true_idx = x <= self.x0
            false_idx = x > self.x0
            splits = feat_map[self.feat_name]
            feat_map[self.feat_name] = list(filter(lambda x: x != self.x0,
                                                   splits))
        # split data into 2 parts
        true_data = data[true_idx]
        false_data = data[false_idx]
        true_y = y[true_idx]
        false_y = y[false_idx]
        # build new tree
        self.true_t = CartTreeNode(self.depth + 1, self.max_depth,
                                   self.min_samples, self.min_m)
        self.true_t.fit(true_data, true_y, feats, feat_map, feat_types)
        self.false_t = CartTreeNode(self.depth + 1, self.max_depth,
                                    self.min_samples, self.min_m)
        self.false_t.fit(false_data, false_y, feats, feat_map, feat_types)


