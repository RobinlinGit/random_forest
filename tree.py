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
from collections import Counter
from multiprocessing import Queue
from preprocess import find_split_x
from utils import (
    gini_score,
    filter_data,
    random_feat,
    log,
    get_most,
    CATEGORICAL, NUMERICAL
)


@log
def train_tree(i, q: Queue, tree_args, X, y, origin_idx, feat_types):
    print(f"{i} th train start")
    t = CartTree(*tree_args)
    t.fit(X, y, origin_idx, feat_types)
    q.put((i, t))
    print(f"{i} th tree complete")


def tree_condition(x, x0, feat_type):
    """tree node condition function

    Args:
        x (int or float): data. 
        x0 (int or float): condition right value.
        type (str): support "categorical", "numerical".
    """
    assert feat_type in [CATEGORICAL, NUMERICAL]
    return x == x0 if feat_type == CATEGORICAL else x <= x0


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
            X (np.ndarray): n samples x n_features.
            y (np.ndarray).
            feat_type (dict): {idx: numerical or categorXical}
        """
        origin_idx = {i: i for i in range(X.shape[1])}
        split_x = {idx: find_split_x(X[:, idx], y) for idx in range(X.shape[1])
                   if feat_type[idx] == NUMERICAL}
        self.root.fit(X, y, split_x, feat_type, origin_idx)

    def predict(self, x):
        """
        Args:
            x (np.ndarray): feature, size (n_samples, n_features).
        Returns:
            label (list): [n samples].
        """
        assert len(x.shape) == 2
        labels = []
        for i in range(x.shape[0]):
            feat = x[i]
            t = self.root
            while not t.is_leaf:
                t = t.true_t if tree_condition(
                    feat[t.feat_col], t.x0, t.data_type) else t.false_t
            labels.append(t.output)
        return labels


class CartTreeNode(object):
    """Cart Decision Tree Algorithm.

    Attributes:
        depth (int): this node's depth.
        true_t, false_t (CartTreeNode): child node.
        max_depth (int): max depth for tree.
        min_samples (int): min sample num for splitting.
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
        split_x = split_x.copy()
        feat_type = feat_type.copy()
        origin_idx = origin_idx.copy()

        # test if reached stop condition
        if len(set(y)) == 1 or\
           len(y) <= self.min_samples or\
           self.depth >= self.max_depth:
            self.output = get_most(y)
            self.is_leaf = True
            return

        # filter data, delete column which has only 1 value
        data, feat_map, feat_type, origin_idx = filter_data(
            data, feat_type, split_x, origin_idx
        )
        if data.shape[1] == 0:
            self.output = get_most(y)
            self.is_leaf = True
            return

        # choose m feat
        feat_idx = random_feat(data.shape[1], self.min_m)
        # iter and find best feat and best value
        best_score = float('inf')
        best_feat = None
        best_value = None
        best_true_idx = None
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
                        best_true_idx = true_idx
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
                        best_true_idx = true_idx
        # update self info
        self.score = best_score
        self.feat_col = origin_idx[best_feat]
        self.x0 = best_value
        self.data_type = feat_type[best_feat]
        # split
        self.split(data, y, best_true_idx, feat_map,
                   feat_type, best_feat, origin_idx)

    def split(self, data, y, true_idx, split_x, feat_types, col, origin_idx):
        """split
        """
        true_split = split_x.copy()
        false_split = split_x.copy()
        if feat_types[col] == NUMERICAL:
            x0idx = split_x[col].index(self.x0)
            true_split[col] = split_x[col][: x0idx]
            false_split[col] = split_x[col][x0idx + 1:]
        false_idx = np.logical_not(true_idx)
        # split data into 2 parts
        true_data = data[true_idx]
        false_data = data[false_idx]
        true_y = y[true_idx]
        false_y = y[false_idx]
        if len(false_y) == 0 or len(true_y) == 0:
            print(self.data_type)
            print(self.feat_col)
            print(self.x0)
            print(split_x[col])
    
        # build new tree
        self.true_t = CartTreeNode(self.depth + 1, self.max_depth,
                                   self.min_samples, self.min_m)
        self.true_t.fit(true_data, true_y, true_split, feat_types, origin_idx)
        self.false_t = CartTreeNode(self.depth + 1, self.max_depth,
                                    self.min_samples, self.min_m)
        self.false_t.fit(false_data, false_y, false_split,
                         feat_types, origin_idx)
