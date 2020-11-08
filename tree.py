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
from functools import reduce
from preprocess import preprocess
from utils import gini, gini_score



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
        self.data, self.feat_map = preprocess(X, y, feat_type)



class CartTreeNode(object):
    """Cart Decision Tree Algorithm.

    Attributes:
        depth (int): this node's depth.
        left, right (CartTreeNode): child node.
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
    
    def fit(self, data, y, feat_map, feat_type):
        """

        Args:
            data (dict): {feat name: {feat value: [idx, ...]}.
            y (np.ndarray).
            feat_map (dict): {feat name: {feat 2 int label}}.
            feat_type (dict): {feat_name: "numerical" or "categorical"}.
        """
        cond0 = self.depth >= self.max_depth
        cond1 = len(y) <= self.min_samples
        count = sum([len(data[x].keys()) for x in data])
        cond2 = count == 1
        if cond0 or cond1 or cond2:
            self.is_leaf = True
            counter = Counter(y)
            self.output = max(counter, key=counter.get)
        else:
            feat_names = list(feat_map.keys())
            scores = {}
            for feat_name in feat_names:
                values = list(data[feat_name].keys())
                if len(values) == 1:
                    # drop this feature
                    del data[feat_name]
                    del feat_map[feat_name]
                    continue
                # calc each feature value's gini score
                if feat_type[feat_name] == "categorical":
                    scores[feat_name] = self.categorical_scores(values, y, data[feat_name])
                else:
                    scores[feat_name] = self.numerical_scores(values, y, data[feat_name],
                                                              feat_map[feat_name])
            # find best feature and value
            best_feat_name = None
            best_v = None
            best_score = float('inf')
            for feat_name in scores:
                for v in scores[feat_name]:
                    s = scores[feat_name][v]
                    if s < best_score:
                        best_score = s
                        best_v = v
                        best_feat_name = feat_name
            
            # update data, store info
        
    def update(self, feat_name, v, s, data, y, feat_map, feat_type):
        """store info, update data and feat_map, split.
        """
        self.feat_name = feat_name
        self.x0 = v
        self.data_type = feat_type[feat_name]

        # update data.
        if self.data_type == "categorical":
            left = data[feat_name][v]
            right = reduce(lambda a, b: a + b,
                           [data[feat_name][i] for i in data[feat_name] if i != v])
            del data[feat_name][v]
            del feat_map[feat_name][v]
            left_y = y[left]
            right_y = y[right]
            left_idx_map = {x: i for i, x in enumerate(left)}
            right_idx_map = {x: i for i, x in enumerate(right)}
            left_data = {
                feat_name: {
                    v: [left_idx_map[idx] for idx in data[feat_name][v]]
                    for v in data[feat_name]
                } for feat_name in data
            }
            right_data = {
                feat_name: {
                    v: [right_idx_map[idx] for idx in data[feat_name][v]]
                    for v in data[feat_name]
                } for feat_name in data
            }
            left_feat_map = feat_map
            del left_feat_map[self.feat_name]
            left_feat_map[self.feat_name] = {
                "int2feat": {self.x0: feat_map[self.feat_name][self.x0]},
                
            }
            right_feat_map = feat_map
            del right
        else:
            split_points = feat_map[feat_name]
            left_x = sorted([x for x in split_points if x < v])
            right_x = sorted([x for x in split_points if x > v])
            choose_idx = len(left_x)
            M = len(split_points)
            left_idx = reduce(lambda a, b: a + b,
                              [data[feat_name][i] for i in range(choose_idx)])
            right_idx = reduce(lambda a, b: a + b, [data[feat_name][i]
                               for i in range(choose_idx, M + 1)])
            left_y = y[left_idx]
            right_y = y[right_idx]
            left_idx_map = {x: i for i, x in enumerate(left_idx)}
            right_idx_map = {x: i for i, x in enumerate(right_idx)}

            # update data
            left_data = {
                feat_name: {
                    v: [left_idx_map[idx] for idx in data[feat_name][v]]
                    for v in data[feat_name]
                } for feat_name in data if feat_name != self.feat_name
            }
            left_data[self.feat_name] = {
                k: [left_idx_map[x] for x in left_data[self.feat_name][k]]
                for k in range(choose_idx)
            }
            right_data = {
                feat_name: {
                    v: [right_idx_map[idx] for idx in data[feat_name][v]]
                    for v in data[feat_name]
                } for feat_name in data if feat_name != self.feat_name
            }
            right_data[self.feat_name] = {
                k - choose_idx: [right_idx_map[x] for x in data[self.feat_name][k]]
                for k in range(choose_idx, M + 1)
            }
            # update feat_map
            left_feat


    def categorical_scores(self, values, y, value2idx):
        """calculate gini for categorical feature.

        Args:
            values (list): [v0, v1, ...].
            y (np.ndarray).
            value2idx (dict): {value: [idx, ...]}
        Returns:
            scores (dict): {value: gini score}
        """
        if len(values) == 2:
            values = values[:1]
        scores = {}
        for v in values:
            other_v = list(filter(lambda x: x != v, value2idx.keys()))
            idx0 = value2idx[v]
            idx1 = reduce(lambda a, b: a + b, [value2idx[i] for i in other_v])
            scores[v] = gini_score(y, idx0, idx1)
        return scores


    def numerical_scores(self, values, y, value2idx, split_points):
        """calc gini score for numerical feature.

        Args:
            split_points (list [float]): [x0, x1, ...], in order.
        Returns:
            scores (dict): {xi: gini}
        """
        assert len(values) == len(split_points) + 1
        scores = {}
        for idx, x in enumerate(split_points):
            v0 = range(0, idx)
            v1 = range(idx, len(split_points) + 1)
            idx0 = reduce(lambda a, b: a + b, [value2idx[i] for i in v0])
            idx1 = reduce(lambda a, b: a + b, [value2idx[i] for i in v1])
            scores[x] = gini_score(y, idx0, idx1)
        return scores
        



        
        
    
