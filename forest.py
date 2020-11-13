#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   forest.py
@Time    :   2020/11/12 20:37:24
@Author  :   lzh
@Version :   1.0
@Contact :   robinlin163@163.com
@Desc    :   Random Forest
'''
import os
import pickle
import numpy as np
from multiprocessing import Queue, Process
from tree import CartTree
from utils import bootstrap_sample


class RandomForest(object):
    """Random Forest

    Attributes:
        num_tree (int).
        min_m (int).
        min_samples (int).
        max_depth (int).
        process_num (int): max num of multi process.
        forests (list [CartTree]).
        use_record (np.ndarray): size [num_samples, num_tree].
        X (np.ndarray): data.
        y (np.ndarray): labels.

    """
    def __init__(self, num_tree, max_depth, min_m, min_samples):
        self.num_tree = num_tree
        self.max_depth = max_depth
        self.min_m = min_m
        self.min_samples = min_samples
        self.forests = [CartTree(max_depth, min_samples, min_m) for _ in range(num_tree)]
        self.oob = None

    def fit(self, X, y, feat_types):
        self.X = X.copy()
        self.y = y.copy()
        self.feat_types = feat_types.copy()
        origin_idx = {i: i for i in range(X.shape[1])}
        self.use_record = np.ones((X.shape[0], self.num_tree), dtype=bool)
        self.use_record *= False
        for i in range(self.num_tree):
            sample_idxs = bootstrap_sample(X.shape[0], X.shape[0])
            X = self.X[sample_idxs]
            y = self.y[sample_idxs]
            self.use_record[sample_idxs, i] = True
            self.forests[i].fit(X, y, origin_idx, feat_types)

        print("complete")

    def get_oob(self):
        if self.oob is None:
            error_num = 0
            totoal_num = 0
            for i, t in enumerate(self.forests):
                record = self.use_record[:, i].flatten()
                # print(record.shape)
                X = self.X[record]
                y = self.y[record]
                pred_y = np.array(self.predict(X))
                totoal_num += len(y)
                error_num += np.sum(y != pred_y)
            self.oob = error_num / totoal_num
        return self.oob

    def predict(self, X, vote=True):
        """
        Args:
            X (np.ndarray): size (num_samples, num_feature)
        """
        result = [t.predict(X) for t in self.forests]
        result = np.array(result).transpose()
        if vote:
            y = [np.argmax(np.bincount(result[i].flatten()))
                 for i in range(result.shape[0])]
            return y
        else:
            return result


def merge_forest(forest_list):
    """merge forest to get a larger forest.

    Args:
        forest_list (list [RandomForest]).
    Returns:
        RandomForest.
    """
    rf = forest_list[0]
    for f in forest_list[1:]:
        assert np.all(f.X == rf.X)
        assert np.all(f.y == rf.y)
        rf.use_record = np.hstack((rf.use_record, f.use_record))
        rf.num_tree += f.num_tree
        rf.forests += f.forests
    return rf


def train_forests(num_tree, num_process, max_depth, min_samples, min_m,
                  X, y, feat_types, folder, filename):
    """train forest parallel.

    Args:
        num_process (int).
    Returns:
        rf (RandomForest).
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    p_list = []
    sub_tree_num = int(num_tree / num_process)
    for i in range(num_process):
        p = Process(target=train_one,
                    args=(X, y, feat_types, os.path.join(folder, str(i)), sub_tree_num,
                          max_depth, min_m, min_samples,))
        p_list.append(p)
        p.start()
    for p in p_list:
        p.join()

    print("All process finish")
    forest_list = []
    for i in range(num_process):
        fn = os.path.join(folder, str(i))
        with open(fn, "rb") as f:
            forest_list.append(pickle.load(f))
        os.remove(fn)
    rf = merge_forest(forest_list)
    
    with open(os.path.join(folder, filename), "wb") as f:
        pickle.dump(rf, f)


def train_one(X, y, feat_types, filename,
                  sub_tree_num, max_depth, min_m, min_samples):
    print(f"pid {os.getpid()} start")
    rf = RandomForest(sub_tree_num, max_depth, min_m, min_samples)
    rf.fit(X, y, feat_types)
    with open(filename, "wb") as f:
        pickle.dump(rf, f)