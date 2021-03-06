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
import json
from tqdm import tqdm
from multiprocessing import Process
from sklearn.tree import DecisionTreeClassifier
from tree import CartTree
from utils import bootstrap_sample, balance_sample


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
        balance (bool): whether to balance +1/-1 samples num.
        use_sklearn (bool): test whether cart tree is working.
    """
    def __init__(self, num_tree, max_depth, min_m, min_samples,
                 balance=False, use_sklearn=False):
        self.num_tree = num_tree
        self.max_depth = max_depth
        self.min_m = min_m
        self.min_samples = min_samples
        self.use_sklearn = use_sklearn
        if not use_sklearn:
            self.forests = [CartTree(max_depth, min_samples, min_m) for _ in range(num_tree)]
        else:
            self.forests = [DecisionTreeClassifier()
                            for _ in range(num_tree)]
        self.oob = None
        self.balance = balance

    def fit(self, X, y, feat_types):
        self.X = X.copy()
        self.y = y.copy()
        self.feat_types = feat_types.copy()
        origin_idx = {i: i for i in range(X.shape[1])}
        self.use_record = np.ones((X.shape[0], self.num_tree), dtype=bool)
        self.use_record *= False
        self.N = self.X.shape[0]
        for i in range(self.num_tree):
            if not self.balance:
                sample_idxs = bootstrap_sample(self.N, self.N)
            else:
                sample_idxs = balance_sample(self.y)
            X = self.X[sample_idxs]
            y = self.y[sample_idxs]
            # print(f"sample size: {X.shape}, {len(sample_idxs)}, {len(set(sample_idxs))}")
            self.use_record[sample_idxs, i] = True
            if not self.use_sklearn:
                self.forests[i].fit(X, y, origin_idx, feat_types)
            else:
                self.forests[i].fit(X, y)

        print("complete")

    def get_oob(self):
        if self.oob is None:
            self.oob = -np.ones(self.use_record.shape)
            for i in tqdm(range(self.X.shape[1]), desc="get oob"):
                record = np.logical_not(self.use_record[:, i])
                idxes = np.where(record)[0]
                pred_y = self.forests[i].predict(self.X[idxes, :])
                pred_y = [int(k) for k in pred_y]
                for j, idx in enumerate(idxes):
                    self.oob[idx, i] = pred_y[j]
            oob = []
            for i in range(self.oob.shape[0]):
                pred_y = [int(x) for x in self.oob[i] if x != -1]
                oob.append(pred_y)
            self.oob = oob
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
                  X, y, feat_types, balance, folder,
                  all_data=False):
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
                    args=(X, y, feat_types, os.path.join(folder, f"{i}.rf"), sub_tree_num,
                          max_depth, min_m, min_samples, balance, ))
        p_list.append(p)
        p.start()
    for p in p_list:
        p.join()

    print("All process finish")
    forest_list = []
    for i in range(num_process):
        fn = os.path.join(folder, f"{i}.rf")
        with open(fn, "rb") as f:
            forest_list.append(pickle.load(f))
        os.remove(fn)
    rf = merge_forest(forest_list)
    oob = rf.get_oob()
    print("oob calc done")
    filename = os.path.join(folder, f"{num_tree}-{max_depth}-{min_samples}-{min_m}-all.json")
    oob_filename = os.path.join(folder, f"{num_tree}-{max_depth}-{min_samples}-{min_m}.json")
    if all_data:
        result = rf.predict(X, vote=False)
        result = [[int(x) for x in r] for r in result]
        with open(filename, "w") as f:
            f.write(json.dumps(result))
    with open(oob_filename, "w") as f:
        f.write(json.dumps(oob))
    


def train_one(X, y, feat_types, filename,
                  sub_tree_num, max_depth, min_m, min_samples, balance):
    print(f"pid {os.getpid()} start")
    rf = RandomForest(sub_tree_num, max_depth, min_m, min_samples, balance)
    rf.fit(X, y, feat_types)
    with open(filename, "wb") as f:
        pickle.dump(rf, f)