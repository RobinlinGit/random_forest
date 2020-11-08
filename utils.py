#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2020/11/08 14:02:29
@Author  :   lzh 
@Version :   1.0
@Desc    :   utils functions
'''
import numpy as np
from time import time
from collections import Counter


def gini(x):
    """gini score g = 1 - \sum_k p_k^2

    Args:
        x (list or np.ndarray): label list.
    Returns:
        gini score (float).
    """
    count = Counter(x)
    n = len(x)
    ps = [(x / n) ** 2 for k, x in count.items()]
    gini_score = 1 - np.sum(ps)
    return gini_score


def gini_score(y, idx0, idx1):
    """calc weight gini.
    
    Args:
        y (np.ndarray).
        idx0 (list [int]).
        idx1 (list [int]).
    Returns:
        weighted gini score (float).
    """
    N = len(y)
    assert len(idx0) + len(idx1) == N
    score = len(idx0) / N * gini(y[idx0]) + len(idx1) / N * gini(y[idx1])
    return score



def log(func):  
    def wrapper(*args, **kw):  
        local_time = time()  
        result = func(*args, **kw)
        print(f"function {func.__name__} cost time {time() - local_time} s")
        return result
    return wrapper