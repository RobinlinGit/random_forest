#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   tree.py
@Time    :   2020/11/02 11:34:55
@Author  :   lzh
@Version :   1.0
@Contact :   robinlin163@163.com
@Desc    :   CART tree
'''
import numpy as np


class Tree(object):

    def __init__(self):
        self.left = None
        self.right = None
        self.cond = None

    def split(self):
        pass

    def fit(self, X, y):
        pass
    
