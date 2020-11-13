#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data_experiment.py
@Time    :   2020/11/13 19:38:58
@Author  :   lzh
@Version :   1.0
@Contact :   robinlin163@163.com
@Desc    :   data's preprocess, including miss value fix
'''

# %%
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from preprocess import preprocess, format_data


# %%
df = pd.read_csv("./bank-additional-full.csv")
y = df['y'].values
y = np.vectorize({"yes": 1, "no": 0}.__getitem__)(y)
df = df.drop(columns=["y", "duration"])

# %%
for col in df.columns:
    if df[col].dtype == object and col != "default":
        counter = Counter(df[col])
        v = counter.most_common()[0][0]
        df[col] = df[col].replace("unknown", v)
    elif col not in  ["default", "y"]:
        m = df[col].mean()
        df[col] = df[col].replace(999, m)
        
# %%
feat_types = {}
for col in df:
    feat_types[col] = "categorical" if df[col].dtype == object else "numerical"
data, feats, feat_map = preprocess(df, y, feat_types)
split_x, feat_type, origin_idx = format_data(data, feats, feat_map, feat_types)


# %%
pack = {
    "data": data,
    "feat_map": feat_map,
    "feat_type": feat_type,
    "origin_idx": origin_idx,
    "y": y
}
with open("processed.data", "wb") as f:
    pickle.dump(pack, f)