# %%
import numpy as np
import pandas as pd
from preprocess import preprocess
from tree import CartTree
# %%

df = pd.read_csv("./bank-additional-full.csv")
y = df['y'].values
y = np.vectorize({"yes": 1, "no": 0}.__getitem__)(y)
df = df.drop(columns=["y", "duration"])
feat_types = {}
for col in df:
    feat_types[col] = "categorical" if df[col].dtype == object else "numerical"

# %%
data, feats, feat_map = preprocess(df.copy(), y, feat_types)
# %%
t = CartTree(10, 20, 2)
t.fit(df.copy(), y, feat_types)
# %%
# %%
