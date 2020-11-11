# %%
import numpy as np
import pandas as pd
from preprocess import preprocess
from tree import CartTreeNode
# %%

df = pd.read_csv("./bank-additional-full.csv")
y = df['y'].values
y = np.vectorize({"yes": 1, "no": 0}.__getitem__)(y)
df = df.drop(columns=["y"])
feat_types = {}
for col in df:
    feat_types[col] = "categorical" if df[col].dtype == object else "numerical"

# %%
print(df.dtypes)

# %%
data, feat_names, feat_map = preprocess(df, y, feat_types)
# %%

# %%
n = CartTreeNode(0, 10, 20, 2)
n.fit(data, y, feat_names, feat_map, feat_types)
# %%
