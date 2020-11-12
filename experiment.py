# %%
import numpy as np
import pandas as pd
import random
from preprocess import preprocess, format_data
from sklearn.metrics import classification_report
from utils import filter_data
from tree import RandomForest, CartTree
# %%

df = pd.read_csv("./bank-additional-full.csv")
y = df['y'].values
y = np.vectorize({"yes": 1, "no": 0}.__getitem__)(y)
df = df.drop(columns=["y", "duration"])
feat_types = {}
for col in df:
    feat_types[col] = "categorical" if df[col].dtype == object else "numerical"

# %%
data, feats, feat_map = preprocess(df, y, feat_types)
split_x, feat_type, origin_idx = format_data(data, feats, feat_map, feat_types)

# %%
idxes = list(range(data.shape[0]))
random.shuffle(idxes)
tl = int(0.8 * len(idxes))
train_X, train_y = data[idxes[: tl]], y[idxes[: tl]]
test_X, test_y = data[idxes[tl:]], y[idxes[tl:]]
# %%
t = CartTree(10, 20, 2)
t.fit(train_X, train_y, split_x, origin_idx, feat_type)
# %%
# forest = RandomForest(10, 10, 2, 10)
# forest.fit(train_X, train_y, feat_type, split_x)

# %%
pred_y = t.predict(test_X)
print(classification_report(test_y, pred_y))