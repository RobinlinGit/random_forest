# %%
import pickle
import json
from collections import Counter
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from utils import balance_sample, bootstrap_sample
from sklearn.metrics import precision_recall_curve, classification_report, average_precision_score
import matplotlib.pyplot as plt


#%%
with open("processed.data", "rb") as f:
    pack = pickle.load(f)
y = pack['y']

# %%
with open("sklearn_result.json", "r") as f:
    oob = json.load(f)

# %%
oob = [np.sum(x) / len(x) for x in oob]

# %%
precision, recall, threshold = precision_recall_curve(y, oob)
average_precision_score(y, oob)
# %%
plt.plot(recall, precision)


# %%
data = pack["data"]
rf = RandomForestClassifier(n_estimators=100,
                            max_depth=40,
                            oob_score=True)
rf.fit(data, y)
pred_y = rf.predict_proba(data)

# %%
