# %%
import numpy as np
import pandas as pd
import random
import pickle
from multiprocessing import freeze_support, Queue, Pool
from preprocess import preprocess, format_data
from sklearn.metrics import classification_report
from forest import RandomForest, train_forests




if __name__ == "__main__":
    freeze_support()


    df = pd.read_csv("./bank-additional-full.csv")
    y = df['y'].values
    y = np.vectorize({"yes": 1, "no": 0}.__getitem__)(y)
    df = df.drop(columns=["y", "duration"])
    feat_types = {}
    for col in df:
        feat_types[col] = "categorical" if df[col].dtype == object else "numerical"

    data, feats, feat_map = preprocess(df, y, feat_types)
    split_x, feat_type, origin_idx = format_data(data, feats, feat_map, feat_types)

    train_forests(4, 4, 10, 20, 2, data, y, feat_type, "rf", "rf")
