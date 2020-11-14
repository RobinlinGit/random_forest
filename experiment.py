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
    with open("processed_keep.data", "rb") as f:
        pack = pickle.load(f)
    data = pack["data"]
    y = pack["y"]
    feat_type = pack["feat_type"]
    for num_tree in [50, 100, 200, 500, 1000, 2000]:
        for max_depth in [10, 20, 30, 40]:
            for min_samples in [1, 5, 10, 20]:
                min_m = 1
                filename = f"./result/{num_tree}-{max_depth}-{min_samples}-{min_m}.json"
                train_forests(
                    num_tree,
                    int(num_tree / 50),
                    max_depth,
                    min_samples,
                    min_m,
                    data,
                    y,
                    feat_type,
                    True,
                    "./result_keep",
                    True
                )