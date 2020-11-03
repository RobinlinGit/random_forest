# %%
import numpy as np
import pandas as pd

# %%

df = pd.read_csv("./bank-additional-full.csv")
# %%
print(df.dtypes)
# %%
print(set(df["duration"]))
# %%
data = df["pdays"].values
print(data.dtype)
# %%
print(data[:10])
# %%
print(set(data))
# %%
print(data)
y = df["y"].values
print(y[:10])
# %%
print(y.names)
# %%
