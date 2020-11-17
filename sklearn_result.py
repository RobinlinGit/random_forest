import pickle
import json
from forest import RandomForest



with open("origin.data", "rb") as f:
    pack = pickle.load(f)
data = pack["data"]
y = pack["y"]
feat_type = pack["feat_type"]

rf = RandomForest(1000, 40, 10, 10, True, True)
rf.fit(data, y, feat_type)
result = rf.get_oob()
with open("./sklearn_result.json", "w") as f:
    f.write(json.dumps(result))
