import pickle
from sklearn.ensemble import RandomForestClassifier


with open("origin.data", "rb") as f:
    pack = pickle.load(f)
data = pack["data"]
y = pack["y"]
feat_type = pack["feat_type"]

rf = RandomForestClassifier(n_estimators=1000,
                            oob_score=True)
rf.fit(data, y)
pred_y = rf.predict_proba(data)
