"""
train_model.py — trains the anomaly detector offline and saves anomaly_bundle.joblib.
Run once (e.g. in Colab); the app then just loads the saved bundle.
"""
import pandas as pd, numpy as np, joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

URL = "https://raw.githubusercontent.com/Mamcose/NSL-KDD-Network-Intrusion-Detection/master/NSL_KDD_Train.csv"
COLS = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
"wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
"root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files",
"num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate",
"srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
"srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
"dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
"dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]

df = pd.read_csv(URL, header=None, names=COLS)
df["anomaly"] = (df["label"] != "normal").astype(int)

X = pd.get_dummies(df.drop(columns=["label","anomaly"]),
                   columns=["protocol_type","service","flag"])
scaler = StandardScaler().fit(X.values)
Xs = scaler.transform(X.values)

# train on NORMAL traffic only — the model learns 'normal', never sees attacks
normal_idx = np.where(df["anomaly"].values == 0)[0]
iso = IsolationForest(n_estimators=200, contamination=0.1, random_state=42).fit(Xs[normal_idx])

# flag line = 95th percentile of normal-traffic anomaly scores
thr = float(np.quantile(-iso.score_samples(Xs[normal_idx]), 0.95))

joblib.dump({"model": iso, "scaler": scaler,
             "feat_cols": X.columns.tolist(),
             "raw_cols": [c for c in COLS if c != "label"],
             "threshold": thr}, "anomaly_bundle.joblib")
print("saved anomaly_bundle.joblib  (threshold {:.3f})".format(thr))
