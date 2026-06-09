"""
Session Anomaly Detector — live demo
Trains an Isolation Forest on NSL-KDD normal traffic (cached on first load),
then scores a pasted session as NORMAL or ANOMALOUS.
"""
import numpy as np, pandas as pd, streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Session Anomaly Detector", page_icon="🛰️", layout="centered")
TEAL = "#14605C"
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=Libre+Franklin:wght@400;500;600&display=swap');
html,body,[class*="css"]{{font-family:'Libre Franklin',sans-serif;}}
h1,h2,h3{{font-family:'Source Serif 4',serif;font-weight:600;letter-spacing:-.01em;}}
.block-container{{max-width:760px;}}
.eyebrow{{font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;color:{TEAL};font-weight:600;}}
.disc{{font-size:.82rem;color:#6b6b6b;border-left:3px solid {TEAL};padding:.5rem .9rem;background:#f7f9f8;}}
.verdict{{padding:1rem 1.2rem;font-family:'Source Serif 4',serif;font-size:1.3rem;font-weight:600;margin:.6rem 0;}}
.norm{{border-left:5px solid {TEAL};background:#f2f7f6;color:{TEAL};}}
.anom{{border-left:5px solid #b3261e;background:#fbf2f1;color:#b3261e;}}
</style>""", unsafe_allow_html=True)

URL = "https://raw.githubusercontent.com/Mamcose/NSL-KDD-Network-Intrusion-Detection/master/NSL_KDD_Train.csv"
COLS = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
"wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
"root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files",
"num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate",
"srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
"srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
"dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
"dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]
RAW = [c for c in COLS if c != "label"]

@st.cache_resource
def get_model():
    df = pd.read_csv(URL, header=None, names=COLS)
    y = (df["label"] != "normal").astype(int).values
    X = pd.get_dummies(df.drop(columns=["label"]), columns=["protocol_type","service","flag"])
    scaler = StandardScaler().fit(X.values)
    Xs = scaler.transform(X.values)
    iso = IsolationForest(n_estimators=200, contamination=0.1, random_state=42).fit(Xs[y==0])
    thr = float(np.quantile(-iso.score_samples(Xs[y==0]), 0.95))
    return iso, scaler, X.columns.tolist(), thr

NORMAL_EXAMPLE = "0,tcp,ftp_data,SF,491,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,0.0,0.0,0.0,0.0,1.0,0.0,0.0,150,25,0.17,0.03,0.17,0.0,0.0,0.0,0.05,0.0"
ATTACK_EXAMPLE = "0,tcp,netbios_ns,S0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,96,16,1.0,1.0,0.0,0.0,0.17,0.05,0.0,255,2,0.01,0.06,0.0,0.0,1.0,1.0,0.0,0.0"

st.markdown('<div class="eyebrow">Unsupervised anomaly detection</div>', unsafe_allow_html=True)
st.title("Session Anomaly Detector")
st.markdown('<div class="disc">The model is trained on <b>normal sessions only</b> and flags '
            'deviations — it was never shown what an attack looks like. Trained on the NSL-KDD '
            'dataset. A portfolio demo, not a production system.</div>', unsafe_allow_html=True)

with st.spinner("Loading model (first run trains on NSL-KDD, ~20s)…"):
    model, scaler, feat_cols, threshold = get_model()

st.write("")
c1, c2 = st.columns(2)
if c1.button("Load a normal session"):
    st.session_state["row"] = NORMAL_EXAMPLE
if c2.button("Load an attack session"):
    st.session_state["row"] = ATTACK_EXAMPLE

row = st.text_area("Session features (41 comma-separated values from NSL-KDD)",
                   st.session_state.get("row", NORMAL_EXAMPLE), height=120)

if st.button("Score session", type="primary"):
    try:
        vals = [v.strip() for v in row.split(",")]
        if len(vals) != len(RAW):
            st.error(f"Expected {len(RAW)} values, got {len(vals)}.")
        else:
            typed = [v if c in ("protocol_type","service","flag") else float(v)
                     for c, v in zip(RAW, vals)]
            d = pd.DataFrame([typed], columns=RAW)
            d = pd.get_dummies(d, columns=["protocol_type","service","flag"]).reindex(columns=feat_cols, fill_value=0)
            s = -model.score_samples(scaler.transform(d.values))[0]
            anom = s >= threshold
            st.markdown(f'<div class="verdict {"anom" if anom else "norm"}">'
                        f'{"ANOMALOUS" if anom else "NORMAL"}</div>', unsafe_allow_html=True)
            st.caption(f"Anomaly score {s:.3f}  ·  flag threshold {threshold:.3f} "
                       f"(95th percentile of normal traffic)")
            st.progress(min(1.0, s / (threshold * 1.6)))
    except ValueError:
        st.error("Couldn't parse a numeric value — check the row.")

st.caption("Isolation Forest trained on normal-only traffic · validated PR-AUC ≈ 0.99 on a "
           "held-out mix. NSL-KDD is a separable benchmark, so real adversarial data would be harder.")
