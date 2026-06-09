"""
Session Anomaly Detector — live demo
Loads a pre-trained Isolation Forest (trained offline on NSL-KDD normal traffic)
and scores a pasted session as NORMAL or ANOMALOUS.
"""
import joblib, numpy as np, pandas as pd, streamlit as st

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
.verdict{{padding:1rem 1.2rem;border-radius:0;font-family:'Source Serif 4',serif;font-size:1.3rem;font-weight:600;margin:.6rem 0;}}
.norm{{border-left:5px solid {TEAL};background:#f2f7f6;color:{TEAL};}}
.anom{{border-left:5px solid #b3261e;background:#fbf2f1;color:#b3261e;}}
</style>""", unsafe_allow_html=True)

B = joblib.load("anomaly_bundle.joblib")

NORMAL_EXAMPLE = "0,tcp,ftp_data,SF,491,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,0.0,0.0,0.0,0.0,1.0,0.0,0.0,150,25,0.17,0.03,0.17,0.0,0.0,0.0,0.05,0.0"
ATTACK_EXAMPLE = "0,tcp,netbios_ns,S0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,96,16,1.0,1.0,0.0,0.0,0.17,0.05,0.0,255,2,0.01,0.06,0.0,0.0,1.0,1.0,0.0,0.0"

st.markdown('<div class="eyebrow">Unsupervised anomaly detection</div>', unsafe_allow_html=True)
st.title("Session Anomaly Detector")
st.markdown('<div class="disc">The model was trained on <b>normal sessions only</b> and flags '
            'deviations — it was never shown what an attack looks like. Trained offline on the '
            'NSL-KDD dataset; this app loads the saved model and scores a session you paste in. '
            'A portfolio demo, not a production system.</div>', unsafe_allow_html=True)

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
        if len(vals) != len(B["raw_cols"]):
            st.error(f"Expected {len(B['raw_cols'])} values, got {len(vals)}.")
        else:
            typed = []
            for c, v in zip(B["raw_cols"], vals):
                if c in ("protocol_type", "service", "flag"):
                    typed.append(v)
                else:
                    typed.append(float(v))
            df = pd.DataFrame([typed], columns=B["raw_cols"])
            df = pd.get_dummies(df, columns=["protocol_type", "service", "flag"])
            df = df.reindex(columns=B["feat_cols"], fill_value=0)
            s = -B["model"].score_samples(B["scaler"].transform(df.values))[0]
            anom = s >= B["threshold"]
            cls = "anom" if anom else "norm"
            label = "ANOMALOUS" if anom else "NORMAL"
            st.markdown(f'<div class="verdict {cls}">{label}</div>', unsafe_allow_html=True)
            st.caption(f"Anomaly score {s:.3f}  ·  flag threshold {B['threshold']:.3f} "
                       f"(set at the 95th percentile of normal traffic)")
            st.progress(min(1.0, s / (B['threshold'] * 1.6)))
    except ValueError:
        st.error("Couldn't parse a numeric value — check the row.")

st.caption("Trained on normal-only traffic · Isolation Forest · validated PR-AUC ≈ 0.99 on a "
           "held-out mix. NSL-KDD is a separable benchmark, so real adversarial data would be harder.")
