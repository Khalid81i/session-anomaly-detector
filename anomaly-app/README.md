# Session Anomaly Detector

A live demo of **unsupervised anomaly detection** for abnormal session behaviour.

The model (Isolation Forest) is trained on **normal sessions only** — it never sees an
attack during training — and flags sessions that deviate from the learned normal. This
mirrors real-world anomaly problems (e.g. anti-piracy, intrusion detection) where you
rarely have labels for what's malicious, so you learn "normal" and flag the unusual.

- **Data:** NSL-KDD (network sessions; 125,973 rows)
- **Method:** Isolation Forest, trained on normal-only traffic
- **Validation:** PR-AUC ≈ 0.99 on a held-out normal+attack mix (NSL-KDD is a separable
  benchmark, so this reflects the dataset as much as the model — real adversarial data
  would be harder)
- **Serving:** model trained offline (`train_model.py` → `anomaly_bundle.joblib`); the app
  loads the saved model and scores a pasted session.

## Run
```
pip install -r requirements.txt
streamlit run app.py
```
Use the **Load a normal / attack session** buttons, then **Score session**.

*Portfolio demo, not a production system.*
