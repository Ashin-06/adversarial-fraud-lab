"""Attack studio: watch an evasion attack unfold against each defence.

    python -m advlab train      # once (a couple of minutes)
    streamlit run app.py
"""
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from advlab.attacks import greedy, profit, transfer
from advlab.defenses import PROBING_LIMIT
from advlab.threat import ThreatModel

st.set_page_config(page_title="Adversarial Fraud Lab", layout="wide")
st.title("Adversarial Fraud Lab")
st.caption("A fraudster holding a stolen card adjusts what they control (amount, time, merchant, device, VPN, "
           "address, email) until the bank's model approves. Compare how each defence holds up.")

ART = Path("artifacts/lab.joblib")
if not ART.exists():
    st.info("Run `python -m advlab train` once to build the models, then reload.")
    st.stop()


@st.cache_resource
def lab():
    return joblib.load(ART)


parts, base_threat, models, sur = lab()
raw, y = parts["test"]

with st.sidebar:
    st.subheader("Attacker")
    budget = st.slider("Effort budget", 0.0, 8.0, float(base_threat.budget), 0.5,
                       help="device spoof 2 · aged email 1.5 · drop address 1.5 · VPN 0.5 · slowing down 0.5")
    min_amount = st.number_input("Smallest theft worth doing (₹)", 500, 20000, int(base_threat.min_amount), 500)
    lockout = st.toggle(f"Card lockout after {PROBING_LIMIT} declines", value=False)
threat = ThreatModel(min_amount=float(min_amount), budget=float(budget))

frauds = raw[(y == 1) & (raw["amount"] >= threat.min_amount).to_numpy()]
caught = frauds[models[0].score_raw(frauds) >= models[0].threshold].reset_index(drop=True)
idx = st.selectbox("Stolen-card transaction (caught by the baseline model)", caught.index,
                   format_func=lambda i: f"#{i}: ₹{caught.loc[i, 'amount']:,.0f} at {int(caught.loc[i, 'hour']):02d}:00, "
                                         f"{caught.loc[i, 'mcc']}, new device={bool(caught.loc[i, 'new_device'])}")
orig = caught.loc[idx]

tab_attack, tab_results = st.tabs(["Attack one transaction", "Benchmark results"])
with tab_attack:
    cols = st.columns(len(models))
    for col, m in zip(cols, models):
        r = greedy(m, orig, threat, max_queries=PROBING_LIMIT if lockout else 400)
        tr = transfer(m, sur, orig, threat)
        _, stolen = profit(m, sur, orig, threat)
        with col:
            st.markdown(f"#### {m.name}")
            st.metric("Black-box attack", "EVADED" if r.success else "caught", f"{r.queries} attempts", delta_color="off")
            st.metric("Offline (surrogate) attack", "EVADED" if tr.success else "caught")
            st.metric("Profit-seeking attacker gets", f"₹{stolen:,.0f}", f"of ₹{orig['amount']:,.0f}", delta_color="off")
            fig = go.Figure(go.Scatter(y=[s for _, _, s in r.path], mode="lines+markers",
                                       text=[f"{f} → {v}" for f, v, _ in r.path], hoverinfo="text+y"))
            fig.add_hline(y=m.threshold, line_dash="dot", annotation_text="decline above")
            fig.update_layout(height=260, margin=dict(t=10, b=10), xaxis_title="step", yaxis_title="fraud score")
            st.plotly_chart(fig, width="stretch", key=f"p{m.name}")
            steps = pd.DataFrame([{"change": f, "to": v, "score": round(s, 3)} for f, v, s in r.path[1:]])
            if len(steps):
                st.dataframe(steps, hide_index=True, width="stretch")
    st.markdown("**Original transaction**")
    st.dataframe(orig.to_frame().T, hide_index=True, width="stretch")

with tab_results:
    res = Path("results")
    if (res / "results.csv").exists():
        st.image(str(res / "summary.png"))
        st.dataframe(pd.read_csv(res / "results.csv").round(3), hide_index=True, width="stretch")
    else:
        st.info("Run `python -m advlab evaluate` to produce the benchmark.")
