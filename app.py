"""
app.py — Motor Incipient Failure Detection Dashboard
LozanoLsa · Project 03 · K-Nearest Neighbors · 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KNN · Motor Failure Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── FULL CSS INJECTION ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;600&family=Instrument+Serif:ital@0;1&display=swap');

/* ╔══ VARIABLES ══╗ */
:root {
    --bg:       #080c12;
    --surface:  #0e1420;
    --card:     #121922;
    --card2:    #161f2e;
    --border:   #1e2d45;
    --blue:     #3b82f6;
    --blue2:    #60a5fa;
    --teal:     #2dd4bf;
    --danger:   #f87171;
    --warn:     #fbbf24;
    --text:     #c8d8f0;
    --muted:    #4e6a8a;
    --fh: 'Syne', sans-serif;
    --fm: 'JetBrains Mono', monospace;
    --fs: 'Instrument Serif', Georgia, serif;
}

/* ╔══ BASE ══╗ */
.stApp {
    background: var(--bg) !important;
    color: var(--text);
    font-family: var(--fh);
}
.block-container {
    padding: 1.8rem 2.4rem 3rem !important;
    max-width: 1400px !important;
}
#MainMenu, footer, header { visibility: hidden; }

/* ╔══ SIDEBAR ══╗ */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 1.5rem 1rem !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    font-family: var(--fm) !important;
    font-size: 0.72rem !important;
    color: var(--muted) !important;
    letter-spacing: 0.05em;
}
[data-testid="stSidebar"] label {
    font-family: var(--fm) !important;
    font-size: 0.7rem !important;
    color: var(--text) !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}

/* ╔══ SLIDERS ══╗ */
[data-testid="stSlider"] [role="slider"] {
    background: var(--blue) !important;
    border: 2px solid var(--blue2) !important;
    box-shadow: 0 0 8px rgba(59,130,246,0.5) !important;
}
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] {
    font-family: var(--fm) !important;
    font-size: 0.65rem !important;
    color: var(--blue2) !important;
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    padding: 1px 5px !important;
    border-radius: 3px !important;
}
[data-testid="stSlider"] > div > div > div > div {
    background: var(--blue) !important;
}

/* ╔══ SELECTBOX ══╗ */
[data-testid="stSelectbox"] > div > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: var(--fm) !important;
    font-size: 0.78rem !important;
    border-radius: 3px !important;
}

/* ╔══ METRIC CARDS ══╗ */
[data-testid="stMetric"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-top: 2px solid var(--blue) !important;
    padding: 1rem 1.1rem 0.9rem !important;
    border-radius: 3px !important;
}
[data-testid="stMetricLabel"] > div {
    font-family: var(--fm) !important;
    font-size: 0.6rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.18em !important;
    color: var(--muted) !important;
    font-weight: 400 !important;
}
[data-testid="stMetricValue"] > div {
    font-family: var(--fm) !important;
    font-size: 1.7rem !important;
    font-weight: 600 !important;
    color: var(--blue2) !important;
    line-height: 1.1 !important;
}
[data-testid="stMetricDelta"] > div {
    font-family: var(--fm) !important;
    font-size: 0.68rem !important;
}

/* ╔══ TABS ══╗ */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
    background: transparent !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: var(--fm) !important;
    font-size: 0.68rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    color: var(--muted) !important;
    padding: 0.5rem 1.2rem !important;
    border: none !important;
    border-radius: 0 !important;
    background: transparent !important;
    transition: all 0.2s !important;
}
[data-testid="stTabs"] [role="tab"]:hover {
    color: var(--blue2) !important;
    background: rgba(59,130,246,0.06) !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--blue) !important;
    border-bottom: 2px solid var(--blue) !important;
    background: transparent !important;
}
[data-testid="stTabsContent"] {
    padding-top: 1.4rem !important;
}

/* ╔══ ALERTS ══╗ */
[data-testid="stAlert"] {
    border-radius: 2px !important;
    font-family: var(--fm) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.04em !important;
    border: none !important;
    border-left: 3px solid !important;
}

/* ╔══ EXPANDER ══╗ */
[data-testid="stExpander"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
    margin-bottom: 6px !important;
}
[data-testid="stExpander"] summary {
    font-family: var(--fm) !important;
    font-size: 0.72rem !important;
    color: var(--text) !important;
    letter-spacing: 0.06em !important;
}
[data-testid="stExpander"] p {
    font-family: var(--fm) !important;
    font-size: 0.72rem !important;
    color: var(--muted) !important;
    line-height: 1.7 !important;
}

/* ╔══ DATAFRAME ══╗ */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
}
[data-testid="stDataFrame"] th {
    font-family: var(--fm) !important;
    font-size: 0.62rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    background: var(--card2) !important;
    color: var(--muted) !important;
    border-bottom: 1px solid var(--border) !important;
}
[data-testid="stDataFrame"] td {
    font-family: var(--fm) !important;
    font-size: 0.72rem !important;
    color: var(--text) !important;
    background: var(--card) !important;
}

/* ╔══ DIVIDER / CAPTION ══╗ */
hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }
[data-testid="stCaptionContainer"] p {
    font-family: var(--fm) !important;
    font-size: 0.62rem !important;
    color: var(--muted) !important;
    letter-spacing: 0.08em !important;
}

/* ╔══ HEADINGS ══╗ */
h1, h2, h3 {
    font-family: var(--fh) !important;
    color: var(--text) !important;
    letter-spacing: -0.01em !important;
}
h2 { font-size: 1.1rem !important; font-weight: 700 !important; }
h3 { font-size: 0.9rem !important; font-weight: 600 !important; }
p, li { font-family: var(--fh) !important; font-size: 0.88rem !important; }

/* ╔══ CUSTOM COMPONENTS ══╗ */
.lsa-header {
    border-bottom: 1px solid var(--border);
    padding-bottom: 1.2rem;
    margin-bottom: 0.2rem;
}
.lsa-project-tag {
    font-family: var(--fm);
    font-size: 0.6rem;
    color: var(--blue);
    text-transform: uppercase;
    letter-spacing: 0.22em;
    margin-bottom: 4px;
}
.lsa-title {
    font-family: var(--fh);
    font-size: 1.85rem;
    font-weight: 800;
    color: #fff;
    line-height: 1.1;
    letter-spacing: -0.02em;
}
.lsa-tagline {
    font-family: var(--fs);
    font-style: italic;
    font-size: 0.9rem;
    color: var(--muted);
    margin-top: 4px;
}
.lsa-chip {
    display: inline-block;
    background: rgba(59,130,246,0.1);
    border: 1px solid rgba(59,130,246,0.3);
    color: var(--blue2);
    font-family: var(--fm);
    font-size: 0.58rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 2px;
    margin-right: 5px;
}
.lsa-section {
    font-family: var(--fm);
    font-size: 0.6rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.2em;
    margin-bottom: 10px;
    padding-bottom: 5px;
    border-bottom: 1px solid var(--border);
}
.lsa-footer {
    margin-top: 2.5rem;
    padding-top: 0.8rem;
    border-top: 1px solid var(--border);
    font-family: var(--fm);
    font-size: 0.58rem;
    color: var(--muted);
    letter-spacing: 0.1em;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
DATA_PATH    = "motor_failure_prediction_data.csv"
RANDOM_STATE = 42
FEATURES = ["vib_rms_mms", "vib_peak_to_peak_mms", "bearing_temp_c",
            "motor_current_a", "dominant_freq_hz", "load_pct"]
TARGET = "incipient_failure"

FEAT_LABELS = {
    "vib_rms_mms":          "Vibration RMS",
    "vib_peak_to_peak_mms": "Vib Peak-to-Peak",
    "bearing_temp_c":       "Bearing Temp",
    "motor_current_a":      "Motor Current",
    "dominant_freq_hz":     "Dominant Freq",
    "load_pct":             "Load %",
}
ACTION_MAP = {
    "vib_rms_mms":          "Schedule vibration analysis — RMS above 3.5 mm/s warrants immediate bearing inspection",
    "vib_peak_to_peak_mms": "Check for imbalance or looseness — peak-to-peak elevation indicates impact events",
    "bearing_temp_c":       "Inspect lubrication and bearing condition — temp above 75°C indicates friction buildup",
    "motor_current_a":      "Check mechanical load and electrical connections — overcurrent signals increased resistance",
    "dominant_freq_hz":     "Perform spectrum analysis — frequency shift below 27 Hz may indicate misalignment",
    "load_pct":             "Review load assignment — sustained load above 88% accelerates mechanical wear",
}
METRIC_EXPL = {
    "Accuracy":  "Out of every 100 motor readings, the model classifies this many correctly.",
    "Precision": "When the model flags a motor as failing, how often it's actually in failure state.",
    "Recall":    "Out of all truly failing motors, how many the model catches. Missed failures are the costliest error.",
    "F1 Score":  "Balances precision and recall — critical when false negatives (missed failures) are expensive.",
    "AUC-ROC":   "Separates failing from normal motors across all thresholds. 1.0 = perfect.",
}

# ─── PLOTLY COLORS ────────────────────────────────────────────────────────────
C_BLUE   = "#3b82f6"
C_BLUE2  = "#60a5fa"
C_TEAL   = "#2dd4bf"
C_DANGER = "#f87171"
C_WARN   = "#fbbf24"
C_MUTED  = "#4e6a8a"

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono", color=C_MUTED, size=10),
    xaxis=dict(gridcolor="#1e2d45", linecolor="#1e2d45", zeroline=False),
    yaxis=dict(gridcolor="#1e2d45", linecolor="#1e2d45", zeroline=False),
    margin=dict(l=4, r=4, t=40, b=4),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
)

def plot(fig, h=300):
    fig.update_layout(height=h, **PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

# ─── DATA & MODEL ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        return pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        return pd.read_csv(
            "https://raw.githubusercontent.com/LozanoLsa/Motor_Failures_Prediction/main/motor_failure_prediction_data.csv"
        )

@st.cache_resource
def train_model(df):
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )
    pipe = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())])
    grid = GridSearchCV(
        pipe,
        {"knn__n_neighbors": [3, 5, 7, 9, 11, 15],
         "knn__weights": ["uniform", "distance"],
         "knn__p": [1, 2]},
        cv=5, scoring="accuracy", n_jobs=-1
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    yp  = best.predict(X_test)
    ypr = best.predict_proba(X_test)[:, 1]
    metrics = {
        "Accuracy":  accuracy_score(y_test, yp),
        "Precision": precision_score(y_test, yp),
        "Recall":    recall_score(y_test, yp),
        "F1 Score":  f1_score(y_test, yp),
        "AUC-ROC":   roc_auc_score(y_test, ypr),
    }
    pi = permutation_importance(best, X_test, y_test, n_repeats=10,
                                random_state=RANDOM_STATE, n_jobs=-1)
    imp_df = pd.DataFrame({
        "Feature":    FEATURES,
        "Importance": pi.importances_mean,
        "Std":        pi.importances_std,
    }).sort_values("Importance", ascending=False)
    return best, grid.best_params_, X_train, X_test, y_train, y_test, yp, ypr, metrics, imp_df

df = load_data()
best_knn, best_params, X_train, X_test, y_train, y_test, y_pred, y_prob, metrics, imp_df = train_model(df)
failure_rate = df[TARGET].mean()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="lsa-project-tag">LozanoLsa · Project 03</div>
    <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:800;
                color:#fff;margin-bottom:6px;">Motor Failure<br>Predictor</div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                color:#4e6a8a;line-height:1.7;margin-bottom:12px;">
        KNN trained on 1,500 motor<br>condition records · 6 sensors
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="lsa-section">// Vibration</div>', unsafe_allow_html=True)
    vib_rms = st.slider("RMS (mm/s)",          0.5,  7.0,  2.2, 0.1)
    vib_pp  = st.slider("Peak-to-Peak (mm/s)", 1.0, 14.0,  5.0, 0.1)

    st.markdown('<div class="lsa-section">// Thermal & Electrical</div>', unsafe_allow_html=True)
    bear_t  = st.slider("Bearing Temp (°C)",   40.0, 100.0, 60.0, 0.5)
    curr_a  = st.slider("Motor Current (A)",   18.0,  45.0, 28.0, 0.5)

    st.markdown('<div class="lsa-section">// Spectral & Load</div>', unsafe_allow_html=True)
    freq_hz = st.slider("Dominant Freq (Hz)",  20.0, 40.0, 30.0, 0.5)
    load_p  = st.slider("Load (%)",            30.0, 100.0, 70.0, 1.0)

    st.divider()
    st.caption(
        f"k={best_params.get('knn__n_neighbors')} · "
        f"p={best_params.get('knn__p')} · "
        f"{best_params.get('knn__weights')}"
    )
    st.caption("Where f(x) meets Kaizen · 2026")

# ─── PREDICT ──────────────────────────────────────────────────────────────────
def predict_s(vr, vp, bt, mc, fd, lp):
    row = pd.DataFrame([{
        "vib_rms_mms": vr, "vib_peak_to_peak_mms": vp,
        "bearing_temp_c": bt, "motor_current_a": mc,
        "dominant_freq_hz": fd, "load_pct": lp,
    }])
    p = best_knn.predict_proba(row)[0, 1]
    return p, int(p >= 0.5)

pred_prob, pred_class = predict_s(vib_rms, vib_pp, bear_t, curr_a, freq_hz, load_p)

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="lsa-header">
    <div class="lsa-project-tag">ML Project #03 · K-Nearest Neighbors · Predictive Maintenance</div>
    <div class="lsa-title">Motor Failures Are Not Random</div>
    <div class="lsa-tagline">Distance in sensor space is operational similarity.</div>
    <div style="margin-top:10px;">
        <span class="lsa-chip">KNN</span>
        <span class="lsa-chip">6 Sensors</span>
        <span class="lsa-chip">97.8% Accuracy</span>
        <span class="lsa-chip">AUC 0.996</span>
        <span class="lsa-chip">GridSearchCV</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "DATA EXPLORER", "PERFORMANCE", "SIMULATOR", "RISK DRIVERS", "ACTION PLAN"
])

# ══ TAB 1 ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="lsa-section">// Dataset overview</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Records",     f"{len(df):,}")
    k2.metric("Incipient Failure", f"{df[TARGET].sum():,}")
    k3.metric("Normal",            f"{(df[TARGET]==0).sum():,}")
    k4.metric("Failure Rate",      f"{failure_rate:.1%}")
    st.divider()

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown('<div class="lsa-section">// Class distribution</div>', unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=["Normal", "Incipient Failure"],
            values=[(df[TARGET]==0).sum(), df[TARGET].sum()],
            marker_colors=[C_BLUE, C_DANGER],
            hole=0.52, textinfo="percent+label",
            textfont=dict(family="JetBrains Mono", size=11),
        ))
        plot(fig_pie, 270)
    with c2:
        st.markdown('<div class="lsa-section">// Feature scatter</div>', unsafe_allow_html=True)
        sc_y = st.selectbox("Y axis:", FEATURES, index=2, format_func=lambda x: FEAT_LABELS.get(x, x))
        fig_sc = px.scatter(
            df, x="vib_rms_mms", y=sc_y,
            color=df[TARGET].map({0: "Normal", 1: "Incipient Failure"}),
            color_discrete_map={"Normal": C_BLUE, "Incipient Failure": C_DANGER},
            opacity=0.45,
            labels={"vib_rms_mms": "Vibration RMS (mm/s)", sc_y: FEAT_LABELS.get(sc_y, sc_y)},
        )
        fig_sc.update_traces(marker=dict(size=4))
        plot(fig_sc, 270)

    st.divider()
    st.markdown('<div class="lsa-section">// Feature distributions</div>', unsafe_allow_html=True)
    feat_sel = st.selectbox("Feature:", FEATURES, format_func=lambda x: FEAT_LABELS.get(x, x))
    fig_hist = go.Figure()
    for cls, color, name in [(0, C_BLUE, "Normal"), (1, C_DANGER, "Incipient Failure")]:
        fig_hist.add_trace(go.Histogram(
            x=df[df[TARGET]==cls][feat_sel], name=name,
            marker_color=color, opacity=0.65, nbinsx=30,
        ))
    fig_hist.update_layout(barmode="overlay",
                           xaxis_title=FEAT_LABELS.get(feat_sel, feat_sel),
                           yaxis_title="Count")
    plot(fig_hist, 280)

# ══ TAB 2 ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="lsa-section">// Model metrics</div>', unsafe_allow_html=True)
    m_cols = st.columns(5)
    for i, (name, val) in enumerate(metrics.items()):
        m_cols[i].metric(name, f"{val:.3f}")
    st.divider()

    cm_arr = confusion_matrix(y_test, y_pred)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="lsa-section">// Confusion matrix</div>', unsafe_allow_html=True)
        fig_cm = go.Figure(go.Heatmap(
            z=cm_arr,
            x=["Pred: Normal", "Pred: Failure"],
            y=["Actual: Normal", "Actual: Failure"],
            colorscale=[[0, "#0a1525"], [1, C_BLUE]],
            text=cm_arr, texttemplate="%{text}",
            textfont=dict(family="JetBrains Mono", size=18, color="#fff"),
            showscale=False,
        ))
        plot(fig_cm, 300)
    with c2:
        st.markdown('<div class="lsa-section">// Metric breakdown</div>', unsafe_allow_html=True)
        metric_names = list(metrics.keys())
        metric_vals  = list(metrics.values())
        fig_bar = go.Figure(go.Bar(
            x=metric_vals, y=metric_names, orientation="h",
            marker_color=[C_BLUE if v >= 0.9 else C_WARN for v in metric_vals],
            text=[f"{v:.3f}" for v in metric_vals],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=11, color="#c8d8f0"),
        ))
        fig_bar.update_layout(xaxis_range=[0, 1.05])
        plot(fig_bar, 300)

    st.divider()
    st.markdown('<div class="lsa-section">// Metric explanations</div>', unsafe_allow_html=True)
    for name, expl in METRIC_EXPL.items():
        with st.expander(f"{name}  —  {metrics[name]:.3f}"):
            st.write(expl)

# ══ TAB 3 ══════════════════════════════════════════════════════════════════════
with tab3:
    left, right = st.columns([1, 2])
    with left:
        st.markdown('<div class="lsa-section">// Failure probability</div>', unsafe_allow_html=True)
        gauge_color = C_DANGER if pred_class == 1 else C_TEAL
        fg = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_prob * 100,
            number={"suffix": "%", "font": {"size": 38, "family": "JetBrains Mono", "color": "#fff"}},
            title={"text": "P(Incipient Failure)", "font": {"size": 11, "family": "JetBrains Mono", "color": C_MUTED}},
            gauge={
                "axis": {"range": [0, 100], "tickfont": {"size": 9, "family": "JetBrains Mono"}, "tickcolor": "#1e2d45"},
                "bar": {"color": gauge_color, "thickness": 0.22},
                "bgcolor": "#0e1420", "bordercolor": "#1e2d45",
                "steps": [
                    {"range": [0, 35],   "color": "rgba(45,212,191,0.08)"},
                    {"range": [35, 65],  "color": "rgba(251,191,36,0.08)"},
                    {"range": [65, 100], "color": "rgba(248,113,113,0.10)"},
                ],
                "threshold": {"line": {"color": "#fff", "width": 2}, "thickness": 0.75, "value": 50},
            }
        ))
        fg.update_layout(height=290, paper_bgcolor="rgba(0,0,0,0)",
                         margin=dict(t=50, b=10, l=20, r=20))
        st.plotly_chart(fg, use_container_width=True)
        if pred_class == 1:
            st.error("INCIPIENT FAILURE — SCHEDULE INSPECTION")
        else:
            st.success("NORMAL OPERATION — CONTINUE MONITORING")
        st.caption(f"Fleet avg: {failure_rate:.1%}  ·  This motor: {pred_prob:.1%}  ·  Δ {pred_prob - failure_rate:+.1%}")

    with right:
        st.markdown('<div class="lsa-section">// Deviation from normal-class profile</div>', unsafe_allow_html=True)
        sc_obj      = best_knn.named_steps["scaler"]
        X_curr_sc   = sc_obj.transform([[vib_rms, vib_pp, bear_t, curr_a, freq_hz, load_p]])
        normal_mean = sc_obj.transform([df[df[TARGET]==0][FEATURES].mean().values])[0]
        contrib     = X_curr_sc[0] - normal_mean
        cd = pd.DataFrame({"Feature": FEATURES, "Contribution": contrib})
        cd["Label"] = cd["Feature"].map(FEAT_LABELS)
        cd = cd.reindex(cd["Contribution"].abs().sort_values(ascending=False).index).head(6)
        fc = go.Figure(go.Bar(
            x=cd["Contribution"], y=cd["Label"], orientation="h",
            marker_color=[C_DANGER if v > 0 else C_TEAL for v in cd["Contribution"]],
            text=[f"{v:+.2f}σ" for v in cd["Contribution"]],
            textfont=dict(family="JetBrains Mono", size=10, color="#c8d8f0"),
            textposition="outside",
        ))
        fc.update_layout(xaxis_title="Std. deviations from normal mean")
        plot(fc, 270)
        if pred_prob >= 0.70:
            st.error("Priority: HIGH · Schedule inspection before end of shift")
        elif pred_prob >= 0.35:
            st.warning("Priority: MEDIUM · Increase monitoring — next 48h")
        else:
            st.success("Priority: LOW · Standard monitoring schedule")

    st.divider()
    st.markdown('<div class="lsa-section">// Scenario comparison</div>', unsafe_allow_html=True)
    bp = predict_s(2.0, 5.0, 60.0, 28.0, 30.0, 70.0)[0]
    wp = predict_s(4.8, 9.8, 88.0, 37.0, 23.0, 92.0)[0]
    cdf = pd.DataFrame([
        {"Scenario": "Best case — normal operation",   "P(Failure)": f"{bp:.1%}",        "Status": "Normal"  if bp < 0.5 else "Failure", "Δ vs current": f"{bp - pred_prob:+.1%}"},
        {"Scenario": "Your current readings",          "P(Failure)": f"{pred_prob:.1%}", "Status": "Normal"  if pred_class==0 else "Failure", "Δ vs current": "—"},
        {"Scenario": "Worst case — incipient failure", "P(Failure)": f"{wp:.1%}",        "Status": "Normal"  if wp < 0.5 else "Failure", "Δ vs current": f"{wp - pred_prob:+.1%}"},
    ])
    st.dataframe(cdf, use_container_width=True, hide_index=True)

# ══ TAB 4 ══════════════════════════════════════════════════════════════════════
with tab4:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="lsa-section">// Permutation importance</div>', unsafe_allow_html=True)
        is_ = imp_df.sort_values("Importance", ascending=True)
        fi = go.Figure(go.Bar(
            x=is_["Importance"], y=is_["Feature"].map(FEAT_LABELS), orientation="h",
            error_x=dict(type="data", array=is_["Std"].values,
                         color=C_MUTED, thickness=1.5, width=4),
            marker_color=[C_DANGER if v > 0.02 else C_BLUE for v in is_["Importance"]],
            text=[f"Δ−{v:.1%}" for v in is_["Importance"]],
            textfont=dict(family="JetBrains Mono", size=9, color="#c8d8f0"),
            textposition="outside",
        ))
        fi.update_layout(xaxis_title="Mean accuracy drop")
        plot(fi, 340)
    with c2:
        st.markdown('<div class="lsa-section">// Class-conditional feature means</div>', unsafe_allow_html=True)
        mn  = df[df[TARGET]==0][FEATURES].mean()
        mf  = df[df[TARGET]==1][FEATURES].mean()
        mdf = pd.DataFrame({"Feature": FEATURES, "Normal": mn.values, "Failure": mf.values})
        mdf["Label"] = mdf["Feature"].map(FEAT_LABELS)
        fmean = go.Figure()
        fmean.add_trace(go.Bar(name="Normal",            x=mdf["Label"], y=mdf["Normal"],  marker_color=C_BLUE))
        fmean.add_trace(go.Bar(name="Incipient Failure", x=mdf["Label"], y=mdf["Failure"], marker_color=C_DANGER))
        fmean.update_layout(barmode="group", xaxis_tickangle=-20)
        plot(fmean, 340)

    st.divider()
    st.markdown('<div class="lsa-section">// 2D decision boundary — Vibration RMS vs Bearing Temperature</div>', unsafe_allow_html=True)
    sc_obj  = best_knn.named_steps["scaler"]
    knn_obj = best_knn.named_steps["knn"]
    xi, yi  = FEATURES.index("vib_rms_mms"), FEATURES.index("bearing_temp_c")
    xv = np.linspace(df["vib_rms_mms"].min()-0.5, df["vib_rms_mms"].max()+0.5, 120)
    yv = np.linspace(df["bearing_temp_c"].min()-2, df["bearing_temp_c"].max()+2, 120)
    xx, yy = np.meshgrid(xv, yv)
    gm = np.tile(sc_obj.mean_, (xx.ravel().shape[0], 1))
    gm[:, xi] = xx.ravel()
    gm[:, yi] = yy.ravel()
    pg = knn_obj.predict_proba(sc_obj.transform(gm))[:, 1].reshape(xx.shape)
    fmap = go.Figure()
    fmap.add_trace(go.Contour(
        x=xv, y=yv, z=pg,
        colorscale=[[0, "#0a2040"], [0.5, "#1e4080"], [1, "#c0392b"]],
        opacity=0.75, contours=dict(showlabels=False),
        colorbar=dict(title=dict(text="P(Failure)", font=dict(family="JetBrains Mono", size=10))),
    ))
    for cls, color, name in [(0, C_BLUE, "Normal"), (1, C_DANGER, "Incipient Failure")]:
        mask = df[TARGET] == cls
        fmap.add_trace(go.Scatter(
            x=df.loc[mask, "vib_rms_mms"], y=df.loc[mask, "bearing_temp_c"],
            mode="markers", name=name,
            marker=dict(color=color, size=3, opacity=0.35),
        ))
    fmap.update_layout(xaxis_title="Vibration RMS (mm/s)", yaxis_title="Bearing Temperature (°C)")
    plot(fmap, 430)
    st.caption("Non-linear KNN boundary — confirms why a distance-based model fits this problem better than a linear classifier.")

# ══ TAB 5 ══════════════════════════════════════════════════════════════════════
with tab5:
    if pred_prob >= 0.70:
        pl, hz, ac = "HIGH",   "Before end of shift", "Take motor offline — inspect bearings, lubrication, and electrical connections"
    elif pred_prob >= 0.35:
        pl, hz, ac = "MEDIUM", "Within 48 hours",     "Increase sensor monitoring frequency — flag for next maintenance window"
    else:
        pl, hz, ac = "LOW",    "Standard schedule",   "Normal operation — maintain standard monitoring interval"

    badge_color = {"HIGH": C_DANGER, "MEDIUM": C_WARN, "LOW": C_TEAL}[pl]
    st.markdown(f"""
    <div style="background:var(--card);border:1px solid var(--border);border-left:3px solid {badge_color};
                padding:1.2rem 1.4rem;border-radius:2px;margin-bottom:1rem;">
        <div style="font-family:var(--fm);font-size:0.6rem;color:var(--muted);
                    text-transform:uppercase;letter-spacing:.18em;margin-bottom:8px;">// Action plan</div>
        <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:12px;">
            <div>
                <div style="font-family:var(--fm);font-size:0.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:.12em;">Priority</div>
                <div style="font-family:var(--fh);font-size:1.3rem;font-weight:800;color:{badge_color};">{pl}</div>
            </div>
            <div>
                <div style="font-family:var(--fm);font-size:0.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:.12em;">Est. Failure Prob</div>
                <div style="font-family:var(--fm);font-size:1.3rem;font-weight:600;color:#fff;">{pred_prob:.1%}</div>
            </div>
            <div>
                <div style="font-family:var(--fm);font-size:0.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:.12em;">Horizon</div>
                <div style="font-family:var(--fm);font-size:0.85rem;color:var(--text);">{hz}</div>
            </div>
            <div>
                <div style="font-family:var(--fm);font-size:0.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:.12em;">Owner</div>
                <div style="font-family:var(--fm);font-size:0.85rem;color:var(--text);">Maintenance Engineer</div>
            </div>
        </div>
        <div style="margin-top:12px;font-family:var(--fm);font-size:0.72rem;color:var(--text);">
            <span style="color:var(--muted);">Action → </span>{ac}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="lsa-section">// Active risk factors</div>', unsafe_allow_html=True)
    rfs = []
    if vib_rms > 3.5: rfs.append("vib_rms_mms")
    if vib_pp  > 7.5: rfs.append("vib_peak_to_peak_mms")
    if bear_t  > 75:  rfs.append("bearing_temp_c")
    if curr_a  > 33:  rfs.append("motor_current_a")
    if freq_hz < 27:  rfs.append("dominant_freq_hz")
    if load_p  > 88:  rfs.append("load_pct")

    if rfs:
        for f in rfs[:4]:
            with st.expander(f"▲  {FEAT_LABELS.get(f, f)}  —  above normal threshold"):
                st.write(ACTION_MAP[f])
    else:
        st.success("All sensor readings are within normal operating ranges.")

    st.divider()
    st.caption("This tool supports maintenance decisions — it does not replace certified condition monitoring procedures or safety protocols.")

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="lsa-footer">
    LozanoLsa · Where f(x) meets Kaizen · KNN Motor Failure Predictor · Project 03 · v2.0
</div>
""", unsafe_allow_html=True)
