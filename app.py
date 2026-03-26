"""
app.py — Motor Incipient Failure Detection Dashboard
LozanoLsa · Operational Excellence · ML Portfolio · 2026
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
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

st.set_page_config(page_title="Motor Failure Predictor", page_icon="⚙️",
                   layout="wide", initial_sidebar_state="expanded")

DATA_PATH    = "motor_failure_prediction_data.csv"
RANDOM_STATE = 42
FEATURES = ["vib_rms_mms", "vib_peak_to_peak_mms", "bearing_temp_c",
            "motor_current_a", "dominant_freq_hz", "load_pct"]
TARGET = "incipient_failure"

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    [data-testid="metric-container"] {
        background: #1E2130; border-radius: 8px;
        padding: 12px 16px; border-left: 3px solid #4C9BE8;
    }
</style>""", unsafe_allow_html=True)

METRIC_EXPL = {
    "Accuracy":  "Out of every 100 motor readings, the model classifies this many correctly.",
    "Precision": "When the model flags a motor as failing, how often it's actually in failure state.",
    "Recall":    "Out of all truly failing motors, how many the model catches. Missed failures are the costliest error.",
    "F1 Score":  "Balances precision and recall — critical when false negatives (missed failures) are expensive.",
    "AUC-ROC":   "Separates failing from normal motors across all thresholds. 1.0 = perfect.",
}
ACTION_MAP = {
    "vib_rms_mms":          "Schedule vibration analysis — RMS above 3.5 mm/s warrants immediate bearing inspection",
    "vib_peak_to_peak_mms": "Check for imbalance or looseness — peak-to-peak elevation indicates impact events",
    "bearing_temp_c":       "Inspect lubrication and bearing condition — temp above 75 C indicates friction buildup",
    "motor_current_a":      "Check mechanical load and electrical connections — overcurrent signals increased resistance",
    "dominant_freq_hz":     "Perform spectrum analysis — frequency shift below 27 Hz may indicate misalignment",
    "load_pct":             "Review load assignment — sustained load above 88% accelerates mechanical wear",
}


@st.cache_data
def load_data():
    try:
        return pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        return pd.read_csv("https://raw.githubusercontent.com/LozanoLsa/03-KNN-Motor-Failure/main/motor_failure_prediction_data.csv")


@st.cache_resource
def train_model(df):
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)
    pipe = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())])
    grid = GridSearchCV(pipe,
                        {"knn__n_neighbors": [3,5,7,9,11,15],
                         "knn__weights": ["uniform","distance"], "knn__p": [1,2]},
                        cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    yp, ypr = best.predict(X_test), best.predict_proba(X_test)[:,1]
    metrics = {"Accuracy": accuracy_score(y_test,yp), "Precision": precision_score(y_test,yp),
               "Recall": recall_score(y_test,yp), "F1 Score": f1_score(y_test,yp),
               "AUC-ROC": roc_auc_score(y_test,ypr)}
    pi = permutation_importance(best, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
    imp_df = pd.DataFrame({"Feature": FEATURES, "Importance": pi.importances_mean,
                           "Std": pi.importances_std}).sort_values("Importance", ascending=False)
    return best, grid.best_params_, X_train, X_test, y_train, y_test, yp, ypr, metrics, imp_df


df = load_data()
best_knn, best_params, X_train, X_test, y_train, y_test, y_pred, y_prob, metrics, imp_df = train_model(df)
failure_rate = df[TARGET].mean()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Motor Failure Predictor")
    st.markdown("KNN model trained on 1,500 motor condition records. Estimates incipient failure probability from 6 real-time sensor signals.")
    st.divider()
    st.markdown("### 📡 Sensor Readings")
    st.markdown("**Vibration**")
    vib_rms  = st.slider("Vibration RMS (mm/s)", 0.5, 7.0, 2.2, 0.1)
    vib_pp   = st.slider("Vibration Peak-to-Peak (mm/s)", 1.0, 14.0, 5.0, 0.1)
    st.markdown("**Thermal & Electrical**")
    bear_t   = st.slider("Bearing Temperature (°C)", 40.0, 100.0, 60.0, 0.5)
    curr_a   = st.slider("Motor Current (A)", 18.0, 45.0, 28.0, 0.5)
    st.markdown("**Spectral & Load**")
    freq_hz  = st.slider("Dominant Frequency (Hz)", 20.0, 40.0, 30.0, 0.5)
    load_p   = st.slider("Load (%)", 30.0, 100.0, 70.0, 1.0)
    st.divider()
    st.caption(f"Best k={best_params.get('knn__n_neighbors')}, "
               f"p={best_params.get('knn__p')}, weights={best_params.get('knn__weights')}")
    st.caption("LozanoLsa · Operational Excellence · ML Portfolio · 2026")


def predict_s(vr, vp, bt, mc, fd, lp):
    row = pd.DataFrame([{"vib_rms_mms": vr, "vib_peak_to_peak_mms": vp, "bearing_temp_c": bt,
                          "motor_current_a": mc, "dominant_freq_hz": fd, "load_pct": lp}])
    p = best_knn.predict_proba(row)[0, 1]
    return p, int(p >= 0.5)


pred_prob, pred_class = predict_s(vib_rms, vib_pp, bear_t, curr_a, freq_hz, load_p)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Explorer", "📈 Model Performance",
    "🎯 Scenario Simulator", "🔍 Risk Drivers", "📋 Action Plan"])

# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Dataset Overview")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Records",    f"{len(df):,}")
    k2.metric("Incipient Failure",f"{df[TARGET].sum():,}")
    k3.metric("Normal",           f"{(df[TARGET]==0).sum():,}")
    k4.metric("Failure Rate",     f"{failure_rate:.1%}")
    st.divider()
    c1, c2 = st.columns([1,2])
    with c1:
        fig_pie = go.Figure(go.Pie(labels=["Normal","Incipient Failure"],
                                   values=[(df[TARGET]==0).sum(), df[TARGET].sum()],
                                   marker_colors=["#4C9BE8","#E8574C"], hole=0.45, textinfo="percent+label"))
        fig_pie.update_layout(title="Class Distribution", showlegend=False, height=285,
                              paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pie, use_container_width=True)
    with c2:
        sc_y = st.selectbox("Scatter Y:", FEATURES, index=2)
        fig_sc = px.scatter(df, x="vib_rms_mms", y=sc_y,
                            color=df[TARGET].map({0:"Normal",1:"Incipient Failure"}),
                            color_discrete_map={"Normal":"#4C9BE8","Incipient Failure":"#E8574C"},
                            opacity=0.4, title=f"Vibration RMS vs {sc_y.replace('_',' ').title()}")
        fig_sc.update_layout(height=285, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_sc, use_container_width=True)
    st.divider()
    ns = st.selectbox("Feature distribution:", FEATURES)
    c3, c4 = st.columns(2)
    with c3:
        fh = px.histogram(df, x=ns, color=TARGET, color_discrete_map={0:"#4C9BE8",1:"#E8574C"},
                          barmode="overlay", opacity=0.7, title=f"Distribution: {ns.replace('_',' ').title()}")
        fh.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fh, use_container_width=True)
    with c4:
        fb = px.box(df, x=df[TARGET].map({0:"Normal",1:"Incipient Failure"}), y=ns,
                    color=df[TARGET].map({0:"Normal",1:"Incipient Failure"}),
                    color_discrete_map={"Normal":"#4C9BE8","Incipient Failure":"#E8574C"},
                    labels={"x":"","y":ns.replace("_"," ").title()},
                    title=f"By Class: {ns.replace('_',' ').title()}")
        fb.update_layout(showlegend=False, height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fb, use_container_width=True)
    fcorr = px.imshow(df[FEATURES+[TARGET]].corr(), color_continuous_scale="RdBu_r",
                      zmin=-1, zmax=1, text_auto=".2f", title="Feature Correlation Matrix", aspect="auto")
    fcorr.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fcorr, use_container_width=True)

# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    st.info(f"Tested on {len(X_test)} unseen motor records — correctly classified {metrics['Accuracy']:.0%}. "
            "In predictive maintenance, **Recall** is the critical metric: a missed failure is far costlier than a false alarm.")
    mc = st.columns(5)
    for col, (n, v) in zip(mc, metrics.items()):
        col.metric(n, f"{v:.1%}")
        col.caption(METRIC_EXPL[n])
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        cm = confusion_matrix(y_test, y_pred)
        fcm = px.imshow(cm, text_auto=True, x=["Pred: Normal","Pred: Failure"],
                        y=["True: Normal","True: Failure"], color_continuous_scale="Blues",
                        title="Confusion Matrix")
        fcm.update_layout(height=360, coloraxis_showscale=False, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fcm, use_container_width=True)
        st.caption("Rows = actual · Columns = predicted · Diagonal = correct")
    with c2:
        fp = go.Figure()
        fp.add_trace(go.Histogram(x=y_prob[y_test==0], name="Actual: Normal",
                                  marker_color="#4C9BE8", opacity=0.7, nbinsx=30))
        fp.add_trace(go.Histogram(x=y_prob[y_test==1], name="Actual: Failure",
                                  marker_color="#E8574C", opacity=0.7, nbinsx=30))
        fp.add_vline(x=0.5, line_dash="dash", line_color="white")
        fp.update_layout(barmode="overlay", title="Predicted Probability by True Class",
                         xaxis_title="P(Incipient Failure)", height=360,
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fp, use_container_width=True)
    rep = pd.DataFrame(classification_report(y_test, y_pred,
                                             target_names=["Normal","Incipient Failure"],
                                             output_dict=True)).T.round(3)
    st.dataframe(rep.style.background_gradient(cmap="Blues", subset=["precision","recall","f1-score"]),
                 use_container_width=True)
    st.caption(f"KNN Pipeline | k={best_params.get('knn__n_neighbors')}, p={best_params.get('knn__p')}, "
               f"weights={best_params.get('knn__weights')} | 70/30 stratified | random_state={RANDOM_STATE}")

# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Motor Condition Simulator")
    left, right = st.columns([1, 2])
    with left:
        fg = go.Figure(go.Indicator(
            mode="gauge+number", value=pred_prob*100,
            number={"suffix":"%","font":{"size":42}},
            title={"text":"Estimated Failure Probability","font":{"size":14}},
            gauge={"axis":{"range":[0,100]},
                   "bar":{"color":"#E8574C" if pred_class==1 else "#4C9BE8"},
                   "steps":[{"range":[0,30],"color":"#1a2e1a"},
                             {"range":[30,60],"color":"#2e2a1a"},
                             {"range":[60,100],"color":"#2e1a1a"}],
                   "threshold":{"line":{"color":"white","width":3},"thickness":0.75,"value":50}}))
        fg.update_layout(height=310, paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=60,b=20,l=20,r=20))
        st.plotly_chart(fg, use_container_width=True)
        if pred_class == 1:
            st.error("**INCIPIENT FAILURE — SCHEDULE INSPECTION**")
        else:
            st.success("**NORMAL OPERATION — CONTINUE MONITORING**")
        st.caption(f"Fleet avg: {failure_rate:.1%}  ·  This motor: {pred_prob:.1%}  ·  Δ {pred_prob-failure_rate:+.1%}")
    with right:
        sc_obj = best_knn.named_steps["scaler"]
        X_curr_sc = sc_obj.transform([[vib_rms, vib_pp, bear_t, curr_a, freq_hz, load_p]])
        normal_mean = sc_obj.transform([df[df[TARGET]==0][FEATURES].mean().values])[0]
        contrib = X_curr_sc[0] - normal_mean
        cd = pd.DataFrame({"Feature": FEATURES, "Contribution": contrib})
        cd = cd.reindex(cd["Contribution"].abs().sort_values(ascending=False).index).head(5)
        fc = go.Figure(go.Bar(x=cd["Contribution"],
                              y=cd["Feature"].str.replace("_"," ").str.title(),
                              orientation="h",
                              marker_color=["#E8574C" if v>0 else "#4C9BE8" for v in cd["Contribution"]]))
        fc.update_layout(title="Deviation from Normal-Class Profile (Top 5)",
                         xaxis_title="Std. deviations above normal mean", height=290,
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         margin=dict(t=50,b=40,l=10,r=10))
        st.plotly_chart(fc, use_container_width=True)
        if pred_prob >= 0.70:
            st.error("**Priority: HIGH** · Schedule inspection before end of shift")
        elif pred_prob >= 0.35:
            st.warning("**Priority: MEDIUM** · Increase monitoring frequency — next 48h")
        else:
            st.success("**Priority: LOW** · Standard monitoring schedule")
    st.divider()
    st.markdown("### Scenario Comparison")
    bp = predict_s(2.0, 5.0, 60.0, 28.0, 30.0, 70.0)[0]
    wp = predict_s(4.8, 9.8, 88.0, 37.0, 23.0, 92.0)[0]
    cdf = pd.DataFrame([
        {"Scenario":"Best case (normal operation)","P(Failure)":f"{bp:.1%}",
         "Class":"Normal" if bp<0.5 else "Failure","Delta vs current":f"{bp-pred_prob:+.1%}"},
        {"Scenario":"Your current readings","P(Failure)":f"{pred_prob:.1%}",
         "Class":"Normal" if pred_class==0 else "Failure","Delta vs current":"—"},
        {"Scenario":"Worst case (incipient failure)","P(Failure)":f"{wp:.1%}",
         "Class":"Normal" if wp<0.5 else "Failure","Delta vs current":f"{wp-pred_prob:+.1%}"},
    ])
    st.dataframe(cdf, use_container_width=True, hide_index=True)

# ── TAB 4 ─────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("What Variables Drive Failure Risk?")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Permutation Importance**")
        st.caption("Accuracy drop when each feature is randomly shuffled (10 repeats).")
        is_ = imp_df.sort_values("Importance", ascending=True)
        fi = go.Figure(go.Bar(x=is_["Importance"],
                              y=is_["Feature"].str.replace("_"," ").str.title(),
                              orientation="h",
                              error_x=dict(type="data", array=is_["Std"].values,
                                           color="white", thickness=1.5, width=4),
                              marker_color=["#E8574C" if v>0 else "#888" for v in is_["Importance"]]))
        fi.update_layout(height=380, xaxis_title="Mean accuracy drop",
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fi, use_container_width=True)
    with c2:
        st.markdown("**Class-Conditional Feature Means**")
        mn = df[df[TARGET]==0][FEATURES].mean()
        mf = df[df[TARGET]==1][FEATURES].mean()
        mdf = pd.DataFrame({"Feature":FEATURES,"Normal":mn.values,"Failure":mf.values})
        fmean = go.Figure()
        fmean.add_trace(go.Bar(name="Normal", x=mdf["Feature"].str.replace("_"," ").str.title(),
                               y=mdf["Normal"], marker_color="#4C9BE8"))
        fmean.add_trace(go.Bar(name="Incipient Failure",
                               x=mdf["Feature"].str.replace("_"," ").str.title(),
                               y=mdf["Failure"], marker_color="#E8574C"))
        fmean.update_layout(barmode="group", height=380, xaxis_tickangle=-25,
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fmean, use_container_width=True)
    st.divider()
    st.markdown("**2D Decision Boundary — Vibration RMS vs Bearing Temperature**")
    sc_obj = best_knn.named_steps["scaler"]
    knn_obj = best_knn.named_steps["knn"]
    xi, yi = FEATURES.index("vib_rms_mms"), FEATURES.index("bearing_temp_c")
    xv = np.linspace(df["vib_rms_mms"].min()-0.5, df["vib_rms_mms"].max()+0.5, 120)
    yv = np.linspace(df["bearing_temp_c"].min()-2, df["bearing_temp_c"].max()+2, 120)
    xx, yy = np.meshgrid(xv, yv)
    gm = np.tile(sc_obj.mean_, (xx.ravel().shape[0], 1))
    gm[:, xi] = xx.ravel(); gm[:, yi] = yy.ravel()
    pg = knn_obj.predict_proba(sc_obj.transform(gm))[:,1].reshape(xx.shape)
    fmap = go.Figure()
    fmap.add_trace(go.Contour(x=xv, y=yv, z=pg, colorscale="RdYlGn_r", opacity=0.8,
                              contours=dict(showlabels=False),
                              colorbar=dict(title="P(Failure)")))
    for cls, color, name in [(0,"#4C9BE8","Normal"),(1,"#E8574C","Incipient Failure")]:
        mask = df[TARGET]==cls
        fmap.add_trace(go.Scatter(x=df.loc[mask,"vib_rms_mms"], y=df.loc[mask,"bearing_temp_c"],
                                  mode="markers", name=name,
                                  marker=dict(color=color, size=4, opacity=0.4)))
    fmap.update_layout(xaxis_title="Vibration RMS (mm/s)", yaxis_title="Bearing Temperature (°C)",
                       height=480, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fmap, use_container_width=True)
    st.caption("Non-linear KNN boundary — confirms why a distance-based model fits this problem better than a linear classifier.")

# ── TAB 5 ─────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Operational Action Plan")
    if pred_prob >= 0.70:
        pl, hz, ac = "🔴 HIGH", "Before end of shift", "Take motor offline — inspect bearings, lubrication, and electrical connections"
    elif pred_prob >= 0.35:
        pl, hz, ac = "🟡 MEDIUM", "Within 48 hours", "Increase sensor monitoring frequency — flag for next scheduled maintenance window"
    else:
        pl, hz, ac = "🟢 LOW", "Standard schedule", "Normal operation — maintain standard monitoring interval"
    st.markdown(f"""
| Field | Value |
|---|---|
| **Priority** | {pl} |
| **Estimated failure probability** | {pred_prob:.1%} |
| **Suggested action** | {ac} |
| **Recommended horizon** | {hz} |
| **Suggested owner** | Maintenance Engineer / Plant Operator |
""")
    st.divider()
    st.markdown("### 🔧 Active Risk Factors")
    rfs = []
    if vib_rms > 3.5:  rfs.append("vib_rms_mms")
    if vib_pp > 7.5:   rfs.append("vib_peak_to_peak_mms")
    if bear_t > 75:    rfs.append("bearing_temp_c")
    if curr_a > 33:    rfs.append("motor_current_a")
    if freq_hz < 27:   rfs.append("dominant_freq_hz")
    if load_p > 88:    rfs.append("load_pct")
    if rfs:
        for f in rfs[:4]:
            if f in ACTION_MAP:
                with st.expander(f"▲ {f.replace('_',' ').title()} — above normal threshold"):
                    st.write(ACTION_MAP[f])
    else:
        st.success("All sensor readings are within normal operating ranges.")
    st.divider()
    st.caption("_This tool supports maintenance decisions — it does not replace certified condition monitoring procedures or safety protocols._")
