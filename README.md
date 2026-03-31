# Motor Failures Are Not Random — K-Nearest Neighbors

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LozanoLsa/Motor_Failures_Prediction/blob/main/03_KNN_Motor_Failure_Prediction.ipynb)

> *"An electric motor doesn't fail suddenly. It sends signals for weeks — vibration that creeps upward, bearings that run progressively hotter, current that increases as mechanical resistance builds. These patterns exist in sensor data long before the motor trips a protection relay."*

---

## 🎯 Business Problem

Unplanned motor failure is one of the most expensive events in industrial operations. Not because motors are cheap to replace — but because of what stops with them: production lines, conveyor systems, HVAC, pumps. The downtime cost dwarfs the repair cost, every time.

The shift from **reactive to predictive maintenance** doesn't require exotic technology. It requires reading the signals that are already there. This project applies KNN to classify motor operating states as normal or incipient failure, using six condition-monitoring signals that any SCADA or PLC system already captures.

**This project is free and fully open.** Complete dataset, full notebook, and working simulator included — because predictive maintenance is too important to sit behind a paywall.

---

## 📊 Dataset

- **1,500 motor operating records** from a condition monitoring system
- **Target:** `incipient_failure` (binary) — 0 = normal operation, 1 = incipient failure state
- **Class balance:** 33.3% incipient failure (realistic monitoring scenario)
- **Source:** Simulated sensor data modeling two distinct motor operating regimes

| Signal | Normal Range | Failure Range | Physical Meaning |
|--------|-------------|---------------|-----------------|
| `vib_rms_mms` | ~2.2 mm/s | ~4.3 mm/s | Overall vibration energy |
| `vib_peak_to_peak_mms` | ~5.0 mm/s | ~9.0 mm/s | Impact events, looseness |
| `bearing_temp_c` | ~60°C | ~82°C | Friction buildup |
| `motor_current_a` | ~28A | ~35A | Mechanical resistance |
| `dominant_freq_hz` | ~29.5 Hz | ~24.5 Hz | Misalignment signature |
| `load_pct` | ~70% | ~85% | Operating load context |

The class separation in this dataset is physically grounded — the feature distributions between normal and failure states reflect real ISO 10816 alarm thresholds. This is not arbitrary labeling.

---

## 🤖 Model

**Algorithm:** K-Nearest Neighbors — `sklearn.neighbors.KNeighborsClassifier`

KNN is the right choice here because of a simple physical argument: **if two motors have similar vibration, temperature, and current profiles, they're likely in the same operating state.** Distance in sensor space maps directly to operational similarity. No linear assumption, no parameterized boundary — just proximity in a physically meaningful feature space.

**Why KNN specifically requires StandardScaler:** A feature in Amperes (range 25–40) would dominate distance calculations over one in mm/s (range 1–6) without normalization. Scaling isn't optional — it's what makes the distance metric physically meaningful.

**Preprocessing:** StandardScaler on all features, wrapped in a Pipeline.  
**Tuning:** GridSearchCV over k × weights × distance metric — best: `k=5`, `weights='distance'`, `p=1` (Manhattan).

---

## 📈 Key Results

| Metric | Value |
|--------|-------|
| Accuracy | 89% |
| ROC-AUC | 0.952 |
| Recall (Failure) | 86.4% |
| F1 (Failure) | 0.88 |

**Confusion matrix (450 test records):**

| | Pred: Normal | Pred: Failure |
|---|---|---|
| **Actual: Normal** | 231 (TN) | 20 (FP) |
| **Actual: Failure** | 27 (FN) | 172 (TP) |

Only few failures missed across 450 test records. In predictive maintenance terms, each FN is a motor that continued running toward breakdown without intervention.

---

## 🔍 Feature Importance (Permutation Importance)

| Feature | Accuracy Drop | Role |
|---------|--------------|------|
| `vib_rms_mms` | Δ −8.2% | Primary alarm signal |
| `vib_peak_to_peak_mms` | Δ −6.1% | Impact event detector |
| `bearing_temp_c` | Δ −3.9% | Thermal confirmation |
| `motor_current_a` | Δ −1.4% | Electrical context |
| `dominant_freq_hz` | Δ −0.6% | Misalignment type |
| `load_pct` | Δ −0.1% | Operating context |

Vibration RMS dominates — a near-doubling from 2.2 to 4.3 mm/s represents a shift well above ISO 10816 alarm thresholds for machinery in this class.

---

## 🗂️ Repository Structure

```
Motor_Failures_Prediction/
├── 03_KNN_Motor_Failure_Prediction.ipynb  # Notebook (no outputs)
├── motor_failure_prediction_data.csv      # Complete dataset (1,500 records)
├── app.py                                 # Motor condition simulator
├── README.md
└── requirements.txt
```

> ✅ **This project is completely free.** Full dataset and simulator included. If this helped you, check out the rest of the portfolio at [lozanolsa.gumroad.com](https://lozanolsa.gumroad.com).

---

## 🚀 How to Run

**Option 1 — Google Colab:** Click the badge above.

**Option 2 — Local:**
```bash
pip install -r requirements.txt
jupyter notebook 03_KNN_Motor_Failure_Prediction.ipynb
```

**Option 3 — Run the simulator:**
```bash
python app.py
```

---

## 💡 Key Learnings

1. **Distance = physics** — in condition monitoring, sensor similarity is operational similarity. KNN formalizes an intuition every maintenance engineer already has.
2. **Permutation importance works on any model** — KNN has no coefficients, but shuffling features and measuring accuracy drop gives model-agnostic interpretability that stakeholders can understand.
3. **StandardScaler is not optional for KNN** — without it, the distance metric is dominated by whichever feature has the largest absolute range. Scale before you measure.
4. **GridSearchCV on distance metric matters** — Manhattan distance (p=1) outperformed Euclidean (p=2) here, consistent with sensor data that often contains outliers where L1 is more robust.
5. **High AUC on physical data is expected, not suspicious** — when two operating states are physically distinct (2.2 vs 4.3 mm/s vibration), a good model should find that boundary easily. The value is in the deployment, not the benchmark.

---

## 👤 Author

**Luis Lozano** | Operational Excellence Manager · Master Black Belt · Machine Learning  
GitHub: [LozanoLsa](https://github.com/LozanoLsa) · Gumroad: [lozanolsa.gumroad.com](https://lozanolsa.gumroad.com)

*Turning Operations into Predictive Systems — Clone it. Fork it. Improve it.*
