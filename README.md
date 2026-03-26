# ⚙️ Motor Failure Prediction using KNN

## Project Overview
This project was a real project used for a **practical machine learning exercise inspired by real-world predictive maintenance scenarios** in industrial environments.

Let’s be clear from the start 🙂  
The goal here is **not** to claim that a single model will prevent all motor failures.  
The goal is to **understand**, **visualize**, and **reason** about how early-stage failures emerge when vibration, temperature, electrical, and load conditions start drifting.

A **K-Nearest Neighbors (KNN)** classifier is used here as a way to explore **non-linear, localized failure patterns** that are often missed by linear models.

No magic.  
Just patterns in the neighborhood.

---

## Problem Statement
Electric motors rarely fail suddenly without warning.

In practice, failures tend to develop gradually through:
- Increasing vibration levels  
- Rising bearing temperatures  
- Electrical overload  
- Changes in operating frequency and load  

Individually, these signals may look harmless.  
Together, they often indicate **incipient failure conditions**.

This project treats motor failure not as a binary surprise, but as a **condition that emerges in specific regions of the operating space**.

If you’ve worked with rotating equipment, this probably sounds familiar.

---

## Objective 🎯
The main objectives of this project are to:

- Build a **binary classification model** to identify incipient motor failures.
- Use **KNN** to capture **localized, non-linear relationships** between condition signals.
- Compare KNN against simpler baselines (Logistic Regression and Naive Bayes).
- Validate model differences using **statistical testing**, not just metrics.

This project is intentionally designed as a **learning and reasoning exercise**. that was the reason the data is limited to 1500 rows.

---

## Dataset Description 📊
The dataset represents a **synthetic but physically realistic motor condition monitoring environment**.

Each row corresponds to a motor observation under a specific operating condition.

### Features (X)
The input variables include:

- **vib_rms**: Root Mean Square vibration level  
- **vib_pp**: Peak-to-peak vibration  
- **temp_bearing**: Bearing temperature  
- **current**: Electrical current draw  
- **freq_dom**: Dominant vibration frequency  
- **load_pct**: Motor load percentage  

### Target Variable (Y)
- **incipient_failure**  
  - **0 = Normal operation**  
  - **1 = Incipient failure**

---

## Data Origin (Real-World Perspective)
In real industrial environments, this type of dataset **does not come from a single source**.

Each variable typically originates from a different system:

- **Vibration signals (vib_rms, vib_pp, freq_dom)**  
  → Online condition monitoring systems, accelerometers, FFT analyzers, vibration routes.

- **Temperature (temp_bearing)**  
  → Embedded sensors, infrared inspections, maintenance rounds.

- **Electrical current (current)**  
  → Motor control centers (MCC), power analyzers, SCADA systems.

- **Load percentage (load_pct)**  
  → PLC signals, production counters, torque estimates.

- **Failure labels (target)**  
  → Maintenance logs, work orders, inspection findings, root cause analysis records.

> In practice, building this dataset requires **integrating information across maintenance, operations, and automation systems**.  
> Understanding the process context is just as important as training the model 🙂.

---

## Modeling Approach 🧠
A **K-Nearest Neighbors (KNN)** classifier was selected because:

- It makes **no assumptions** about linearity.
- It detects failure risk based on **similar historical operating conditions**.
- It naturally captures **local patterns** in the feature space.

Key steps include:
- Feature standardization for distance-based learning
- Hyperparameter tuning (neighbors, distance metric, weighting)
- Decision boundary visualization in 2D projections
- Probability-based predictions for risk assessment

To provide context, KNN is compared against:
- Logistic Regression  
- Naive Bayes  

All models are evaluated using the same train/test split.

---

## Why this case fits KNN particularly well ✅
- Failure patterns are **non-linear** and region-specific.
- Motors fail when combinations of conditions align, not when a single variable crosses a threshold.
- KNN answers a very natural operational question:
  > “Given motors that behaved like this before… what usually happened?”

This makes it a strong candidate for **condition-based risk screening**.

---

## Key Results 📈
Standard classification metrics are reported, but they are **not the headline**.

What really matters is that the model allows us to:
- Identify **high-risk operating regions**
- Visualize **failure-prone neighborhoods**
- Quantify how risk increases when vibration and temperature drift together

KNN outperforms linear baselines in this scenario, and **statistical testing (McNemar)** confirms that the improvement is not due to chance.

Accuracy matters.  
Understanding patterns matters more.

---

## Simulation & Scenarios
A simple **scenario simulator** is included in the notebook.

It allows you to:
- Define hypothetical motor conditions
- Estimate the probability of incipient failure
- Ask “what if?” questions without pretending to predict exact failure times

This is where the model becomes a **decision-support conversation tool**, not just code.

---

## Project Outputs 📂
This repository contains:
- A dataset ('.csv') with motor condition variables
- A Jupyter Notebook with full analysis, visualization, and validation
- A PDF summary with results for non-technical audiences

---

## Next Steps 🚀
If this were taken further, possible extensions include:
- Time-based degradation tracking
- Drift detection across seasons or production regimes
- Comparison with tree-based or ensemble models
- Integration into maintenance planning workflows

But that’s a different project.

---

—
Not magic. Just neighbors.  
**Where data meets reliability engineering**  
LozanoLsa  
Regards from MX