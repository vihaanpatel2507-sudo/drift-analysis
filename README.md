# Concept Drift Detection and Adaptation in Anti-Money-Laundering (AML)

A comprehensive empirical study investigating **concept drift**, **unsupervised drift detection**, **label scarcity constraints**, and **out-of-sample retraining strategies** on large-scale banking data.

---

## 1. Project Overview

Anti-Money-Laundering (AML) machine learning systems deployed in financial institutions operate under challenging conditions:
1. **Extreme Class Imbalance:** Illicit transactions account for approximately **0.1%** of all transactions (roughly 1 in 1,000).
2. **Dynamic Adversaries (Concept Drift):** Criminal networks constantly adapt laundering techniques, rendering static models obsolete over time.
3. **Severe Verification Latency & Label Scarcity:** Ground-truth labels (Suspicious Activity Reports filed after law enforcement investigation) take weeks or months to confirm.

This project investigates the full lifecycle of AML model decay and management:
* **Decay (Phase C):** How rapidly does an AML classifier degrade when deployed on future transactions?
* **Detection (Phase D):** Can unsupervised detectors flag drift without ground-truth labels?
* **Measurement Under Scarcity (Phase E):** Is observed decay genuine drift or an artifact of small label budgets? How resilient are detectors under limited labeling?
* **Mitigation (Phase F):** Does periodic retraining recover performance out-of-sample, and how do sliding windows and class weights impact adaptation?

**Dataset:** SAML-D (Synthetic Anti-Money Laundering Dataset) containing **~9.5 million transactions** divided chronologically into 10 sequential chunks of ~950,485 rows each.

---

## 2. Experimental Architecture

```
[Chunk 1]  ──>  [Chunk 2]  ──>  [Chunk 3]  ──> ... ──>  [Chunk 10]
 Training Era        Future Stream (Evaluated Chronologically & Out-of-Sample)
```

* **Chunk 1 (Training Era):** Used exclusively for baseline model training and feature encoder fitting.
* **Chunks 2–10 (Streaming Future):** Evaluated strictly sequentially to simulate production deployment over time.

---

## 3. Phase-by-Phase Experimental Results

### Phase B — Preprocessing & Temporal Chunking
* **Script:** `phaseB_chunking.py` $\rightarrow$ Outputs: `chunks/chunk_01.csv` ... `chunk_10.csv`
* **Methodology:**
  * Strict timestamp parsing (`YYYY-MM-DD HH:MM:SS`) preserving authentic chronology.
  * Explicit exclusion of direct leakage features (`Laundering_type`) and identifiers (`Sender_account`, `Receiver_account`).
  * **Leakage-Free Imputation:** Missing value statistics (numeric medians, categorical modes) are computed **strictly on Chunk 1** and applied out-of-sample to Chunks 2–10.

---

### Phase C — Baseline Model & Performance Decay
* **Script:** `phaseC_baseline_model.py` $\rightarrow$ Outputs: `results_phaseC/`
* **Setup:**
  * Model: XGBoost Classifier (`n_estimators=300, max_depth=6, lr=0.05`, `scale_pos_weight` set to negative/positive ratio).
  * Decision threshold tuned on held-out stratified validation slice (20% of Chunk 1) to maximize $F_1$.
  * Evaluated across **3 independent random seeds** (42, 7, 123) with mean and 95% confidence intervals.
  * The trained baseline is **frozen** and evaluated across Chunks 1 to 10 without retraining.

#### Empirical Results:

| Chunk | $F_1$ (Mean $\pm$ Std) | PR-AUC / AP | ROC-AUC | Positives / Total Rows |
|:---:|:---:|:---:|:---:|:---:|
| **1 (Train Era)** | **0.199 $\pm$ 0.018** | **0.154 $\pm$ 0.011** | **0.891** | 855 / 950,485 |
| **2** | 0.083 $\pm$ 0.014 | 0.054 $\pm$ 0.006 | 0.716 | 936 / 950,485 |
| **3** | 0.075 $\pm$ 0.005 | 0.051 $\pm$ 0.006 | 0.672 | 974 / 950,485 |
| **4** | 0.100 $\pm$ 0.014 | 0.069 $\pm$ 0.002 | 0.691 | 991 / 950,485 |
| **5** | 0.096 $\pm$ 0.008 | 0.067 $\pm$ 0.006 | 0.683 | 1,016 / 950,485 |
| **6** | 0.072 $\pm$ 0.008 | 0.043 $\pm$ 0.007 | 0.672 | 970 / 950,485 |
| **7** | 0.065 $\pm$ 0.007 | 0.040 $\pm$ 0.004 | 0.683 | 1,022 / 950,485 |
| **8** | 0.064 $\pm$ 0.011 | 0.038 $\pm$ 0.006 | 0.685 | 966 / 950,485 |
| **9** | 0.072 $\pm$ 0.008 | 0.048 $\pm$ 0.007 | 0.700 | 1,040 / 950,485 |
| **10** | 0.105 $\pm$ 0.007 | 0.071 $\pm$ 0.005 | 0.691 | 1,103 / 950,487 |

**Takeaway:** As soon as future data arrives, the model suffers immediate concept drift: **$F_1$ drops by ~59% (from 0.199 down to 0.064–0.105)** and PR-AUC collapses from 0.154 to ~0.04–0.07.

---

### Phase D — Unsupervised Drift Detection
* **Script:** `phaseD_drift_detectors.py` $\rightarrow$ Outputs: `results_phaseD/`
* **Four Detector Families Tested:**
  1. **Numeric Feature PSI:** Population Stability Index over 6 derived numeric features (`Amount`, `Amount_log`, `hour_of_day`, `day_of_week`, `tx_per_sender`, `amount_per_sender_mean`).
  2. **Categorical Feature PSI:** PSI evaluated across currency, bank location, and payment types.
  3. **Output Distribution Monitoring:** Two-sample Kolmogorov-Smirnov (KS) test and PSI on model predicted probabilities.
  4. **Domain Classifier:** XGBoost discriminator trained to distinguish Reference rows from Current chunk rows (3 random seeds, mean $\pm$ 95% CI).

#### Detection Summary:

| Method | Supervised? | Detection Threshold | Alarms Fired (Chunks 2–10) | Observation |
|---|:---:|:---:|:---:|---|
| **Feature PSI (Numeric)** | ❌ No | $PSI > 0.20$ | **0 / 9** (Max $PSI \le 0.036$) | Input distributions appear stable |
| **Feature PSI (Categorical)**| ❌ No | $PSI > 0.20$ | **0 / 9** (Max $PSI \le 0.004$) | Marginal category rates unchanged |
| **Output Probabilities (KS)**| ❌ No | $\alpha = 0.05$ ($crit \approx 0.00197$) | **6 / 9** Chunks | Sensitive hypothesis test on $N \approx 10^6$ |
| **Output Probabilities (PSI)**| ❌ No | $PSI > 0.20$ | **0 / 9** ($PSI \le 0.00016$) | Shift magnitude in scores is subtle |
| **Domain Classifier (XGB)** | ❌ No | $\text{AUC} > 0.55$ | **9 / 9 (All Chunks)** | **Mean AUC = 0.584 – 0.612** |

**Takeaway:** Standard univariate feature PSI is blind to AML drift because aggregate transactional marginals stay constant. However, the **Domain Classifier on behavioral features successfully detects drift across all 9 future chunks without requiring a single label**.

---

### Phase E — Label Scarcity & Detection Reliability
* **Script:** `phaseE_label_scarcity.py` $\rightarrow$ Outputs: `results_phaseE/`
* **Methodology:**
  * Evaluated performance across label availability budgets: **100%, 50%, 10%, 5%, 1%, 0.1%** (3 random seeds per budget).
  * **Matched No-Drift Control:** The identical label budget sweep was evaluated on an unseen held-out slice of **Chunk 1**, where no drift exists.

#### Future Stream vs. No-Drift Control:

| Label Availability | Mean $F_1$ (Drifted Future Chunks 2–10) | No-Drift Control $F_1$ (Chunk 1 Holdout) | Valid Positives Available |
|:---:|:---:|:---:|:---:|
| **100%** | **0.089** | **0.232** | ~850–1,100 |
| **50%** | **0.088** | **0.223** | ~420–550 |
| **10%** | **0.093** | **0.180** | ~85–110 |
| **5%** | **0.099** | **0.197** | ~45–55 |
| **1%** | **0.127** (high variance) | **0.129** (high variance) | ~9–11 |
| **0.1%** | *Insufficient Positives* | *Insufficient Positives* | $<3$ |

**Takeaways:**
1. **Decay is Genuine Drift:** At every reliable label budget (100% down to 5%), the drifted future stream scores less than half of the no-drift control (~0.09 vs ~0.20–0.23). The drop is not a small-sample illusion.
2. **Measurement Breakdown:** Below 1% label availability, the number of observed positive transactions drops below 10, causing severe metric variance and false alarms.

---

### Phase F — Adaptation & Retraining Strategies
* **Script:** `phaseF_retraining.py` $\rightarrow$ Outputs: `results_phaseF/`
* **Methodology:**
  * **Strict Out-of-Sample Protocol:** A model trained on history up to chunk $c-1$ is evaluated strictly on chunk $c$ (never on its training data).
  * Decision thresholds re-tuned on a 20% validation split of the accumulated retraining pool.
  * Strategies tested:
    1. `no_retrain`: Frozen baseline from Phase C.
    2. `periodic_full`: Retrain on all accumulated labeled history every 2 chunks.
    3. `periodic_sliding`: Retrain on the last 2 chunks only (allows forgetting obsolete patterns).
    4. `periodic_weighted`: Cumulative retraining with 10x positive class up-weighting.
    5. **Weight Multiplier Sweep:** Sweeping $w \in \{1, 10, 50, 100\}$ on top of natural imbalance ratio.

#### Out-of-Sample Adaptation Results:

| Strategy | Mean $F_1$ (Chunks 2–10) | Final Chunk $F_1$ (Chunk 10) | Mean PR-AUC | Mean Recall |
|---|:---:|:---:|:---:|:---:|
| `no_retrain` (Frozen Baseline) | 0.0887 | 0.1120 | 0.0595 | 0.0625 |
| `periodic_full` (All History) | **0.1139 (+28% gain)** | **0.1551** | **0.0723** | **0.0655** |
| `periodic_sliding` (Window = 2) | 0.1059 (+19% gain) | 0.1346 | 0.0666 | 0.0648 |
| `periodic_weighted` (10x Boost) | 0.0298 (Severely Degraded) | 0.0232 | 0.0659 | 0.2283 |

#### Class Weight Multiplier Sweep ($w \in \{1, 10, 50, 100\}$):
* **$w=1$ (Natural Imbalance Ratio):** Consistently achieved the highest $F_1$ on every single chunk.
* **$w > 1$ (Aggressive Up-Weighting):** While recall increased (up to 0.50+), precision collapsed to $<0.01$, driving $F_1$ down to ~0.003–0.020. Artificially boosting positive weights severely damages precision in extreme imbalance.

**Takeaway:** Periodic full-history retraining delivers a solid **+28% relative $F_1$ recovery** out-of-sample, while sliding-window retraining offers a scalable alternative (+19%). However, retraining cannot fully restore training-era efficacy (0.199), confirming that concept drift in AML alters the underlying conditional distribution $P(Y|X)$.

---

## 4. How to Run the Experiments

### Prerequisites
Install dependencies (Python 3.9+):
```bash
pip install pandas numpy xgboost matplotlib
```
*(No `scikit-learn` or `scipy` required; all metrics and splits are implemented in `common.py` to prevent platform-specific DLL conflicts).*

### Execution Pipeline (In Order):
```bash
# 1. Preprocess and split raw dataset into 10 temporal chunks
python phaseB_chunking.py --data_path SAML-D.csv

# 2. Train baseline model on Chunk 1 and evaluate temporal decay
python phaseC_baseline_model.py

# 3. Run unsupervised drift detectors (PSI, KS, Domain Classifier)
python phaseD_drift_detectors.py

# 4. Evaluate drift detection and performance under label scarcity
python phaseE_label_scarcity.py

# 5. Run out-of-sample retraining and weight-sweep adaptation
python phaseF_retraining.py
```

---

## 5. Repository Structure

```
.
├── common.py                      # Shared metrics (Average Precision, ROC-AUC), encoders, splits
├── phaseB_chunking.py             # Leakage-free preprocessing and temporal chunking
├── phaseC_baseline_model.py       # Multi-seed baseline training & decay measurement
├── phaseD_drift_detectors.py      # Unsupervised PSI, KS, and Domain Classifier detectors
├── phaseE_label_scarcity.py       # Core experiment: label scarcity vs no-drift control
├── phaseF_retraining.py           # Out-of-sample retraining (full, sliding, weight sweep)
├── results_explained.txt          # Quick-reference numerical explanation of all phases
├── results_phaseC/                # Decay tables, validation metrics, threshold JSON, plots
├── results_phaseD/                # Detector results, PSI per feature, categorical pivot, plots
├── results_phaseE/                # Label scarcity sweep, control results, summary, plots
└── results_phaseF/                # Adaptation comparisons, weight sweep, summary, plots
```

---

## 6. Citation & Research Context

If utilizing this codebase or experimental setup for research, cite:
```bibtex
@article{aml_concept_drift_2026,
  title={Empirical Analysis of Concept Drift, Unsupervised Detection, and Adaptation Bottlenecks in Anti-Money-Laundering},
  author={Patel, Vihaan},
  year={2026}
}
```
