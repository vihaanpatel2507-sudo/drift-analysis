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
| **1 (Train Era)** | **0.178 $\pm$ 0.026** | **0.141** | **0.885** | 855 / 950,485 |
| **2** | 0.083 $\pm$ 0.015 | 0.051 | 0.733 | 936 / 950,485 |
| **3** | 0.077 $\pm$ 0.010 | 0.048 | 0.690 | 974 / 950,485 |
| **4** | 0.104 $\pm$ 0.008 | 0.068 | 0.706 | 991 / 950,485 |
| **5** | 0.101 $\pm$ 0.011 | 0.064 | 0.697 | 1,016 / 950,485 |
| **6** | 0.074 $\pm$ 0.007 | 0.043 | 0.693 | 970 / 950,485 |
| **7** | 0.064 $\pm$ 0.005 | 0.037 | 0.696 | 1,022 / 950,485 |
| **8** | 0.053 $\pm$ 0.008 | 0.033 | 0.700 | 966 / 950,485 |
| **9** | 0.076 $\pm$ 0.014 | 0.045 | 0.720 | 1,040 / 950,485 |
| **10** | 0.107 $\pm$ 0.011 | 0.067 | 0.712 | 1,103 / 950,487 |

**Takeaway:** As soon as future data arrives, the model suffers immediate concept drift: **$F_1$ falls from 0.178 in the training era to 0.053–0.107 across chunks 2–10** (a 40–70% relative drop, ~54% against the future mean) and PR-AUC collapses from 0.141 to 0.033–0.068. ROC-AUC drops from 0.885 to 0.690–0.733, so the degradation is a genuine loss of ranking power, not merely a miscalibrated threshold.

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
| **Feature PSI (Numeric)** | ❌ No | $PSI > 0.20$ | **0 / 9** ($PSI \le 0.036$) | Input distributions appear stable |
| **Feature PSI (Categorical)**| ❌ No | $PSI > 0.20$ | **0 / 9** ($PSI \le 0.0038$) | Marginal category rates unchanged |
| **Output Probabilities (KS)**| ❌ No | $\alpha = 0.05$ (per-chunk $crit \approx 0.002$) | **7 / 9** Chunks | Sensitive hypothesis test on $N \approx 10^6$ |
| **Output Probabilities (PSI)**| ❌ No | $PSI > 0.20$ | **0 / 9** ($PSI \le 0.00019$) | Shift magnitude in scores is subtle |
| **Domain Classifier (XGB)** | ❌ No | $\text{AUC} > 0.55$ | **9 / 9 (All Chunks)** | **Mean AUC = 0.584 – 0.612** |

**Takeaway:** Standard univariate feature PSI is blind to AML drift because aggregate transactional marginals stay constant. However, the **Domain Classifier on behavioral features successfully detects drift across all 9 future chunks without requiring a single label**.

**A caveat worth stating in the paper:** these five detectors do not agree. KS fires on 7/9 chunks while output PSI fires on 0/9 — at $N \approx 10^6$ the KS critical value falls to $\approx 0.002$, so it rejects on differences that are statistically detectable but practically negligible ($PSI \le 0.00019$). KS is therefore a *sensitivity* test, not an *effect-size* test, and the two together bracket the honest answer.

---

### Phase E — Label Scarcity & Detection Reliability
* **Script:** `phaseE_label_scarcity.py` $\rightarrow$ Outputs: `results_phaseE/`
* **Methodology:**
  * Evaluated performance across label availability budgets: **100%, 50%, 10%, 5%, 1%, 0.1%** (3 random seeds per budget).
  * **Matched No-Drift Control:** The identical label budget sweep was evaluated on an unseen held-out slice of **Chunk 1**, where no drift exists.
  * A budget is only reported as a comparison when **all 3 control seeds produced a usable score**. Where they did not, the ratio is left blank rather than being computed against a starved or zero-scoring control.

#### Future Stream vs. No-Drift Control:

| Label Availability | Mean $F_1$ (Drifted Future Chunks 2–10) | No-Drift Control $F_1$ (Chunk 1 Holdout) | Future / Control | Comparable? |
|:---:|:---:|:---:|:---:|:---:|
| **100%** | **0.092** | **0.170** | **0.54** | ✅ 3/3 seeds |
| **50%** | **0.094** | **0.184** | **0.51** | ✅ 3/3 seeds |
| **10%** | **0.088** | **0.166** | **0.53** | ✅ 3/3 seeds |
| **5%** | **0.075** | **0.131** | **0.57** | ✅ 3/3 seeds |
| **1%** | 0.114 (3/3 seeds) | — (only 1/3 control seeds scored) | — | ❌ not comparable |
| **0.1%** | *Insufficient Positives* | *Insufficient Positives* | — | ❌ not measurable |

**Takeaways:**
1. **Decay is Genuine Drift:** At every budget where the measurement is trustworthy (100% down to 5%), the drifted future stream scores only **51–57%** of the no-drift control (0.075–0.094 vs 0.131–0.184). The drop is not a small-sample illusion.
2. **Label scarcity is cheap; drift is expensive:** the drifted stream's $F_1$ is essentially flat from 100% down to 5% labels (0.092 → 0.075, overlapping 95% CIs), so losing labels costs almost nothing in *measuring* decay. Running out of labels mainly blocks *repairing* it.
3. **Measurement breaks below 5%, and we say so rather than improvise a number.** At 1% availability only ~10 positives are visible per chunk, and 2 of the 3 control runs found too few positives to score at all — so the arms cannot be compared. At 0.1% (≈1 positive per chunk) nothing is measurable in either arm. An earlier aggregation averaged over the surviving runs only, which made the starved 1% control render as a clean `0.0000` and the ratio as a spurious `0.98`; `control_n_runs_valid` in the summary CSV now records how many runs actually contributed, and the plot shades this region as unmeasurable rather than drawing a curve to zero.

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
| `no_retrain` (Frozen Baseline) | 0.0916 | 0.1180 | 0.0555 | 0.0528 |
| `periodic_full` (All History) | **0.1144 (+28% gain)** | **0.1551** | **0.0719** | **0.0646** |
| `periodic_sliding` (Window = 2) | 0.1065 (+19% gain) | 0.1346 | 0.0661 | 0.0638 |
| `periodic_weighted` (10x Boost) | 0.0303 (Severely Degraded) | 0.0232 | 0.0656 | 0.2274 |

#### Class Weight Multiplier Sweep ($w \in \{1, 10, 50, 100\}$), mean over chunks 2–10:

| $w$ | Mean $F_1$ | Mean Precision | Mean Recall | Beats the other $w$ on | Beats frozen baseline on |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **0.1004** | **0.5470** | 0.0579 | **9 / 9 chunks** | 7 / 9 chunks |
| 10 | 0.0212 | 0.0112 | 0.2116 | 0 / 9 | 0 / 9 |
| 50 | 0.0076 | 0.0038 | 0.4134 | 0 / 9 | 0 / 9 |
| 100 | 0.0032 | 0.0016 | 0.5588 | 0 / 9 | 0 / 9 |

* **$w=1$ (natural imbalance ratio)** achieves the highest $F_1$ of the four multipliers on **every** chunk. Note that this is a comparison *among the $w$ variants*; $w=1$ still trails the frozen Phase C baseline on chunks 2, 3 and 7, so plain retraining at $w=1$ beats re-weighting but does not beat retraining with the full `periodic_full` protocol (which re-tunes the decision threshold and pools every accumulated chunk).
* **$w > 1$** buys recall (0.058 → 0.559) and pays for it in precision (0.547 → 0.0016), so $F_1$ falls monotonically. Up-weighting a class that is 1-in-1,000 does not make the model better at ranking; it moves the decision threshold far enough that precision collapses.

**Takeaway:** Periodic full-history retraining delivers a solid **+28% relative $F_1$ recovery** out-of-sample, while sliding-window retraining offers a scalable alternative (+19%). However, retraining cannot fully restore training-era efficacy (0.178), confirming that concept drift in AML alters the underlying conditional distribution $P(Y|X)$. Note also that the $w>1$ runs still score *better* PR-AUC than the frozen baseline — the damage is confined to the thresholded decision, not the underlying score quality.

---

## 4. How to Run the Experiments

### Prerequisites
Install dependencies (**Python 3.12+** — `numpy` and `xgboost` both require it):
```bash
pip install -r requirements.txt
```
*Metrics (ROC-AUC, PR-AUC, precision/recall/F1) come from `scikit-learn`; the two-sample KS test comes from `scipy`; PSI is a custom implementation in `common.py`. The label encoder and the stratified split are hand-written to keep the pipeline deterministic and dependency-light.*

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

# 6. Generate publication comparison charts (saves to results_maincomparsion/)
python generate_hero_chart.py
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
├── generate_hero_chart.py         # 4-panel publication hero chart generator
├── check_chunk_balance.py         # Sanity check: class balance per temporal chunk
├── inspect_chunk.py               # Sanity check: schema/values of a single chunk
├── run_all.sh                     # End-to-end pipeline runner
├── RESULTS_EXPLAINED.md           # Comprehensive findings & paper writeup guide
├── results_phaseC/                # Baseline decay metrics & CSV tables
├── results_phaseD/                # Drift detector outputs (PSI, KS, Domain Classifier)
├── results_phaseE/                # Label scarcity vs no-drift control evaluations
├── results_phaseF/                # Adaptation results & class weight sweeps
└── results_maincomparsion/        # Publication-ready retraining comparison figures (PNG/PDF)
├── run_all.sh                     # Full B->F re-run: backs up results, then runs every phase in order
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
