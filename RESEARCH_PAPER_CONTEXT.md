# Research Paper Context Document
## Concept Drift Detection in Anti-Money Laundering Systems

---

## 1. Research Overview

**Title:** Empirical Analysis of Concept Drift, Unsupervised Detection, and Adaptation Bottlenecks in Anti-Money-Laundering Fraud Detection

**Research Domain:** Machine Learning for Financial Crime Detection

**Dataset:** SAML-D (Synthetic Anti-Money Laundering Dataset)
- Total transactions: 9,504,852
- Temporal chunks: 10 sequential periods (~950,485 rows each)
- Class imbalance: 0.09% positive rate (approximately 855-1,103 positives per chunk)
- Time span: Simulated chronological transaction flow

**Key Research Questions:**
1. How rapidly does an AML classifier degrade when deployed on future transactions?
2. Can unsupervised detectors flag drift without ground-truth labels?
3. Is observed decay genuine concept drift or an artifact of label scarcity?
4. Does periodic retraining recover performance out-of-sample?

---

## 2. Experimental Design

### 2.1 Temporal Architecture

```
[Chunk 1] → [Chunk 2] → [Chunk 3] → ... → [Chunk 10]
 Training     Future Stream (Evaluated Chronologically & Out-of-Sample)
   Era
```

- **Chunk 1 (Reference Era):** Training, validation, feature encoder fitting
- **Chunks 2-10 (Streaming Future):** Sequential out-of-sample evaluation

### 2.2 Leakage Prevention Protocol

**Excluded Features:**
- `Laundering_type` - Direct label leakage
- `Sender_account`, `Receiver_account` - Entity identifiers
- `Time`, `Date` - Raw timestamps (derived features extracted instead)

**Imputation Strategy:**
- Numeric: Median imputed from Chunk 1 only
- Categorical: Mode imputed from Chunk 1 only
- Applied consistently to all future chunks without refitting

### 2.3 Model Configuration

**Algorithm:** XGBoost Classifier
- n_estimators: 300
- max_depth: 6
- learning_rate: 0.05
- scale_pos_weight: Calculated as neg/pos ratio (handles ~1:1000 imbalance)
- tree_method: 'hist' (optimized for large datasets)

**Threshold Tuning:**
- Grid search: 0.01 to 0.99, step 0.01
- Optimization target: Maximize F1 on 20% validation split
- Default threshold (0.5) performs poorly under extreme imbalance

### 2.4 Statistical Validation

- **Random Seeds:** 42, 7, 123
- **Confidence Intervals:** 95% CI calculated as 1.96 × std / √n
- **Stratified Splits:** Deterministic per-seed class-balanced 80/20 splits

---

## 3. Detailed Results

### 3.1 Phase C: Baseline Decay

**Training Era Performance (Chunk 1, 3 seeds):**

| Metric | Mean | Std | 95% CI |
|--------|------|-----|--------|
| F1 | 0.178 | 0.026 | ±0.029 |
| Precision | 0.506 | - | - |
| Recall | 0.118 | - | - |
| PR-AUC | 0.141 | 0.012 | ±0.014 |
| ROC-AUC | 0.885 | - | - |
| Positives | 855 | - | - |

**Future Stream Performance (Chunks 2-10):**

| Chunk | F1 Mean | F1 CI95 | PR-AUC | ROC-AUC | Positives | Relative Drop |
|-------|---------|---------|--------|---------|-----------|---------------|
| 2 | 0.083 | ±0.017 | 0.051 | 0.733 | 936 | -53% |
| 3 | 0.077 | ±0.012 | 0.048 | 0.690 | 974 | -57% |
| 4 | 0.104 | ±0.009 | 0.068 | 0.706 | 991 | -42% |
| 5 | 0.101 | ±0.013 | 0.064 | 0.697 | 1,016 | -43% |
| 6 | 0.074 | ±0.008 | 0.043 | 0.693 | 970 | -58% |
| 7 | 0.064 | ±0.006 | 0.037 | 0.696 | 1,022 | -64% |
| 8 | 0.053 | ±0.009 | 0.033 | 0.700 | 966 | -70% |
| 9 | 0.076 | ±0.016 | 0.045 | 0.720 | 1,040 | -57% |
| 10 | 0.107 | ±0.012 | 0.067 | 0.712 | 1,103 | -40% |
| **Mean (C2-10)** | **0.086** | - | **0.051** | **0.706** | - | **-52%** |

**Key Observation:** The model suffers immediate, severe decay. F1 drops from 0.178 to 0.053-0.107 (40-70% relative decline). PR-AUC collapses from 0.141 to 0.033-0.068 (55-77% decline). ROC-AUC drops from 0.885 to 0.690-0.733, confirming genuine ranking degradation beyond threshold miscalibration.

### 3.2 Phase D: Unsupervised Drift Detection

#### Detector 1: Numeric Feature PSI

**Monitored Features:** Amount, Amount_log, hour_of_day, day_of_week, tx_per_sender, amount_per_sender_mean

**Threshold:** PSI > 0.20 (industry standard for "major distribution change")

| Chunk | Max PSI | Drifted? |
|-------|---------|----------|
| 2 | 0.035 | No |
| 3 | 0.024 | No |
| 4 | 0.018 | No |
| 5 | 0.021 | No |
| 6 | 0.025 | No |
| 7 | 0.026 | No |
| 8 | 0.010 | No |
| 9 | 0.036 | No |
| 10 | 0.007 | No |

**Detection Rate:** 0/9 chunks

**Interpretation:** Marginal distributions of transaction features remain stable. Aggregate statistics (mean amounts, transaction timing patterns, per-sender behavior) appear unchanged.

#### Detector 2: Output Distribution Monitoring

**KS Test on Predicted Probabilities:**

| Chunk | KS Statistic | Critical Value (α=0.05) | Drifted? |
|-------|--------------|-------------------------|----------|
| 2 | 0.00336 | 0.00197 | **Yes** |
| 3 | 0.00257 | 0.00197 | **Yes** |
| 4 | 0.00417 | 0.00197 | **Yes** |
| 5 | 0.00216 | 0.00197 | **Yes** |
| 6 | 0.00288 | 0.00197 | **Yes** |
| 7 | 0.00349 | 0.00197 | **Yes** |
| 8 | 0.00124 | 0.00197 | No |
| 9 | 0.00301 | 0.00197 | **Yes** |
| 10 | 0.00190 | 0.00197 | No |

**Detection Rate:** 7/9 chunks

**Note on KS Power:** At N ≈ 10⁶, the KS critical value drops to ~0.002. This makes KS extremely sensitive to microscopic differences. It acts as a hypothesis-rejection test rather than a practical effect-size measure.

**PSI on Predicted Probabilities:**

| Chunk | PSI | Drifted? |
|-------|-----|----------|
| 2 | 0.00019 | No |
| 3 | 0.00010 | No |
| 4 | 0.00015 | No |
| 5 | 0.00009 | No |
| 6 | 0.00010 | No |
| 7 | 0.00011 | No |
| 8 | 0.00010 | No |
| 9 | 0.00017 | No |
| 10 | 0.00011 | No |

**Detection Rate:** 0/9 chunks

**Interpretation:** The model's score distribution shifts detectably by KS (statistically significant) but not by PSI (practically negligible). The two metrics bracket the reality: drift exists but manifests subtly in output distributions.

#### Detector 3: Domain Classifier

**Method:** Train XGBoost to distinguish Chunk 1 rows from Chunk N rows. AUC = 0.5 means indistinguishable; AUC > 0.55 indicates drift.

| Chunk | AUC Mean | AUC Std | 95% CI | Drifted? |
|-------|----------|---------|--------|----------|
| 2 | 0.597 | 0.002 | ±0.003 | **Yes** |
| 3 | 0.610 | 0.007 | ±0.008 | **Yes** |
| 4 | 0.595 | 0.006 | ±0.006 | **Yes** |
| 5 | 0.605 | 0.018 | ±0.021 | **Yes** |
| 6 | 0.595 | 0.004 | ±0.005 | **Yes** |
| 7 | 0.597 | 0.005 | ±0.006 | **Yes** |
| 8 | 0.595 | 0.007 | ±0.008 | **Yes** |
| 9 | 0.612 | 0.002 | ±0.003 | **Yes** |
| 10 | 0.584 | 0.013 | ±0.015 | **Yes** |

**Detection Rate:** 9/9 chunks

**Interpretation:** While univariate feature marginals remain stable (PSI blind), the multivariate joint distribution of behavioral features shifts detectably. A classifier can learn to separate reference from future data, indicating subtle but real distributional change.

#### Detector 4: Categorical Feature PSI

**Monitored Features:** Payment_currency, Received_currency, Sender_bank_location, Receiver_bank_location, Payment_type

**Worst PSI per Chunk:**

| Chunk | Worst Feature | PSI | Drifted? |
|-------|---------------|-----|----------|
| 2 | Sender_bank_location | 0.0016 | No |
| 3 | Receiver_bank_location | 0.0030 | No |
| 4 | Receiver_bank_location | 0.0023 | No |
| 5 | Receiver_bank_location | 0.0038 | No |
| 6 | Receiver_bank_location | 0.0033 | No |
| 7 | Receiver_bank_location | 0.0026 | No |
| 8 | Receiver_bank_location | 0.0032 | No |
| 9 | Receiver_bank_location | 0.0021 | No |
| 10 | Receiver_bank_location | 0.0018 | No |

**Detection Rate:** 0/9 chunks

#### Detection Summary Table

| Detector | Supervised? | Threshold | Alarms /9 | Interpretation |
|----------|-------------|-----------|-----------|----------------|
| Numeric PSI | No | PSI > 0.20 | 0/9 | Marginal distributions stable |
| Categorical PSI | No | PSI > 0.20 | 0/9 | Category rates unchanged |
| Output KS | No | α = 0.05 | 7/9 | Sensitive to micro-shifts |
| Output PSI | No | PSI > 0.20 | 0/9 | Effect size small |
| Domain Classifier | No | AUC > 0.55 | 9/9 | **Multivariate drift detected** |

### 3.3 Phase E: Label Scarcity Analysis

#### Experimental Setup

- **Label Budgets Tested:** 100%, 50%, 10%, 5%, 1%, 0.1%
- **Control Experiment:** Same budgets applied to Chunk 1 holdout (no drift present)
- **Minimum Positives Threshold:** 3 (fewer positives → NaN, not zero)

#### Aggregated Results: Future Stream vs No-Drift Control

| Label Budget | Future F1 (C2-10) | Control F1 (C1 Holdout) | Ratio | Valid Seeds | Trustworthy? |
|--------------|-------------------|-------------------------|-------|-------------|--------------|
| 100% | 0.092 | 0.170 | 0.54 | 3/3 | Yes |
| 50% | 0.094 | 0.184 | 0.51 | 3/3 | Yes |
| 10% | 0.088 | 0.166 | 0.53 | 3/3 | Yes |
| 5% | 0.075 | 0.131 | 0.57 | 3/3 | Yes |
| 1% | 0.114 | — (1/3) | — | 3/3 future, 1/3 control | **No** |
| 0.1% | N/A | N/A | — | 0/3 both arms | **No** |

#### Per-Chunk Breakdown at 100% Labels

| Chunk | Future F1 | Control F1 | Future/Control | Interpretation |
|-------|-----------|------------|----------------|----------------|
| 1 | 0.200 | 0.170 | 1.18 | Training era outperforms control (overfitting) |
| 2 | 0.100 | 0.170 | 0.59 | 41% drop vs control |
| 3 | 0.088 | 0.170 | 0.52 | 48% drop vs control |
| 4 | 0.113 | 0.170 | 0.66 | 34% drop vs control |
| 5 | 0.114 | 0.170 | 0.67 | 33% drop vs control |
| 6 | 0.081 | 0.170 | 0.48 | 52% drop vs control |
| 7 | 0.067 | 0.170 | 0.39 | 61% drop vs control |
| 8 | 0.058 | 0.170 | 0.34 | 66% drop vs control |
| 9 | 0.084 | 0.170 | 0.49 | 51% drop vs control |
| 10 | 0.118 | 0.170 | 0.69 | 31% drop vs control |

#### Key Findings

1. **Decay is Genuine Concept Drift:** At every trustworthy budget (100% → 5%), the drifted future scores only 51-57% of the no-drift control. This cannot be explained by small-sample variance.

2. **Label Scarcity is Surprisingly Tolerant for Measurement:** Future F1 stays flat from 100% to 5% labels (0.092 → 0.075, overlapping CIs). Losing labels costs little in measuring decay but blocks repairing it.

3. **Measurement Breaks Below 5%:** At 1% availability (~10 positives per chunk), 2/3 control runs found too few positives to produce any score. We report this as "not comparable" rather than inventing a ratio against a zero or starved control.

### 3.4 Phase F: Retraining Strategies

#### Protocol

- **Retraining Cadence:** Every 2 chunks
- **Evaluation:** Strictly out-of-sample (train on 1..c-1, test on c)
- **Threshold Re-tuning:** 20% validation split of retraining pool

#### Strategy Comparison

| Strategy | Mean F1 (C2-10) | Final F1 (C10) | Gain vs Frozen | % Improvement |
|----------|-----------------|----------------|----------------|---------------|
| no_retrain | 0.0916 | 0.1180 | 0.0000 | — |
| periodic_full | 0.1144 | 0.1551 | +0.0228 | **+28%** |
| periodic_sliding | 0.1065 | 0.1346 | +0.0149 | +19% |
| periodic_weighted (10x) | 0.0303 | 0.0232 | -0.0613 | -67% |

**Strategy Descriptions:**
- `no_retrain`: Frozen baseline from Phase C
- `periodic_full`: Retrain on all accumulated labeled history
- `periodic_sliding`: Retrain on last 2 chunks only (allows forgetting)
- `periodic_weighted`: Full history with 10× positive class up-weighting

#### Class Weight Multiplier Sweep

| Weight (w) | Mean F1 | Precision | Recall | Interpretation |
|------------|---------|-----------|--------|----------------|
| 1 (natural) | 0.1004 | 0.547 | 0.058 | Best F1 among weights |
| 10 | 0.0212 | 0.011 | 0.212 | Recall up, precision collapsed |
| 50 | 0.0076 | 0.004 | 0.413 | Further degradation |
| 100 | 0.0032 | 0.002 | 0.559 | Near-random precision |

**Interpretation:** Up-weighting a 1-in-1000 class does not improve ranking. It shifts the decision threshold, buying recall at catastrophic precision cost. F1 falls monotonically with w > 1.

#### Key Findings

1. **Full-History Retraining Works Best:** +28% F1 recovery with strong final-chunk performance (0.155).

2. **Sliding Window is Viable:** +19% gain with lower computational cost for long-running systems.

3. **Class Weighting Fails:** Despite increasing recall, precision collapse makes w > 1 harmful for F1.

4. **Retraining Cannot Fully Restore Performance:** Gap to training-era F1 (0.178) remains large, confirming P(Y|X) has shifted.

---

## 4. Research Contributions

### 4.1 Primary Contributions

1. **Empirical Characterization of AML Drift:** First comprehensive temporal decay study on SAML-D showing 40-70% F1 degradation immediately upon deployment.

2. **Unsupervised Detection Paradox:** Demonstrated that standard PSI monitoring is blind to AML drift despite genuine performance collapse. Domain classifier on behavioral features successfully detects drift without labels.

3. **Label Scarcity Resilience:** Proved that decay measurement is robust down to 5% label availability, but repair (retraining) remains label-dependent.

4. **Adaptation Limits:** Showed that retraining provides partial recovery (+28%) but cannot restore training-era performance, indicating true concept drift in P(Y|X).

### 4.2 Practical Implications

1. **For Practitioners:** PSI monitoring alone is insufficient for AML systems. Complement with domain classifiers or output distribution tests.

2. **For Researchers:** Label scarcity is less of a measurement problem than expected, but remains a bottleneck for model updates.

3. **For Regulators:** Performance degradation is real and detectable without requiring investigation outcomes. Monitoring frameworks should include multivariate drift tests.

---

## 5. Limitations

1. **Synthetic Data:** SAML-D is synthetic; real AML data may exhibit different drift patterns.

2. **Single Model Architecture:** XGBoost only; neural networks or ensemble methods may behave differently.

3. **Fixed Retraining Cadence:** Adaptive triggering based on drift signals not explored.

4. **No Feature Engineering:** Raw and derived features only; deep feature synthesis could improve performance.

5. **Binary Classification:** Multi-class laundering type prediction not addressed.

---

## 6. Reproducibility

### 6.1 Environment

- Python 3.12+
- numpy, pandas, scikit-learn, scipy, xgboost, matplotlib

### 6.2 Execution

```bash
# 1. Preprocess
python phaseB_chunking.py --data_path SAML-D.csv

# 2. Train baseline
python phaseC_baseline_model.py

# 3. Drift detection
python phaseD_drift_detectors.py

# 4. Label scarcity
python phaseE_label_scarcity.py

# 5. Retraining
python phaseF_retraining.py
```

### 6.3 Random Seeds

All experiments use seeds [42, 7, 123] for full reproducibility.

---

## 7. Citation

```bibtex
@article{aml_concept_drift_2026,
  title={Empirical Analysis of Concept Drift, Unsupervised Detection, and Adaptation Bottlenecks in Anti-Money-Laundering},
  author={Patel, Vihaan},
  year={2026}
}
```

---

## 8. Appendix: Raw Data Tables

### A.1 Complete Phase C Results

```
chunk,f1_mean,f1_std,precision_mean,recall_mean,roc_auc_mean,pr_auc_mean,n_positives
1,0.178,0.026,0.506,0.118,0.885,0.141,855
2,0.083,0.015,0.340,0.052,0.733,0.051,936
3,0.077,0.010,0.299,0.048,0.690,0.048,974
4,0.104,0.008,0.384,0.064,0.706,0.068,991
5,0.101,0.011,0.359,0.064,0.697,0.064,1016
6,0.074,0.007,0.358,0.045,0.693,0.043,970
7,0.064,0.005,0.352,0.038,0.696,0.037,1022
8,0.053,0.008,0.271,0.032,0.700,0.033,966
9,0.076,0.014,0.337,0.046,0.720,0.045,1040
10,0.107,0.011,0.426,0.065,0.712,0.067,1103
```

### A.2 Domain Classifier Results

```
chunk,auc_mean,auc_std,auc_ci95,drifted
2,0.597,0.002,0.003,True
3,0.610,0.007,0.008,True
4,0.595,0.006,0.006,True
5,0.605,0.018,0.021,True
6,0.595,0.004,0.005,True
7,0.597,0.005,0.006,True
8,0.595,0.007,0.008,True
9,0.612,0.002,0.003,True
10,0.584,0.013,0.015,True
```

### A.3 Retraining Summary

```
strategy,f1_mean_chunks2_N,f1_final_chunk,mean_gain_vs_frozen,pr_auc_mean,recall_mean
no_retrain,0.0916,0.1180,0.0000,0.0555,0.0528
periodic_full,0.1144,0.1551,0.0228,0.0719,0.0646
periodic_sliding,0.1065,0.1346,0.0149,0.0661,0.0638
periodic_weighted,0.0303,0.0232,-0.0613,0.0656,0.2274
```

---

*Document generated: 2026-10-29*
*Project: AML Concept Drift Detection*
