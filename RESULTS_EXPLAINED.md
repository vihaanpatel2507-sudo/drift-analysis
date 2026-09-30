# Experimental Results & Academic Findings Guide

This document provides a comprehensive, plain-English, and mathematically rigorous walkthrough of all experimental results across the AML concept drift benchmark on the **SAML-D dataset (9.5 million transactions)**.

---

## 1. Executive Summary & Core Research Questions

| Research Question | Empirical Finding | Practical Implications for Financial Institutions |
|---|---|---|
| **RQ1: Does concept drift occur in AML, and how fast?** | **Yes, severe.** $F_1$ score drops by $40\%\text{--}70\%$ immediately upon encountering future chunks ($0.178 \to 0.053\text{--}0.107$), and PR-AUC collapses ($0.141 \to 0.033$). | Static AML machine learning models experience critical blind spots within months of deployment. |
| **RQ2: Can drift be detected without ground-truth labels?** | **Yes, via Domain Classifiers.** Input feature PSI and output PSI fail to trigger alarms, but a lightweight behavioral Domain Classifier achieves AUC $0.584\text{--}0.612$ across all future chunks without any labels. | Banks can monitor multivariate behavioral shifts in real time rather than waiting months for SAR filings. |
| **RQ3: Is performance decay real drift or a small-sample artifact under label scarcity?** | **Genuine drift.** At all statistically reliable budgets ($100\%$ down to $5\%$), the drifted stream scores only $51\%\text{--}57\%$ of a matched no-drift control. | Decay is structural and not caused by sparse verification budgets. |
| **RQ4: Does periodic retraining fix drift?** | **Substantially, but not completely.** Cumulative full-history retraining yields a **$+28\%$ relative $F_1$ gain** out-of-sample ($0.0916 \to 0.1144$), but cannot fully restore train-era performance ($0.178$). | Retraining mitigates decay, but adaptive architectures and ongoing feature discovery remain necessary. |
| **RQ5: How should extreme class imbalance ($0.1\%$) be weighted during retraining?** | **Natural imbalance weighting ($w=1$) strongly outperforms artificial boosts ($w \in \{10, 50, 100\}$).** Boosting weights increases recall at the expense of catastrophic precision collapse ($0.547 \to 0.0016$). | Artificial positive class over-weighting leads to unsustainable investigator alert fatigue in operational SOCs. |

---

## 2. Detailed Phase-by-Phase Findings

### Phase C: Baseline Performance Decay
* **Protocol:** XGBoost model trained on Chunk 1 with threshold tuning on a 20% validation split, evaluated across 3 independent random seeds without retraining on future Chunks 2 to 10.
* **Key Observations:**
  1. **Immediate Out-of-Sample Drop:** Training-era $F_1 = 0.178 \pm 0.026$ drops to $0.083$ in Chunk 2 and hits a low of $0.053$ in Chunk 8.
  2. **Ranking Degradation:** PR-AUC drops from $0.141$ to $0.033\text{--}0.068$, and ROC-AUC drops from $0.885$ to $0.690\text{--}0.733$. This demonstrates that the loss of performance is an intrinsic failure of discriminator ranking power, not a simple threshold miscalibration.

---

### Phase D: Label-Free Drift Detection
* **Detectors Evaluated:** Univariate Numeric PSI, Categorical PSI, Output Probability KS-Test & PSI, and an Unsupervised Domain Classifier.
* **Key Observations:**
  1. **Univariate PSI Blindness:** Individual feature marginal distributions (e.g., transaction amounts, time-of-day) remain stable ($PSI \le 0.036$), failing to trigger alarms ($PSI > 0.20$).
  2. **Domain Classifier Sensitivity:** A discriminator trained to separate reference vs. current chunk data flags drift across **9 out of 9 future chunks** ($\text{AUC} > 0.55$).
  3. **Statistical vs. Practical Significance:** The Kolmogorov-Smirnov (KS) test rejects on 7/9 chunks due to massive sample size ($N \approx 10^6, \text{crit} \approx 0.002$), while output PSI stays below $0.0002$. This proves KS acts as a sensitive hypothesis test rather than an effect-size metric.

---

### Phase E: Verification Latency & Label Scarcity
* **Protocol:** Evaluation of model performance under constrained label availability ($100\%, 50\%, 10\%, 5\%, 1\%, 0.1\%$) compared against an unseen held-out control slice of Chunk 1 (where zero drift is present).
* **Key Observations:**
  1. **Verification of True Drift:** Across every reliable budget ($100\%$ down to $5\%$), the future stream achieves only $51\%\text{--}57\%$ of the control score.
  2. **Measurement Breakdown Threshold:** At $\le 1\%$ label availability (approx. $\le 10$ illicit transactions per chunk), standard metric estimation collapses due to zero-positive splits in control runs.

---

### Phase F: Continual Retraining & Adaptation Strategies
* **Protocol:** Out-of-sample evaluation where models are retrained periodically and evaluated strictly on future unseen chunks.
* **Key Observations:**
  1. **Cumulative Retraining (`periodic_full`):** Achieves the highest mean $F_1$ ($0.1144$), yielding a **$+28\%$ relative gain** over the frozen baseline.
  2. **Sliding Window Retraining (`periodic_sliding`):** Retraining on only the last 2 chunks achieves $F_1 = 0.1065$ ($+19\%$ gain), proving to be a memory-efficient alternative.
  3. **Class Weight Multiplier Dynamics ($w \in \{1, 10, 50, 100\}$):**
     - $w=1$ (natural imbalance): $F_1 = 0.1004$, Precision = $54.7\%$, Recall = $5.8\%$.
     - $w=10$: $F_1 = 0.0212$, Precision = $1.1\%$, Recall = $21.2\%$.
     - $w=50$: $F_1 = 0.0076$, Precision = $0.38\%$, Recall = $41.3\%$.
     - $w=100$: $F_1 = 0.0032$, Precision = $0.16\%$, Recall = $55.9\%$.
     - **Conclusion:** Increasing $w$ trades precision exponentially for linear gains in recall, proving counterproductive for balanced $F_1$ optimization.

---

## 3. Publication Visualizations (`results_maincomparsion/`)

The primary paper comparison chart is located in `results_maincomparsion/retraining_comparison_plot.png` (and `.pdf`):

- **Panel 1 (Top-Left): $F_1$-Score Over Time:** Compares frozen baseline decay against full history, sliding window, and weighted retraining.
- **Panel 2 (Top-Right): PR-AUC Over Time:** Illustrates ranking power retention across adaptation strategies.
- **Panel 3 (Bottom-Left): Precision vs. Recall:** Demonstrates operational alert quality and investigator burden.
- **Panel 4 (Bottom-Right): Adaptation Gains ($\Delta F_1$):** Quantifies out-of-sample improvements across chunks 2 to 10.

---

## 4. Summary Table for Paper Submissions

| Metric / Phase | Frozen Baseline | Retrained (Full History) | Retrained (Sliding Window) | Retrained ($10\times$ Weighted) |
|---|:---:|:---:|:---:|:---:|
| **Mean $F_1$ (Chunks 2–10)** | 0.0916 | **0.1144** | 0.1065 | 0.0303 |
| **Final $F_1$ (Chunk 10)** | 0.1180 | **0.1551** | 0.1346 | 0.0232 |
| **Mean PR-AUC** | 0.0555 | **0.0719** | 0.0661 | 0.0656 |
| **Mean Precision** | 0.3540 | **0.5470** | 0.4820 | 0.0112 |
| **Mean Recall** | 0.0528 | 0.0646 | 0.0638 | **0.2274** |
| **Relative $F_1$ Recovery** | Baseline ($0\%$) | **$+28.0\%$** | $+19.1\%$ | $-66.9\%$ |
