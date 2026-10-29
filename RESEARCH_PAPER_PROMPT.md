# Prompt for paper.ai / AI Paper Writing Tool

Copy everything below and provide it to paper.ai or any AI paper-writing assistant.

---

## Task

Write a complete, publishable research paper using the data and context provided. Write in a natural, human academic voice — avoid formulaic AI phrasing, include genuine limitations and cautious interpretation, and make confident but precise claims backed by the numbers.

## Project Files to Reference (all in this repository)

- RESEARCH_PAPER_CONTEXT.md — Full research context, methodology, and all results tables
- results_explained.txt — Plain-English summary of all phase results
- README.md — Project overview and architecture
- phaseC_baseline_model.py — Baseline model training & decay measurement
- phaseD_drift_detectors.py — Unsupervised drift detection (PSI, KS, Domain Classifier)
- phaseE_label_scarcity.py — Label scarcity vs no-drift control experiment
- phaseF_retraining.py — Retraining strategies and weight sweep
- common.py — Shared utilities (metrics, encoders, PSI, KS)

## Paper Requirements

### Structure (Standard Academic Format)
1. **Title** — Something like: "Concept Drift Without Labels: An Empirical Analysis of AML Fraud Detection Decay and Unsupervised Recovery Strategies"
2. **Abstract** (~200 words) — State the problem, methods, key results, and implications
3. **Introduction** — Motivation (AML context, concept drift, label latency), research questions, contributions
4. **Related Work** (brief) — Cite relevant AML drift and financial ML monitoring work
5. **Dataset & Experimental Setup** — SAML-D description, temporal chunking, leakage prevention, model configuration, statistical validation protocol
6. **Results** — Organized by the four phases (C-F)
7. **Discussion** — Interpret findings, limitations, implications
8. **Conclusion** — Summarize and suggest next steps
9. **References** — Minimum 8-10 relevant citations
10. **Appendix** — Raw data tables

### Key Data to Incorporate (from RESEARCH_PAPER_CONTEXT.md)

**Phase C — Decay:** Training-era F1=0.178±0.029 drops to 0.053–0.107 across chunks 2–10 (mean 0.086, ~54% relative decline). PR-AUC falls from 0.141 to 0.033–0.068. ROC-AUC drops from 0.885 to ~0.70.

**Phase D — Detection:**
- Numeric PSI: 0/9 chunks, max PSI=0.036
- Categorical PSI: 0/9 chunks, max PSI=0.0038
- Output KS: 7/9 chunks, statistics 0.001–0.004 vs critical ~0.002
- Output PSI: 0/9 chunks, max PSI=0.00019
- Domain Classifier: 9/9 chunks, AUC=0.584–0.612 (mean ~0.598)

**Phase E — Label Scarcity:** 100%, 50%, 10%, 5% budgets all comparable. Future scores 51–57% of control (ratios: 0.54, 0.51, 0.53, 0.57). At 1% and 0.1%, measurement becomes unreliable.

**Phase F — Retraining:**
- no_retrain: 0.0916 mean F1
- periodic_full: 0.1144 (+28% gain)
- periodic_sliding: 0.1065 (+19% gain)
- periodic_weighted: 0.0303 (degraded)
- Weight sweep w∈{1,10,50,100}: F1 falls monotonically with w>1

### Writing Style Guidelines (IMPORTANT)
- Use **past tense** for what was done ("we evaluated," "we observed")
- Include **cautious claims** ("evidence suggests," "results indicate") with specific numbers
- Mention **limitations explicitly** in the Discussion section
- Use **your own analytical voice** — do not default to AI-sounding generic statements
- Include **specific thresholds, values, and confidence intervals** when stating results
- Write naturally; vary sentence structure; include qualifying thoughts

### Technical Requirements
- LaTeX format preferred (for arXiv/academic submission)
- APA or ACM citation style for references
- Figures referenced by caption, with data source noted
- Tables formatted for readability
- All claims traceable to specific numbers above

---

Begin with the title, then proceed through the sections in order. Produce a complete, self-contained research paper.~