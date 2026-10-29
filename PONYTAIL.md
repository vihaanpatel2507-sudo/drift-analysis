---
name: ponytail-optimization
description: Ponytail skill installed and applied to AML codebase
metadata:
  type: project
---
Ponytail (lazy senior dev mode) applied to main-drift-main. Key optimizations:
- Single-source common.py avoids duplication (already done well)
- Reused sklearn/scipy for metrics instead of custom implementations
- Used pandas native operations (map, groupby) over loops
- Shortcut comments for large-sample KS behavior (already present)
- No new dependencies added - reused existing xgboost/pandas/numpy

Risk: Large-sample KS critical values (~0.002 at N~1M) mean statistical significance != practical importance; code already notes this.
