"""
Plot Retraining Comparison: With vs Without Retraining
A clean, student-friendly script to plot model performance over time.
Saves all charts into the 'results_maincomparsion' folder.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 0. Create output folder
# -------------------------------------------------------------
output_dir = "results_maincomparsion"
os.makedirs(output_dir, exist_ok=True)

# -------------------------------------------------------------
# 1. Load data from CSV files
# -------------------------------------------------------------
# Baseline results (without retraining)
baseline_df = pd.read_csv("results_phaseC/baseline_decay_summary.csv")

# Retraining results (with different retraining strategies)
retrain_df = pd.read_csv("results_phaseF/retraining_results.csv")

# -------------------------------------------------------------
# 2. Filter data by strategy
# -------------------------------------------------------------
# Without retraining (frozen baseline model from Chunk 1)
no_retrain = retrain_df[retrain_df["strategy"] == "no_retrain"].sort_values("chunk")

# With retraining (retrain every 2 chunks on all past data)
retrain_full = retrain_df[retrain_df["strategy"] == "periodic_full"].sort_values("chunk")

# With sliding window retraining (retrain on last 2 chunks only)
retrain_sliding = retrain_df[retrain_df["strategy"] == "periodic_sliding"].sort_values("chunk")

# With class weighting (10x positive class weight)
retrain_weighted = retrain_df[retrain_df["strategy"] == "periodic_weighted"].sort_values("chunk")

# -------------------------------------------------------------
# 3. Create a 2x2 grid of plots
# -------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# -------------------------------------------------------------
# Plot 1: F1 Score over time (Chunks 1 to 10)
# -------------------------------------------------------------
ax1 = axes[0, 0]

ax1.plot(baseline_df["chunk"], baseline_df["f1_mean"], color="red", marker="o",
         linestyle="--", label="Without Retraining (Frozen Model)")
ax1.plot(retrain_full["chunk"], retrain_full["f1"], color="green", marker="s",
         label="With Retraining (Full History)")
ax1.plot(retrain_sliding["chunk"], retrain_sliding["f1"], color="blue", marker="^",
         label="With Retraining (Sliding Window)")
ax1.plot(retrain_weighted["chunk"], retrain_weighted["f1"], color="purple", marker="d",
         label="With Retraining (10x Weighted)")

ax1.set_title("F1-Score Over Time (Concept Drift vs Retraining)")
ax1.set_xlabel("Chunk (Time Period)")
ax1.set_ylabel("F1-Score")
ax1.set_xticks(range(1, 11))
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="lower left", fontsize=8.5)

# -------------------------------------------------------------
# Plot 2: PR-AUC (Precision-Recall Area Under Curve)
# -------------------------------------------------------------
ax2 = axes[0, 1]

ax2.plot(baseline_df["chunk"], baseline_df["pr_auc_mean"], color="red", marker="o",
         linestyle="--", label="Without Retraining (Frozen Model)")
ax2.plot(retrain_full["chunk"], retrain_full["pr_auc"], color="green", marker="s",
         label="With Retraining (Full History)")
ax2.plot(retrain_sliding["chunk"], retrain_sliding["pr_auc"], color="blue", marker="^",
         label="With Retraining (Sliding Window)")
ax2.plot(retrain_weighted["chunk"], retrain_weighted["pr_auc"], color="purple", marker="d",
         label="With Retraining (10x Weighted)")

ax2.set_title("PR-AUC Over Time (Ranking Power)")
ax2.set_xlabel("Chunk (Time Period)")
ax2.set_ylabel("PR-AUC")
ax2.set_xticks(range(1, 11))
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(loc="lower left", fontsize=8.5)

# -------------------------------------------------------------
# Plot 3: Precision vs Recall
# -------------------------------------------------------------
ax3 = axes[1, 0]

ax3.scatter(no_retrain["recall"] * 100, no_retrain["precision"] * 100,
            color="red", marker="o", s=50, label="Without Retraining")
ax3.scatter(retrain_full["recall"] * 100, retrain_full["precision"] * 100,
            color="green", marker="s", s=50, label="With Retraining (Full)")
ax3.scatter(retrain_sliding["recall"] * 100, retrain_sliding["precision"] * 100,
            color="blue", marker="^", s=50, label="With Retraining (Sliding)")
ax3.scatter(retrain_weighted["recall"] * 100, retrain_weighted["precision"] * 100,
            color="purple", marker="d", s=50, label="With Retraining (10x Weighted)")

ax3.set_title("Precision vs Recall (Alert Quality)")
ax3.set_xlabel("Recall (%)")
ax3.set_ylabel("Precision (%)")
ax3.grid(True, linestyle="--", alpha=0.5)
ax3.legend(loc="upper right", fontsize=8.5)

# -------------------------------------------------------------
# Plot 4: F1 Score Gain from Retraining (Bar Chart)
# -------------------------------------------------------------
ax4 = axes[1, 1]

# Calculate gain (Retrained F1 - Frozen F1)
f1_frozen = no_retrain.set_index("chunk")["f1"]
f1_gain_full = retrain_full.set_index("chunk")["f1"] - f1_frozen
f1_gain_sliding = retrain_sliding.set_index("chunk")["f1"] - f1_frozen

chunks = retrain_full["chunk"].values
width = 0.35
x = list(range(len(chunks)))

ax4.bar([i - width/2 for i in x], f1_gain_full.values, width=width,
        color="green", label="Full History Gain (Delta F1)")
ax4.bar([i + width/2 for i in x], f1_gain_sliding.values, width=width,
        color="blue", label="Sliding Window Gain (Delta F1)")

ax4.axhline(0, color="black", linewidth=0.8)
ax4.set_title("F1-Score Gain from Retraining")
ax4.set_xlabel("Chunk (Time Period)")
ax4.set_ylabel("F1 Gain (vs Without Retraining)")
ax4.set_xticks(x)
ax4.set_xticklabels([f"Chunk {c}" for c in chunks], rotation=30, ha="right")
ax4.grid(True, linestyle="--", alpha=0.5, axis="y")
ax4.legend(loc="upper left", fontsize=8.5)

# -------------------------------------------------------------
# 4. Adjust layout and save the charts
# -------------------------------------------------------------
plt.suptitle("Model Performance With and Without Retraining (SAML-D Dataset)", fontsize=13, y=0.98)
plt.tight_layout()

png_path = os.path.join(output_dir, "retraining_comparison_plot.png")
pdf_path = os.path.join(output_dir, "retraining_comparison_plot.pdf")

plt.savefig(png_path, dpi=200)
plt.savefig(pdf_path)
print(f"Saved plots to {png_path} and {pdf_path}")
