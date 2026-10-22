"""
Plot Retraining Comparison: Frozen vs Retrained
A simple, student-friendly chart showing the key finding:
retraining helps recover from concept drift.
Saves to 'results_maincomparsion' folder.
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
# 2. Select the two strategies to compare
# -------------------------------------------------------------
# Frozen baseline (no retraining) - from retrain_df
frozen = retrain_df[retrain_df["strategy"] == "no_retrain"].sort_values("chunk")

# Best retraining strategy: Full History
retrained = retrain_df[retrain_df["strategy"] == "periodic_full"].sort_values("chunk")

# -------------------------------------------------------------
# 3. Calculate key statistics
# -------------------------------------------------------------
# Average F1 improvement
avg_f1_improvement = (retrained["f1"].mean() - frozen["f1"].mean()) * 100

# Final chunk (Chunk 10) F1 values
frozen_final = frozen[frozen["chunk"] == 10]["f1"].values[0]
retrained_final = retrained[retrained["chunk"] == 10]["f1"].values[0]
final_improvement = ((retrained_final - frozen_final) / frozen_final) * 100

# -------------------------------------------------------------
# 4. Create a simple 2x2 grid of plots
# -------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Colors
COLOR_FROZEN = "#E74C3C"   # Red
COLOR_RETRAINED = "#27AE60"  # Green

# -------------------------------------------------------------
# Panel 1: F1-Score Over Time (The Main Finding)
# -------------------------------------------------------------
ax1 = axes[0, 0]

ax1.plot(baseline_df["chunk"], baseline_df["f1_mean"],
         color=COLOR_FROZEN, linewidth=2.5, marker="o", markersize=8,
         linestyle="--", label="Frozen (No Retraining)")
ax1.plot(retrained["chunk"], retrained["f1"],
         color=COLOR_RETRAINED, linewidth=2.5, marker="s", markersize=8,
         label="Retrained (Full History)")

# Add annotation for improvement
ax1.annotate(f"+{avg_f1_improvement:.1f}% avg gain",
             xy=(6, 0.12), fontsize=11, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

ax1.set_title("F1-Score Over Time\n(Concept Drift Recovery)", fontsize=12, fontweight="bold")
ax1.set_xlabel("Time Chunk", fontsize=10)
ax1.set_ylabel("F1-Score", fontsize=10)
ax1.set_xticks(range(1, 11))
ax1.set_ylim(0, 0.22)
ax1.grid(True, linestyle="--", alpha=0.4)
ax1.legend(loc="upper right", fontsize=10)

# -------------------------------------------------------------
# Panel 2: PR-AUC Over Time (Ranking Power)
# -------------------------------------------------------------
ax2 = axes[0, 1]

ax2.plot(baseline_df["chunk"], baseline_df["pr_auc_mean"],
         color=COLOR_FROZEN, linewidth=2.5, marker="o", markersize=8,
         linestyle="--", label="Frozen (No Retraining)")
ax2.plot(retrained["chunk"], retrained["pr_auc"],
         color=COLOR_RETRAINED, linewidth=2.5, marker="s", markersize=8,
         label="Retrained (Full History)")

ax2.set_title("Ranking Quality (PR-AUC)\n(Higher is Better)", fontsize=12, fontweight="bold")
ax2.set_xlabel("Time Chunk", fontsize=10)
ax2.set_ylabel("PR-AUC", fontsize=10)
ax2.set_xticks(range(1, 11))
ax2.set_ylim(0, 0.18)
ax2.grid(True, linestyle="--", alpha=0.4)
ax2.legend(loc="upper right", fontsize=10)

# -------------------------------------------------------------
# Panel 3: Precision vs Recall Trade-off
# -------------------------------------------------------------
ax3 = axes[1, 0]

# Connect points with lines to show progression over time
ax3.plot(frozen["recall"] * 100, frozen["precision"] * 100,
         color=COLOR_FROZEN, linewidth=2, linestyle="--", alpha=0.6)
ax3.scatter(frozen["recall"] * 100, frozen["precision"] * 100,
            color=COLOR_FROZEN, s=60, marker="o", label="Frozen", zorder=5)

ax3.plot(retrained["recall"] * 100, retrained["precision"] * 100,
         color=COLOR_RETRAINED, linewidth=2, linestyle="-", alpha=0.6)
ax3.scatter(retrained["recall"] * 100, retrained["precision"] * 100,
            color=COLOR_RETRAINED, s=60, marker="s", label="Retrained", zorder=5)

ax3.set_title("Precision vs Recall\n(Operational Trade-off)", fontsize=12, fontweight="bold")
ax3.set_xlabel("Recall (%) - Caught Frauds", fontsize=10)
ax3.set_ylabel("Precision (%) - True Alerts", fontsize=10)
ax3.grid(True, linestyle="--", alpha=0.4)
ax3.legend(loc="upper right", fontsize=10)
ax3.set_xlim(0, 30)
ax3.set_ylim(0, 100)

# -------------------------------------------------------------
# Panel 4: Summary Bar Chart (F1 Gain Per Chunk)
# -------------------------------------------------------------
ax4 = axes[1, 1]

chunks = retrained["chunk"].values
f1_gain = (retrained.set_index("chunk")["f1"] - frozen.set_index("chunk")["f1"]) * 100  # Convert to percentage

# Color bars green if positive (improvement), red if negative
colors = [COLOR_RETRAINED if g > 0 else COLOR_FROZEN for g in f1_gain.values]

bars = ax4.bar(chunks, f1_gain.values, color=colors, edgecolor="black", linewidth=0.5)

# Add value labels on bars
for bar, val in zip(bars, f1_gain.values):
    ypos = bar.get_height()
    ax4.annotate(f"+{val:.1f}" if val >= 0 else f"{val:.1f}",
                 xy=(bar.get_x() + bar.get_width()/2, ypos),
                 ha="center", va="bottom" if ypos >= 0 else "top",
                 fontsize=8)

ax4.axhline(0, color="black", linewidth=1)
ax4.set_title("F1 Improvement Per Chunk\n(Retrained - Frozen)", fontsize=12, fontweight="bold")
ax4.set_xlabel("Time Chunk", fontsize=10)
ax4.set_ylabel("F1 Improvement (percentage points)", fontsize=10)
ax4.set_xticks(chunks)
ax4.grid(True, linestyle="--", alpha=0.4, axis="y")

# -------------------------------------------------------------
# 5. Add overall title and save
# -------------------------------------------------------------
fig.suptitle("Key Finding: Retraining Recovers F1-Score Lost to Concept Drift\n"
             f"Average Improvement: +{avg_f1_improvement:.1f} percentage points | "
             f"Final Chunk Improvement: +{final_improvement:.1f}%",
             fontsize=13, fontweight="bold", y=1.02)

plt.tight_layout()

# Save outputs
png_path = os.path.join(output_dir, "retraining_comparison_plot.png")
pdf_path = os.path.join(output_dir, "retraining_comparison_plot.pdf")

plt.savefig(png_path, dpi=200, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")

print(f"✓ Saved to {png_path}")
print(f"✓ Saved to {pdf_path}")
print(f"\nSummary:")
print(f"  - Average F1 improvement: +{avg_f1_improvement:.1f} percentage points")
print(f"  - Final chunk (Chunk 10) F1: Frozen={frozen_final:.3f}, Retrained={retrained_final:.3f}")
print(f"  - Final chunk improvement: +{final_improvement:.1f}%")