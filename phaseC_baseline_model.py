"""
Phase C: Baseline AML Model Training & Temporal Performance Decay Measurement

Trains an XGBoost classifier on the initial reference era (Chunk 1), tunes
the decision threshold on a held-out stratified validation slice, and evaluates
the frozen model across subsequent chunks to measure concept drift decay.

Uses multi-seed execution (default seeds: 42, 7, 123) and reports mean +/- 95% CI.
No scikit-learn / scipy dependencies (uses common.py).
"""

import argparse
import glob
import json
import os
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from common import (
    BASELINE_SEEDS,
    load_chunk,
    simple_train_test_split,
    evaluate,
    tune_threshold,
    add_seed_manifest,
    ci95,
    list_chunk_files,
)

DEFAULT_CHUNK_DIR = "chunks"
DEFAULT_OUT_DIR = "results_phaseC"


def train_baseline_for_seed(chunk_1_path, seed):
    """
    Train an XGBoost baseline on 80% of Chunk 1, tuning the decision threshold
    on the remaining 20% validation slice to maximize F1.
    """
    X1, y1, encoders = load_chunk(chunk_1_path, fit_encoders=True)
    X_train, X_val, y_train, y_val = simple_train_test_split(X1, y1, test_size=0.2, seed=seed)

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_train, y_train)

    val_proba = model.predict_proba(X_val)[:, 1]
    best_threshold = tune_threshold(model, X_val, y_val)
    val_pred = (val_proba >= best_threshold).astype(int)
    val_metrics = evaluate(y_val.values, val_pred, val_proba)

    val_metrics.update({
        "seed": seed,
        "threshold": best_threshold,
        "n_val": len(y_val),
    })

    return model, encoders, best_threshold, val_metrics


def evaluate_stream(model, encoders, threshold, chunk_files, seed):
    """Evaluate a frozen model across all chunks in chronological order."""
    results = []
    for cnum, cf in chunk_files:
        X, y, _ = load_chunk(cf, encoders=encoders, fit_encoders=False)
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= threshold).astype(int)

        m = evaluate(y.values, pred, proba)
        m.update({
            "chunk": cnum,
            "seed": seed,
            "threshold": threshold,
        })
        results.append(m)
    return results


def main():
    parser = argparse.ArgumentParser(description="Baseline model training and performance decay measurement.")
    parser.add_argument("--chunks_dir", type=str, default=DEFAULT_CHUNK_DIR, help="Directory containing chunk CSVs")
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR, help="Output directory for results")
    parser.add_argument("--seeds", type=int, nargs="+", default=BASELINE_SEEDS, help="Random seeds for training")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Discover chunk files
    if not os.path.exists(args.chunks_dir):
        print(f"Error: Chunks directory '{args.chunks_dir}' does not exist.")
        return

    raw_files = list_chunk_files(args.chunks_dir)
    if not raw_files:
        print(f"No chunk_*.csv files found in '{args.chunks_dir}'.")
        return

    chunk_files = [(cnum, os.path.join(args.chunks_dir, fname)) for cnum, fname in raw_files]
    chunk_1_path = chunk_files[0][1]

    print("=" * 70)
    print("PHASE C: BASELINE MODEL TRAINING & TEMPORAL PERFORMANCE DECAY")
    print(f"Chunks found: {len(chunk_files)} | Seeds: {args.seeds}")
    print("=" * 70)

    all_seed_results = []
    val_records = []
    thresholds_map = {}
    saved_bundle = None

    for seed in args.seeds:
        print(f"\n--- Running Seed {seed} ---")
        model, encoders, threshold, val_metrics = train_baseline_for_seed(chunk_1_path, seed=seed)
        val_records.append(val_metrics)
        thresholds_map[str(seed)] = threshold

        print(f"Seed {seed} Validation: F1={val_metrics['f1']:.4f}, "
              f"ROC-AUC={val_metrics['roc_auc']:.4f}, PR-AUC={val_metrics['pr_auc']:.4f}, "
              f"Threshold={threshold:.2f}")

        # Keep the first seed's bundle as the default reference artifact
        if saved_bundle is None:
            saved_bundle = {"model": model, "encoders": encoders, "threshold": threshold}

        seed_results = evaluate_stream(model, encoders, threshold, chunk_files, seed=seed)
        all_seed_results.extend(seed_results)

        for r in seed_results:
            print(f"  Chunk {r['chunk']:02d}: F1={r['f1']:.4f} | PR-AUC={r['pr_auc']:.4f} | "
                  f"ROC-AUC={r['roc_auc']:.4f} (Positives: {r['n_positives']})")

    # 1. Save detailed results across seeds
    results_df = pd.DataFrame(all_seed_results)
    results_df.to_csv(os.path.join(args.out_dir, "baseline_decay_results.csv"), index=False)

    # 2. Save validation metrics
    val_df = pd.DataFrame(val_records)
    val_df.to_csv(os.path.join(args.out_dir, "baseline_validation_metrics.csv"), index=False)

    # 3. Save threshold configuration JSON
    threshold_meta = {
        "threshold": thresholds_map.get("42", list(thresholds_map.values())[0]),
        "tuned_on": "chunk_01 validation slice (20% stratified split)",
        "protocol": "F1-maximizing threshold, grid 0.01..0.99 step 0.01",
        "per_seed_thresholds": thresholds_map,
        "seeds": args.seeds,
    }
    with open(os.path.join(args.out_dir, "baseline_threshold.json"), "w") as f_th:
        json.dump(threshold_meta, f_th, indent=2)

    # 4. Compute and save summary table (mean, std, 95% CI)
    summary_rows = []
    n_seeds = len(args.seeds)
    for cnum in sorted(results_df["chunk"].unique()):
        sub = results_df[results_df["chunk"] == cnum]
        f1_mean = float(sub["f1"].mean())
        f1_std = float(sub["f1"].std(ddof=1)) if len(sub) > 1 else 0.0
        pr_mean = float(sub["pr_auc"].mean())
        pr_std = float(sub["pr_auc"].std(ddof=1)) if len(sub) > 1 else 0.0

        summary_rows.append({
            "chunk": cnum,
            "f1_mean": f1_mean,
            "f1_std": f1_std,
            "precision_mean": float(sub["precision"].mean()),
            "recall_mean": float(sub["recall"].mean()),
            "roc_auc_mean": float(sub["roc_auc"].mean()),
            "pr_auc_mean": pr_mean,
            "pr_auc_std": pr_std,
            "n_positives": float(sub["n_positives"].iloc[0]),
            "n_rows": float(sub["n_rows"].iloc[0]),
            "f1_ci95": ci95(f1_std, n_seeds),
            "pr_auc_ci95": ci95(pr_std, n_seeds),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = add_seed_manifest(summary_df, args.seeds)
    summary_df.to_csv(os.path.join(args.out_dir, "baseline_decay_summary.csv"), index=False)

    # 5. Save reference model bundle
    if saved_bundle:
        with open(os.path.join(args.out_dir, "baseline_model.pkl"), "wb") as f_pkl:
            pickle.dump(saved_bundle, f_pkl)

    # 6. Save summary plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].errorbar(summary_df["chunk"], summary_df["f1_mean"], yerr=summary_df["f1_ci95"],
                    fmt="-o", color="crimson", label="F1 (mean +/- 95% CI)", capsize=4)
    axes[0].plot(summary_df["chunk"], summary_df["precision_mean"], "--s", color="steelblue", label="Precision")
    axes[0].plot(summary_df["chunk"], summary_df["recall_mean"], "--^", color="darkorange", label="Recall")
    axes[0].set_xlabel("Chunk (temporal order)")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Concept Drift: Baseline F1 Decay Over Time")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].errorbar(summary_df["chunk"], summary_df["pr_auc_mean"], yerr=summary_df["pr_auc_ci95"],
                    fmt="-s", color="forestgreen", label="PR-AUC / AP", capsize=4)
    axes[1].plot(summary_df["chunk"], summary_df["roc_auc_mean"], "-o", color="navy", label="ROC-AUC")
    axes[1].set_xlabel("Chunk (temporal order)")
    axes[1].set_ylabel("AUC")
    axes[1].set_title("AUC Metrics Over Time")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "accuracy_decay_plot.png"), dpi=150)
    plt.close()

    print("\n" + "=" * 70)
    print("PHASE C COMPLETE")
    print(f"Summary saved to: {args.out_dir}/baseline_decay_summary.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()