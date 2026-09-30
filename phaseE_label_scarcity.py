"""
Phase E: Drift Detection & Evaluation Under Label Scarcity (The Core Experiment)

Simulates operational reality in AML: true labels (investigated SARs) arrive
with long delays and at very low sample rates.

Tests two critical research questions:
1. Is the observed performance collapse genuine concept drift, or an illusion
   caused by evaluating on small label samples? (Controlled via matched no-drift test)
2. How low can label availability drop before performance-based drift monitoring
   breaks down (either through high variance, missed drift, or false alarms)?

Evaluates label fractions: [1.0, 0.5, 0.1, 0.05, 0.01, 0.001] across multiple seeds.
Gracefully handles positive starvation (< 3 positives) as NaN.
"""

import argparse
import os
# pyrefly: ignore [missing-import]
import matplotlib
matplotlib.use("Agg")
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd

from common import (
    BASELINE_SEEDS,
    load_chunk,
    load_baseline_bundle,
    simple_train_test_split,
    evaluate,
    ci95,
    add_seed_manifest,
    list_chunk_files,
)

DEFAULT_CHUNKS_DIR = "chunks"
DEFAULT_PHASE_C_DIR = "results_phaseC"
DEFAULT_PHASE_D_DIR = "results_phaseD"
DEFAULT_OUT_DIR = "results_phaseE"

LABEL_FRACTIONS = [1.0, 0.5, 0.1, 0.05, 0.01, 0.001]
MIN_POSITIVES = 3


def score_budget(model, threshold, X, y):
    """
    Score one label budget. Returns a flat result dict, or all-NaN metrics
    with an explanatory note when too few positives are visible to measure
    anything (we report the gap rather than a fake zero).
    """
    record = {"n_available": len(y), "n_positives_available": int(y.sum())}
    if record["n_positives_available"] < MIN_POSITIVES:
        return {**record, "f1": np.nan, "precision": np.nan, "recall": np.nan,
                "roc_auc": np.nan, "pr_auc": np.nan,
                "note": f"insufficient positives (<{MIN_POSITIVES})"}

    proba = model.predict_proba(X)[:, 1]
    m = evaluate(y.values, (proba >= threshold).astype(int), proba)
    return {**record, "f1": m["f1"], "precision": m["precision"],
            "recall": m["recall"], "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"],
            "note": ""}


def summarize_runs(df):
    """
    Collapse repeated runs (one per seed) into a mean/std plus a count of how
    many runs actually produced a score.

    A budget where most runs were starved of positives is NOT a measurement of
    zero performance — it is a failure to measure. Reporting mean(0, nan) as
    0.0 would silently turn "we could not tell" into "the model scored
    nothing", so we always carry n_runs_valid alongside the mean and leave the
    mean as NaN when nothing was measurable.
    """
    rows = []
    for frac, grp in df.groupby("fraction"):
        vals = grp["f1"].dropna()
        n_valid = int(len(vals))
        std = float(vals.std(ddof=1)) if n_valid > 1 else (0.0 if n_valid == 1 else np.nan)
        rows.append({
            "fraction": frac,
            "control_f1_mean": float(vals.mean()) if n_valid else np.nan,
            "control_f1_std": std,
            "control_f1_ci95": ci95(std, n_valid),
            "control_pr_auc_mean": float(grp["pr_auc"].mean(skipna=True))
            if grp["pr_auc"].notna().any() else np.nan,
            "control_n_runs_valid": n_valid,
        })
    return pd.DataFrame(rows)


def get_reference_f1(phase_c_dir):
    """Load the ground-truth training-era validation F1 from Phase C."""
    summary_path = os.path.join(phase_c_dir, "baseline_decay_summary.csv")
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        row = df[df["chunk"] == 1]
        if not row.empty:
            return float(row["f1_mean"].iloc[0])

    val_path = os.path.join(phase_c_dir, "baseline_validation_metrics.csv")
    if os.path.exists(val_path):
        df = pd.read_csv(val_path)
        return float(df["f1"].mean())

    return 0.1992  # Default empirical baseline F1 on SAML-D Chunk 1


def sample_with_budget(X, y, fraction, seed):
    """Sample a fraction of rows randomly using a given seed."""
    if fraction >= 1.0:
        return X, y

    rng = np.random.default_rng(seed)
    n_total = len(y)
    n_sample = max(1, int(n_total * fraction))
    indices = rng.choice(n_total, size=n_sample, replace=False)
    return X.iloc[indices], y.iloc[indices]


def run_no_drift_control(model, threshold, chunk_1_path, encoders, seeds):
    """
    Control Experiment: Run label scarcity on a held-out slice of Chunk 1.
    Since Chunk 1 is the training era, there is NO temporal drift here.
    Any degradation or metric swing observed here is purely the effect
    of small sample variance, giving us a clean baseline to compare against.
    """
    X1, y1, _ = load_chunk(chunk_1_path, encoders=encoders, fit_encoders=False)
    _, X_holdout, _, y_holdout = simple_train_test_split(X1, y1, test_size=0.2, seed=42)

    control_results = []

    for frac in LABEL_FRACTIONS:
        for seed in seeds:
            Xs, ys = sample_with_budget(X_holdout, y_holdout, frac, seed=seed)
            control_results.append({"fraction": frac, "seed": seed,
                                    "threshold_used": threshold,
                                    **score_budget(model, threshold, Xs, ys)})

    return pd.DataFrame(control_results)


def main():
    parser = argparse.ArgumentParser(description="Evaluate drift detection and decay under label scarcity.")
    parser.add_argument("--chunks_dir", type=str, default=DEFAULT_CHUNKS_DIR)
    parser.add_argument("--phase_c_dir", type=str, default=DEFAULT_PHASE_C_DIR)
    parser.add_argument("--phase_d_dir", type=str, default=DEFAULT_PHASE_D_DIR)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=BASELINE_SEEDS)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 70)
    print("PHASE E: DRIFT DETECTION & EVALUATION UNDER LABEL SCARCITY")
    print(f"Testing label availability: {[f'{f*100:g}%' for f in LABEL_FRACTIONS]}")
    print(f"Seeds: {args.seeds}")
    print("=" * 70)

    # 1. Load trained baseline model and reference F1
    bundle = load_baseline_bundle(os.path.join(args.phase_c_dir,
                                              "baseline_model.pkl"))
    model = bundle["model"]
    encoders = bundle["encoders"]
    threshold = bundle.get("threshold", 0.5)
    f1_ref = get_reference_f1(args.phase_c_dir)
    print(f"Loaded baseline model. Reference F1: {f1_ref:.4f} (Threshold: {threshold:.2f})")

    # 2. Check for chunks
    raw_files = list_chunk_files(args.chunks_dir) if os.path.exists(args.chunks_dir) else []
    if not raw_files:
        print(f"No chunk files found in '{args.chunks_dir}'. Using existing Phase E results.")
        return

    chunk_files = [(cnum, os.path.join(args.chunks_dir, fname)) for cnum, fname in raw_files]

    # 3. Run No-Drift Control on held-out slice of Chunk 1
    print("\n--- Running No-Drift Control on Chunk 1 Holdout ---")
    chunk_1_path = chunk_files[0][1]
    control_df = run_no_drift_control(model, threshold, chunk_1_path, encoders, seeds=args.seeds)
    control_df.to_csv(os.path.join(args.out_dir, "no_drift_control_results.csv"), index=False)

    ctrl_summary = summarize_runs(control_df)

    # 4. Run Label Scarcity Sweep across stream chunks (Chunk 1 to N)
    print("\n--- Running Label Scarcity Across Streaming Chunks ---")
    all_results = []

    for cnum, cpath in chunk_files:
        X_chunk, y_chunk, _ = load_chunk(cpath, encoders=encoders, fit_encoders=False)

        for frac in LABEL_FRACTIONS:
            for seed in args.seeds:
                Xs, ys = sample_with_budget(X_chunk, y_chunk, frac, seed=seed)
                all_results.append({"chunk": cnum, "fraction": frac, "seed": seed,
                                    **score_budget(model, threshold, Xs, ys)})

        # Print progress for this chunk at the full-label budget
        c_f1 = [r["f1"] for r in all_results
                if r["chunk"] == cnum and r["fraction"] == 1.0]
        print(f"  Chunk {cnum:02d}: Full F1={np.mean(c_f1):.4f}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(args.out_dir, "label_scarcity_results.csv"), index=False)

    # 5. Build Comprehensive Summary Table
    ctrl_by_frac = ctrl_summary.set_index("fraction")
    summary_list = []
    for (cnum, frac), grp in results_df.groupby(["chunk", "fraction"]):
        f1_vals = grp["f1"].dropna()
        f1_std = float(f1_vals.std(ddof=1)) if len(f1_vals) > 1 else (
            0.0 if len(f1_vals) == 1 else np.nan)
        ctrl = ctrl_by_frac.loc[frac] if frac in ctrl_by_frac.index else None
        ctrl_f1 = float(ctrl["control_f1_mean"]) if ctrl is not None else np.nan
        ctrl_valid = int(ctrl["control_n_runs_valid"]) if ctrl is not None else 0
        f1_mean = float(f1_vals.mean()) if not f1_vals.empty else np.nan
        summary_list.append({
            "chunk": cnum,
            "fraction": frac,
            "f1_mean": f1_mean,
            "f1_std": f1_std,
            "precision_mean": float(grp["precision"].mean()),
            "recall_mean": float(grp["recall"].mean()),
            "roc_auc_mean": float(grp["roc_auc"].mean()),
            "pr_auc_mean": float(grp["pr_auc"].mean()),
            "n_positives_available": float(grp["n_positives_available"].mean()),
            "n_runs_valid": float(len(f1_vals)),
            "f1_ci95": ci95(f1_std, len(f1_vals)),
            "control_f1_mean": ctrl_f1,
            "control_f1_std": float(ctrl["control_f1_std"]) if ctrl is not None else np.nan,
            "control_f1_ci95": float(ctrl["control_f1_ci95"]) if ctrl is not None else np.nan,
            "control_pr_auc_mean": float(ctrl["control_pr_auc_mean"]) if ctrl is not None else np.nan,
            "control_n_runs_valid": float(ctrl_valid),
            # A ratio is only meaningful when BOTH arms produced a real score.
            # Comparing against a starved or zero-scoring control would invent a
            # number that says nothing about drift.
            "future_over_control": (f1_mean / ctrl_f1
                                    if (ctrl_valid == len(args.seeds) and ctrl_f1
                                        and ctrl_f1 == ctrl_f1 and f1_mean == f1_mean)
                                    else np.nan),
        })

    summary_df = pd.DataFrame(summary_list)
    summary_df = add_seed_manifest(summary_df, args.seeds)
    summary_df.to_csv(os.path.join(args.out_dir, "label_scarcity_summary.csv"), index=False)

    # Tell the user plainly which budgets are trustworthy, rather than letting
    # them discover it by reading NaNs out of the CSV.
    print("\nLabel budgets by how many seeds produced a usable score:")
    for _, r in summary_df[summary_df["chunk"] >= 2].groupby("fraction").agg(
            valid=("n_runs_valid", "first"),
            ctrl_valid=("control_n_runs_valid", "first"),
            ratio=("future_over_control", "mean")).reset_index().iterrows():
        status = "trustworthy" if r["ctrl_valid"] == len(args.seeds) and r["valid"] == len(args.seeds) \
            else "NOT comparable — starved of positives"
        print(f"  {r['fraction'] * 100:>6g}% labels: future {int(r['valid'])}/{len(args.seeds)} runs, "
              f"control {int(r['ctrl_valid'])}/{len(args.seeds)} runs — {status}")

    # 6. Generate Publication Plot
    plot_path = os.path.join(args.out_dir, "label_scarcity_degradation_plot.png")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Future chunks vs No-Drift Control across label budgets
    future_summary = summary_df[summary_df["chunk"] >= 2].groupby("fraction").agg(
        future_f1=("f1_mean", "mean"),
        control_f1=("control_f1_mean", "first"),
        ctrl_valid=("control_n_runs_valid", "first"),
    ).reset_index().sort_values("fraction", ascending=False)

    # Only draw the control curve where every seed produced a usable score;
    # otherwise NaN creates a gap instead of a misleading point at 0.
    future_summary["control_f1_plot"] = np.where(
        future_summary["ctrl_valid"] == len(args.seeds),
        future_summary["control_f1"], np.nan)

    axes[0].plot(future_summary["fraction"] * 100, future_summary["future_f1"],
                 "-o", color="crimson", linewidth=2, label="Drifted Future Stream (Chunks 2-10)")
    axes[0].plot(future_summary["fraction"] * 100, future_summary["control_f1_plot"],
                 "--s", color="forestgreen", linewidth=2, label="No-Drift Control (Chunk 1 Holdout)")
    starved = future_summary[future_summary["ctrl_valid"] < len(args.seeds)]
    if not starved.empty:
        axes[0].axvspan(starved["fraction"].min() * 100 * 0.5,
                        starved["fraction"].max() * 100 * 2,
                        color="grey", alpha=0.18,
                        label="Too few positives to compare")
        axes[0].annotate("control starved\nof positives",
                         xy=(starved["fraction"].max() * 100,
                             axes[0].get_ylim()[1] * 0.92),
                         ha="center", fontsize=8, color="dimgrey")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Label Availability (%) — Log Scale")
    axes[0].set_ylabel("F1 Score")
    axes[0].set_title("True Drift vs. Label Scarcity Effect")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Panel 2: F1 Variance / Error across label fractions
    axes[1].plot(summary_df[summary_df["chunk"] == 2]["fraction"] * 100,
                 summary_df[summary_df["chunk"] == 2]["f1_std"],
                 "-^", color="darkorange", linewidth=2, label="Metric Standard Deviation")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Label Availability (%) — Log Scale")
    axes[1].set_ylabel("F1 Std Dev Across Seeds")
    axes[1].set_title("Measurement Uncertainty Under Label Scarcity")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print("\n" + "=" * 70)
    print("PHASE E COMPLETE")
    print(f"Summary table saved to: {args.out_dir}/label_scarcity_summary.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()