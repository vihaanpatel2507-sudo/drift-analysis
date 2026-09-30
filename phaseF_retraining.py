"""
Phase F: Adaptation — Retraining Strategies vs the Frozen Baseline (SAML-D)

Compares mitigation strategies against the frozen phase C baseline:

  no_retrain        frozen baseline (the phase C/D reference)
  periodic_full     every RETRAIN_EVERY chunks, retrain on ALL labeled data
                    accumulated so far (cumulative window)
  periodic_sliding  same schedule, but train only on the last WINDOW chunks
                    (lets the model FORGET stale patterns)
  periodic_weighted periodic_full with the positive class up-weighted

Two experiments:
  A: the four strategies above, evaluated OUT-OF-SAMPLE — a model trained on
     chunks 1..c-1 is scored on chunk c, never on the data it was fit on. The
     last chunk doubles as the deploy-time "final model" view.
  B: a positive-weight sweep over w in {1, 10, 50, 100} applied on top of the
     pool's natural imbalance ratio (scale_pos_weight = w * neg/pos), training
     on reference + the most recent chunk and evaluating on the next. Each run
     logs its wall-clock fit time and F1 gain per training second, so the CSV
     answers "is retraining worth the compute?".

Every retrained model has its decision threshold re-tuned on a 20% validation
slice of its OWN retraining pool (never on the eval chunk); the frozen
baseline keeps the phase C threshold for fairness.

Outputs (results_phaseF/):
  retraining_results.csv        experiment A: chunk x strategy metrics
  retraining_weight_sweep.csv   experiment B: weight sweep (+gain/second)
  retraining_summary.csv        per-strategy aggregates
  retraining_comparison_plot.png
"""

import os
import time
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

from common import (
    BASELINE_SEEDS, load_chunk, load_baseline_bundle,
    simple_train_test_split, evaluate, tune_threshold,
    add_seed_manifest, list_chunk_files,
)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

DEFAULT_CHUNKS_DIR = "chunks"
DEFAULT_MODEL_PATH = os.path.join("results_phaseC", "baseline_model.pkl")
DEFAULT_OUT_DIR = "results_phaseF"

RETRAIN_EVERY = 2   # retrain cadence (in chunks) for the periodic strategies
WINDOW = 2          # [F2] sliding window size (in chunks)
WEIGHT_BOOST = 10   # boost multiplier for the periodic_weighted strategy
SPW_SWEEP = [1, 10, 50, 100]   # [F1] positive-weight multipliers
RETRAIN_SEED = 42


def fit_xgb(X_train, y_train, seed=RETRAIN_SEED, weight_mult=1.0):
    """Fit an XGBoost model; scale_pos_weight = weight_mult * neg/pos ([F3])."""
    neg = float((y_train == 0).sum())
    pos = float((y_train == 1).sum())
    spw = weight_mult * (neg / max(pos, 1.0))
    model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=spw, eval_metric='aucpr',
        random_state=seed, n_jobs=-1, tree_method='hist')
    t0 = time.time()
    model.fit(X_train, y_train)
    return model, time.time() - t0, spw


def retrain_with_validation(X_pool, y_pool, seed=RETRAIN_SEED, weight_mult=1.0):
    """
    Train on 80% of the labeled pool, tune the threshold on the remaining
    20% (never on the upcoming eval chunk). Returns
    (model, threshold, fit_seconds, spw_used).
    """
    X_tr, X_val, y_tr, y_val = simple_train_test_split(
        X_pool, y_pool, test_size=0.2, seed=seed)
    model, fit_s, spw = fit_xgb(X_tr, y_tr, seed=seed, weight_mult=weight_mult)
    threshold = tune_threshold(y_val, model.predict_proba(X_val)[:, 1])
    return model, threshold, fit_s, spw


def build_pool(encoded, chunk_nums):
    """Concatenate the given chunks into one (X, y) retraining pool."""
    X = pd.concat([encoded[k][0] for k in chunk_nums], axis=0, ignore_index=True)
    y = pd.Series(pd.concat([encoded[k][1] for k in chunk_nums],
                            axis=0, ignore_index=True))
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks_dir", type=str, default=DEFAULT_CHUNKS_DIR)
    ap.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    ap.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    ap.add_argument("--retrain_every", type=int, default=RETRAIN_EVERY)
    ap.add_argument("--window", type=int, default=WINDOW)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 70)
    print("PHASE F: ADAPTATION (retraining strategies, out-of-sample)")
    print(f"retrain_every={args.retrain_every} sliding_window={args.window} "
          f"weighted_boost={WEIGHT_BOOST} sweep={SPW_SWEEP}")
    print("=" * 70)

    bundle = load_baseline_bundle(args.model_path)
    base_model = bundle["model"]
    encoders = bundle.get("encoders")
    base_threshold = bundle.get("threshold", 0.5)

    chunk_files = list_chunk_files(args.chunks_dir)

    # ---- pre-encode all chunks once -------------------------------------
    encoded = {}   # chunk -> (X, y)
    for cnum, cf in chunk_files:
        encoded[cnum] = load_chunk(os.path.join(args.chunks_dir, cf),
                                   encoders=encoders, fit_encoders=False)[:2]
        print(f"  encoded chunk {cnum:02d}: {len(encoded[cnum][1]):,} rows")

    chunk_nums = sorted(encoded)
    ref_chunk = chunk_nums[0]

    # ==================================================================
    # EXPERIMENT A: strategy comparison, evaluated OUT-OF-SAMPLE on chunk c
    # ==================================================================
    print("\n--- Experiment A: retraining strategies ---")
    strategies = ["no_retrain", "periodic_full", "periodic_sliding",
                  "periodic_weighted"]
    current = {s: (base_model, base_threshold) for s in strategies}
    rows_A = []

    for c in chunk_nums[1:]:
        # --- update the adaptive strategies on their schedule -----------
        if (c - 1) % args.retrain_every == 0:
            pool_chunks = chunk_nums[: chunk_nums.index(c)]  # 1..c-1
            X_pool, y_pool = build_pool(encoded, pool_chunks)

            # periodic_full: all labeled history
            mdl, thr, fit_s, _ = retrain_with_validation(X_pool, y_pool)
            current["periodic_full"] = (mdl, thr)
            print(f"  [chunk {c:02d}] retrained periodic_full on "
                  f"{len(X_pool):,} rows ({fit_s:.1f}s)")

            # periodic_sliding: last `window` chunks only
            X_win, y_win = build_pool(encoded, pool_chunks[-args.window:])
            mdl, thr, fit_s, _ = retrain_with_validation(X_win, y_win)
            current["periodic_sliding"] = (mdl, thr)
            print(f"  [chunk {c:02d}] retrained periodic_sliding on "
                  f"{len(X_win):,} rows ({fit_s:.1f}s)")

            # periodic_weighted: full history + up-weighted positives
            mdl, thr, fit_s, _ = retrain_with_validation(
                X_pool, y_pool, weight_mult=WEIGHT_BOOST)
            current["periodic_weighted"] = (mdl, thr)
            print(f"  [chunk {c:02d}] retrained periodic_weighted "
                  f"(boost={WEIGHT_BOOST}) ({fit_s:.1f}s)")

        # --- evaluate every strategy on chunk c (untouched by training) --
        X_c, y_c = encoded[c]
        for s in strategies:
            model, thr = current[s]
            proba = model.predict_proba(X_c)[:, 1]
            pred = (proba >= thr).astype(int)
            m = evaluate(y_c, pred, proba)
            retrained = (s != "no_retrain"
                         and (c - 1) % args.retrain_every == 0)
            rows_A.append({
                "chunk": c, "strategy": s, "threshold": thr,
                "f1": m["f1"], "precision": m["precision"],
                "recall": m["recall"], "roc_auc": m["roc_auc"],
                "pr_auc": m["pr_auc"], "n_positives": m["n_positives"],
                "retrained_this_chunk": bool(retrained),
                "seed": RETRAIN_SEED,
            })
        r = {x["strategy"]: x["f1"] for x in rows_A if x["chunk"] == c}
        print(f"  chunk {c:02d}: F1 no_retrain={r['no_retrain']:.4f} "
              f"full={r['periodic_full']:.4f} "
              f"sliding={r['periodic_sliding']:.4f} "
              f"weighted={r['periodic_weighted']:.4f}")

    results_A = pd.DataFrame(rows_A)
    results_A.to_csv(os.path.join(args.out_dir, "retraining_results.csv"),
                     index=False)

    # ==================================================================
    # EXPERIMENT B: positive-weight sweep, OUT-OF-SAMPLE. For each future
    # chunk c: retrain on reference + chunk c-1 with scale_pos_weight =
    # w * (neg/pos), evaluate on chunk c.
    # ==================================================================
    print("\n--- Experiment B: up-weight sweep ---", flush=True)
    rows_B = []
    sweep_path = os.path.join(args.out_dir, "retraining_weight_sweep.csv")
    frozen_f1 = results_A[results_A["strategy"] == "no_retrain"].set_index("chunk")["f1"]

    for c in chunk_nums[1:]:
        X_pool, y_pool = build_pool(encoded, [ref_chunk, c - 1])
        natural_ratio = (y_pool == 0).sum() / max((y_pool == 1).sum(), 1)
        base_f1 = float(frozen_f1.loc[c])

        for w in SPW_SWEEP:
            print(f"  [sweep] chunk {c:02d} w={w}: fitting...", flush=True)
            model, thr, fit_s, spw = retrain_with_validation(
                X_pool, y_pool, weight_mult=float(w))
            X_c, y_c = encoded[c]
            proba = model.predict_proba(X_c)[:, 1]
            m = evaluate(y_c, (proba >= thr).astype(int), proba)
            gain = m["f1"] - base_f1
            rows_B.append({
                "chunk": c, "weight_mult": w,
                "spw_used": spw,
                "spw_natural_ratio": natural_ratio,
                "f1": m["f1"], "precision": m["precision"],
                "recall": m["recall"], "roc_auc": m["roc_auc"],
                "pr_auc": m["pr_auc"],
                "f1_gain_vs_frozen": gain,
                "fit_seconds": fit_s,
                "f1_gain_per_second": gain / fit_s if fit_s > 0 else np.nan,
                "seed": RETRAIN_SEED,
            })
            # persist progress after every (chunk, weight) pair so a long
            # sweep can be inspected or resumed
            pd.DataFrame(rows_B).to_csv(sweep_path, index=False)
            print(f"  chunk {c:02d} w={w:<3}: F1={m['f1']:.4f} "
                  f"(gain {gain:+.4f}, {fit_s:.1f}s)", flush=True)

    results_B = pd.DataFrame(rows_B)
    results_B.to_csv(sweep_path, index=False)

    # ==================================================================
    # summary tables
    # ==================================================================
    base_f1_by_chunk = results_A[results_A["strategy"] == "no_retrain"].set_index("chunk")["f1"]
    summary_rows = []
    for s in strategies:
        sub = results_A[results_A["strategy"] == s]
        gain = (sub.set_index("chunk")["f1"] - base_f1_by_chunk).dropna()
        summary_rows.append({
            "strategy": s,
            "f1_mean_chunks2_N": sub["f1"].mean(),
            "f1_final_chunk": sub[sub["chunk"] == sub["chunk"].max()]["f1"].iloc[0],
            "mean_gain_vs_frozen": gain.mean(),
            "max_gain_vs_frozen": gain.max(),
            "pr_auc_mean": sub["pr_auc"].mean(),
            "recall_mean": sub["recall"].mean(),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df = add_seed_manifest(summary_df, BASELINE_SEEDS[:1] + [RETRAIN_SEED])
    summary_df.to_csv(os.path.join(args.out_dir, "retraining_summary.csv"),
                      index=False)

    # best weight per chunk (from the sweep)
    best_w = (results_B.loc[results_B.groupby("chunk")["f1"].idxmax()]
              [["chunk", "weight_mult", "f1", "f1_gain_vs_frozen"]])
    print("\nBest weight multiplier per chunk (out-of-sample):")
    print(best_w.to_string(index=False))

    print("\n" + "=" * 70)
    print("STRATEGY SUMMARY (F1, out-of-sample)")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    plot_comparison(results_A, results_B, args.out_dir)
    print(f"\nArtifacts saved in {args.out_dir}/")
    return summary_df, results_A, results_B


def plot_comparison(results_A, results_B, out_dir):
    """Panel 1: F1 per chunk per strategy. Panel 2: weight sweep."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    colors = {"no_retrain": "crimson", "periodic_full": "seagreen",
              "periodic_sliding": "darkorange",
              "periodic_weighted": "steelblue"}
    for s, color in colors.items():
        sub = results_A[results_A["strategy"] == s].sort_values("chunk")
        ax.plot(sub["chunk"], sub["f1"], marker="o", color=color, label=s)
    ax.set_title("Retraining strategies vs frozen baseline")
    ax.legend(fontsize=8)

    ax = axes[1]
    for w in sorted(results_B["weight_mult"].unique()):
        sub = results_B[results_B["weight_mult"] == w].sort_values("chunk")
        ax.plot(sub["chunk"], sub["f1"], marker="s", label=f"w={w:g}")
    ax.set_title("Positive-weight sweep (reference + prev chunk)")
    ax.legend(fontsize=8)

    for ax in axes:
        ax.set_xlabel("Chunk (evaluated out-of-sample)")
        ax.set_ylabel("F1-score")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "retraining_comparison_plot.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()