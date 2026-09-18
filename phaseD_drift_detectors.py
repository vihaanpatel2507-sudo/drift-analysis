"""
Phase D: Unsupervised (label-free) Drift Detection (SAML-D)

Monitors the chunk stream with detectors that need NO labels — exactly the
production constraint (fraud labels arrive weeks late or never).

Improvements implemented here (see improvements.txt):
  [D1] HIGH  PSI is computed over a RICH SET OF DERIVED NUMERIC FEATURES,
             not just raw "Amount": Amount, Amount_log, hour_of_day,
             day_of_week, tx_per_sender, amount_per_sender_mean (all
             leakage-free, computable within a single chunk). A per-feature
             PSI table is saved so "PSI saw nothing" is inspectable.
  [D2] HIGH  A fourth, fully label-free detector family is added:
             OUTPUT-DISTRIBUTION MONITORING. It compares the distribution of
             the model's predicted fraud probabilities between the reference
             chunk and each new chunk (two-sample KS + PSI on the scores).
             This is what real ML monitoring does when labels are missing,
             because drift that matters shows up as a shifted prediction
             distribution even when inputs look stable.
  [D3] MEDIUM The domain classifier is now more robust: 3 seeds
             (DETECTOR_SEEDS), mean +/- 95% CI reported, and a transparent
             decision rule (AUC > 0.55 => drift signal, i.e. clearly better
             than the 0.5 coin-flip floor).
  [D4] MEDIUM PSI is also computed per CATEGORICAL feature (payment
             currency, payment type, bank locations...) and saved as a
             pivot table (chunks x features) next to the numeric one.
  + The KS detector uses the proper two-sample critical value
    1.358 * sqrt((n1+n2)/(n1*n2)) (alpha=0.05) instead of a guess.
  + All detectors report a value AND an explicit threshold AND a boolean
    'drifted' flag in one long-format CSV -> easy to consume downstream.

Outputs (results_phaseD/):
  drift_detector_results.csv   long format: chunk x method x value x drifted
  psi_per_feature.csv          [D1] per numeric feature PSI per chunk
  psi_categorical_pivot.csv    [D4] chunks x categorical-feature PSI
  drift_detectors_plot.png     2x2 panel: PSI-max / KS / domain AUC / PSI(out)
"""

import os
import argparse
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

from common import (
    CATEGORICAL_COLS, DETECTOR_SEEDS,
    calculate_psi, calculate_categorical_psi, ks_statistic,
    derive_psi_features, PSI_NUMERIC_FEATURES, load_chunk,
    simple_train_test_split, roc_auc_manual, add_seed_manifest, ci95,
    list_chunk_files,
)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

DEFAULT_CHUNKS_DIR = "chunks"
DEFAULT_MODEL_PATH = os.path.join("results_phaseC", "baseline_model.pkl")
DEFAULT_OUT_DIR = "results_phaseD"

PSI_THRESHOLD = 0.2          # standard "major distribution change" cut-off
DOMAIN_AUC_THRESHOLD = 0.55  # clearly better than the 0.5 coin-flip floor
KS_ALPHA = 0.05
DOMAIN_SAMPLE = 15000        # per-class cap for the domain classifier


def load_baseline(model_path=DEFAULT_MODEL_PATH):
    """Load the phase C bundle (model + encoders + threshold)."""
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    if isinstance(bundle, dict) and "model" in bundle:
        return bundle
    # very old format: the raw model object was pickled directly
    return {"model": bundle, "encoders": None, "threshold": 0.5}


def ks_critical_value(n_ref, n_cur, alpha=KS_ALPHA):
    """
    Two-sample KS critical value at alpha=0.05.
    
    NOTE ON LARGE-SAMPLE POWER (important for research paper):
    With N ~ 10^6 rows, the critical value drops to ~0.00197. At this scale,
    almost any microscopic difference reaches statistical significance (p < 0.05).
    Hence, KS acts as a sensitive hypothesis rejection test, while output PSI
    serves as a practical effect-size check.
    """
    c_alpha = np.sqrt(-0.5 * np.log(alpha / 2.0))  # 1.358 at alpha=0.05
    return c_alpha * np.sqrt((n_ref + n_cur) / (n_ref * n_cur))


def encode_categoricals(df, encoders):
    """Encode the categorical columns exactly like phase C training did."""
    out = pd.DataFrame(index=df.index)
    for col in CATEGORICAL_COLS:
        if col in df.columns and encoders and col in encoders:
            le = encoders[col]
            vals = le.transform(df[col].astype(str).tolist())
            unseen_id = le.mapping["__UNSEEN__"]
            out[col] = np.where(vals == -1, unseen_id, vals)
    return out


def run_psi_numeric(ref_feats, cur_feats, chunk_num):
    """[D1] PSI for every derived numeric feature of one chunk vs reference."""
    rows = []
    for feat in PSI_NUMERIC_FEATURES:
        if feat not in ref_feats.columns:
            continue
        psi = calculate_psi(ref_feats[feat].values, cur_feats[feat].values)
        rows.append({
            "chunk": chunk_num,
            "feature": feat,
            "psi": psi,
            "drifted": bool(psi > PSI_THRESHOLD),
            "threshold": PSI_THRESHOLD,
        })
    return rows


def run_output_distribution(ref_probs, cur_probs, chunk_num):
    """
    [D2] Label-free OUTPUT monitoring: KS + PSI on predicted probabilities.
    Returns two result rows (one per statistic).
    """
    ks = ks_statistic(ref_probs, cur_probs)
    crit = ks_critical_value(len(ref_probs), len(cur_probs))
    psi_out = calculate_psi(ref_probs, cur_probs, bins=20)
    return [
        {"chunk": chunk_num, "method": "KS(predicted probabilities)",
         "value": ks, "threshold": crit, "drifted": bool(ks > crit)},
        {"chunk": chunk_num, "method": "PSI(predicted probabilities)",
         "value": psi_out, "threshold": PSI_THRESHOLD,
         "drifted": bool(psi_out > PSI_THRESHOLD)},
    ]


def run_domain_classifier(ref_df, cur_df, encoders, seed):
    """
    [D3] One domain-classifier fit for one seed: can a model tell reference
    rows from current-chunk rows? AUC ~ 0.5 => indistinguishable (no drift);
    AUC clearly > 0.5 => the two distributions differ (drift).
    Uses monitored numeric features + encoded categoricals (no IDs, no labels).
    """
    ref_feats = derive_psi_features(ref_df)
    cur_feats = derive_psi_features(cur_df)
    ref_cats = encode_categoricals(ref_df, encoders)
    cur_cats = encode_categoricals(cur_df, encoders)

    ref_x = pd.concat([ref_feats, ref_cats], axis=1)
    cur_x = pd.concat([cur_feats, cur_cats], axis=1)
    feature_cols = list(ref_x.columns)

    ref_x = ref_x.sample(min(DOMAIN_SAMPLE, len(ref_x)), random_state=seed)
    cur_x = cur_x.sample(min(DOMAIN_SAMPLE, len(cur_x)), random_state=seed)

    X_dom = pd.concat([ref_x, cur_x], axis=0, ignore_index=True)
    y_dom = pd.Series(np.concatenate([np.zeros(len(ref_x)), np.ones(len(cur_x))]))

    X_tr, X_te, y_tr, y_te = simple_train_test_split(
        X_dom[feature_cols], y_dom, test_size=0.3, seed=seed)

    clf = XGBClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.1,
        eval_metric='aucpr', random_state=seed, n_jobs=-1, tree_method='hist')
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    return roc_auc_manual(y_te, proba)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks_dir", type=str, default=DEFAULT_CHUNKS_DIR)
    ap.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    ap.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 70)
    print("PHASE D: LABEL-FREE DRIFT DETECTION (input + output monitoring)")
    print(f"seeds for domain classifier: {DETECTOR_SEEDS}")
    print("=" * 70)

    bundle = load_baseline(args.model_path)
    model = bundle["model"]
    encoders = bundle.get("encoders")

    chunk_files = list_chunk_files(args.chunks_dir)

    # ---- pass 1: load everything we need -------------------------------
    raw = {}    # chunk -> raw DataFrame
    feats = {}  # chunk -> derived numeric features  [D1]
    probs = {}  # chunk -> model predicted probabilities [D2]
    for cnum, cf in chunk_files:
        df = pd.read_csv(os.path.join(args.chunks_dir, cf))
        raw[cnum] = df
        feats[cnum] = derive_psi_features(df)
        X, _, _ = load_chunk(os.path.join(args.chunks_dir, cf),
                             encoders=encoders, fit_encoders=False)
        probs[cnum] = model.predict_proba(X)[:, 1]
        print(f"  loaded chunk {cnum:02d}: {len(df):,} rows")

    ref_chunk = min(raw.keys())
    print(f"\nReference chunk = {ref_chunk} (training era). "
          f"Monitoring chunks {sorted(set(raw) - {ref_chunk})}")

    # ---- [D1] PSI over derived numeric features -------------------------
    print("\n[D1] PSI over derived numeric features:", PSI_NUMERIC_FEATURES)
    psi_rows = []
    for cnum in sorted(raw):
        if cnum == ref_chunk:
            continue
        psi_rows.extend(run_psi_numeric(feats[ref_chunk], feats[cnum], cnum))
    psi_feature_df = pd.DataFrame(psi_rows)
    psi_feature_df.to_csv(os.path.join(args.out_dir, "psi_per_feature.csv"),
                          index=False)

    psi_summary = (psi_feature_df.groupby("chunk")
                   .agg(psi_max=("psi", "max"),
                        psi_mean=("psi", "mean"),
                        n_features_drifted=("drifted", "sum"))
                   .reset_index())
    psi_summary["drifted"] = psi_summary["n_features_drifted"] > 0
    print(psi_summary.to_string(index=False))

    # ---- [D4] categorical PSI pivot --------------------------------------
    print("\n[D4] PSI over categorical features:")
    cat_rows = []
    for cnum in sorted(raw):
        if cnum == ref_chunk:
            continue
        for col in CATEGORICAL_COLS:
            if col in raw[cnum].columns:
                psi = calculate_categorical_psi(raw[ref_chunk][col],
                                                raw[cnum][col])
                cat_rows.append({"chunk": cnum, "feature": col, "psi": psi})
    cat_df = pd.DataFrame(cat_rows)
    if not cat_df.empty:
        cat_pivot = cat_df.pivot(index="chunk", columns="feature",
                                 values="psi")
        cat_pivot.to_csv(os.path.join(args.out_dir,
                                      "psi_categorical_pivot.csv"))
        cat_worst = cat_df.loc[cat_df.groupby("chunk")["psi"].idxmax()]
        print(cat_worst.to_string(index=False))

    # ---- [D2] output-distribution monitoring -----------------------------
    print("\n[D2] Output-distribution monitoring (predicted probabilities):")
    out_rows = []
    for cnum in sorted(raw):
        if cnum == ref_chunk:
            continue
        rows = run_output_distribution(probs[ref_chunk], probs[cnum], cnum)
        out_rows.extend(rows)
        for r in rows:
            print(f"  chunk {cnum:02d} {r['method']}: value={r['value']:.4f} "
                  f"thr={r['threshold']:.4f} drifted={r['drifted']}")

    # ---- [D3] domain classifier over seeds --------------------------------
    print(f"\n[D3] Domain classifier ({len(DETECTOR_SEEDS)} seeds):")
    dom_rows = []
    for cnum in sorted(raw):
        if cnum == ref_chunk:
            continue
        aucs = np.array([run_domain_classifier(raw[ref_chunk], raw[cnum],
                                               encoders, s)
                         for s in DETECTOR_SEEDS], dtype=float)
        std = float(aucs.std(ddof=1)) if len(aucs) > 1 else np.nan
        dom_rows.append({
            "chunk": cnum,
            "method": "DomainClassifier(XGB)",
            "value": float(aucs.mean()),
            "threshold": DOMAIN_AUC_THRESHOLD,
            "drifted": bool(aucs.mean() > DOMAIN_AUC_THRESHOLD),
            "auc_std": std,
            "auc_ci95": ci95(std, len(aucs)),
        })
        print(f"  chunk {cnum:02d}: AUC={aucs.mean():.3f} "
              f"(+/- {0.0 if np.isnan(std) else std:.3f}) "
              f"drifted={aucs.mean() > DOMAIN_AUC_THRESHOLD}")

    # ---- assemble the long-format results CSV ----------------------------
    summary_rows = []
    for _, r in psi_summary.iterrows():
        summary_rows.append({"chunk": int(r["chunk"]),
                             "method": "PSI(6 numeric features)",
                             "value": r["psi_max"],
                             "threshold": PSI_THRESHOLD,
                             "drifted": bool(r["drifted"]),
                             "detail": f"{int(r['n_features_drifted'])}/6 features over {PSI_THRESHOLD}"})
    if not cat_df.empty:
        for _, r in cat_worst.iterrows():
            summary_rows.append({"chunk": int(r["chunk"]),
                                 "method": "PSI(categorical, worst feature)",
                                 "value": r["psi"], "threshold": PSI_THRESHOLD,
                                 "drifted": bool(r["psi"] > PSI_THRESHOLD),
                                 "detail": f"worst={r['feature']}"})
    for r in out_rows:
        summary_rows.append({**r, "detail": ""})
    for r in dom_rows:
        summary_rows.append({**r, "detail": f"auc_std={r['auc_std']}"})

    drift_df = pd.DataFrame(summary_rows)
    drift_df = add_seed_manifest(drift_df, DETECTOR_SEEDS)
    drift_df.to_csv(os.path.join(args.out_dir, "drift_detector_results.csv"),
                    index=False)

    # ---- decision table ---------------------------------------------------
    print("\n" + "=" * 70)
    print("DRIFT DECISION TABLE (label-free detectors)")
    print("=" * 70)
    for cnum in sorted(set(drift_df["chunk"])):
        sub = drift_df[drift_df["chunk"] == cnum]
        fired = sub[sub["drifted"]]["method"].tolist()
        verdict = "DRIFT" if fired else "no drift"
        print(f"  chunk {cnum:02d}: {verdict:<10} "
              f"fired: {fired if fired else '-'}")

    plot_panels(psi_summary, drift_df, args.out_dir)
    print(f"\nArtifacts saved in {args.out_dir}/")
    return drift_df


def plot_panels(psi_summary, drift_df, out_dir):
    """2x2 panel: PSI-max, KS on outputs, PSI on outputs, domain AUC."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (1) PSI over numeric features (max across features)
    ax = axes[0][0]
    ax.bar(psi_summary["chunk"], psi_summary["psi_max"], color="steelblue")
    ax.axhline(PSI_THRESHOLD, color="red", linestyle="--", linewidth=1,
               label=f"PSI threshold ({PSI_THRESHOLD})")
    ax.set_title("[D1] PSI - max over derived numeric features")
    ax.set_xlabel("Chunk"); ax.set_ylabel("PSI (max)")
    ax.legend(); ax.grid(alpha=0.3)

    # (2) KS on predicted probabilities
    ks_rows = drift_df[drift_df["method"] == "KS(predicted probabilities)"]
    ax = axes[0][1]
    ax.bar(ks_rows["chunk"], ks_rows["value"], color="darkorange")
    ax.plot(ks_rows["chunk"], ks_rows["threshold"], "r--", linewidth=1,
            label="KS critical value (alpha=0.05)")
    ax.set_title("[D2] KS on predicted probabilities")
    ax.set_xlabel("Chunk"); ax.set_ylabel("KS statistic")
    ax.legend(); ax.grid(alpha=0.3)

    # (3) PSI on predicted probabilities
    psi_out = drift_df[drift_df["method"] == "PSI(predicted probabilities)"]
    ax = axes[1][0]
    ax.bar(psi_out["chunk"], psi_out["value"], color="seagreen")
    ax.axhline(PSI_THRESHOLD, color="red", linestyle="--", linewidth=1)
    ax.set_title("[D2] PSI on predicted probabilities")
    ax.set_xlabel("Chunk"); ax.set_ylabel("PSI (output scores)")
    ax.grid(alpha=0.3)

    # (4) domain classifier AUC with CI
    dom = drift_df[drift_df["method"] == "DomainClassifier(XGB)"]
    ax = axes[1][1]
    ci = dom["auc_ci95"].fillna(0).values if "auc_ci95" in dom else np.zeros(len(dom))
    ax.bar(dom["chunk"], dom["value"], yerr=ci, capsize=3, color="indianred")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1,
               label="coin flip (0.5)")
    ax.axhline(DOMAIN_AUC_THRESHOLD, color="red", linestyle="--", linewidth=1,
               label=f"drift threshold ({DOMAIN_AUC_THRESHOLD})")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("[D3] Domain classifier AUC (mean +/- 95% CI, 3 seeds)")
    ax.set_xlabel("Chunk"); ax.set_ylabel("AUC")
    ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle("Label-free drift detection (input + output monitoring)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "drift_detectors_plot.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()