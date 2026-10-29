"""
Phase D: Unsupervised (label-free) Drift Detection (SAML-D)

Monitors the chunk stream with detectors that need NO labels — exactly the
production constraint (fraud labels arrive weeks late or never).

Four detector families:
  [D1] PSI over derived numeric features (Amount, Amount_log, hour_of_day,
       day_of_week, tx_per_sender, amount_per_sender_mean) rather than raw
       "Amount" alone — a per-feature table is saved so "PSI saw nothing"
       stays inspectable.
  [D2] Output-distribution monitoring: two-sample KS + PSI on the model's
       predicted probabilities. This is what real ML monitoring falls back
       to when labels are missing, since drift that matters shows up as a
       shifted score distribution even when the inputs look stable.
  [D3] Domain classifier: can a model separate reference rows from streaming
       rows? AUC ~0.5 => indistinguishable; clearly above => drift.
  [D4] PSI per categorical feature (currencies, bank locations, payment type),
       saved as a chunks x features pivot next to the numeric one.

NOTE ON LARGE-SAMPLE KS POWER: with N ~ 10^6 rows the alpha=0.05 critical value
falls to ~0.002, so KS behaves as a sensitive hypothesis-rejection test while
output PSI acts as the practical effect-size check.

Outputs (results_phaseD/):
  drift_detector_results.csv   long format: chunk x method x value x drifted
  psi_per_feature.csv          [D1] per numeric feature PSI per chunk
  psi_categorical_pivot.csv    [D4] chunks x categorical-feature PSI
  drift_detectors_plot.png     2x2 panel: PSI-max / KS / domain AUC / PSI(out)
"""

import os
import argparse

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib
matplotlib.use("Agg")
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

# pyrefly: ignore [missing-import]
from xgboost import XGBClassifier

from common import (
    CATEGORICAL_COLS, DETECTOR_SEEDS,
    calculate_psi, calculate_categorical_psi, ks_statistic,
    derive_psi_features, PSI_NUMERIC_FEATURES, load_baseline_bundle,
    load_chunk, simple_train_test_split, roc_auc_manual, add_seed_manifest,
    ci95, list_chunk_files,
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


def build_domain_matrix(df, encoders):
    """
    [D3] Feature view used by the domain classifier: the derived numeric
    monitoring features plus the encoded categoricals. No IDs, no labels, so
    the classifier never sees the target.
    """
    cats = pd.DataFrame(index=df.index)
    for col in CATEGORICAL_COLS:
        if col in df.columns and encoders and col in encoders:
            le = encoders[col]
            vals = le.transform(df[col].astype(str).tolist())
            cats[col] = np.where(vals == -1, le.mapping["__UNSEEN__"], vals)
    return pd.concat([derive_psi_features(df), cats], axis=1)


def run_domain_classifier(ref_x, cur_x, seed):
    """
    [D3] One domain-classifier fit for one seed. The caller passes the
    pre-built feature matrices so the expensive feature engineering is done
    once per chunk pair, not once per seed.
    """
    ref_x = ref_x.sample(min(DOMAIN_SAMPLE, len(ref_x)), random_state=seed)
    cur_x = cur_x.sample(min(DOMAIN_SAMPLE, len(cur_x)), random_state=seed)

    X_dom = pd.concat([ref_x, cur_x], axis=0, ignore_index=True)
    y_dom = pd.Series(np.concatenate([np.zeros(len(ref_x)), np.ones(len(cur_x))]))

    X_tr, X_te, y_tr, y_te = simple_train_test_split(X_dom, y_dom, test_size=0.3, seed=seed)

    clf = XGBClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.1,
        eval_metric='aucpr', random_state=seed, n_jobs=-1, tree_method='hist')
    clf.fit(X_tr, y_tr)
    return roc_auc_manual(y_te, clf.predict_proba(X_te)[:, 1])


def ks_with_effect_size(reference, current, n_subsample=10000, alpha=KS_ALPHA):
    """KS statistic with effect-size reporting via subsampling."""
    from scipy.stats import ks_2samp
    stat = float(ks_2samp(reference, current).statistic)
    crit = ks_critical_value(len(reference), len(current), alpha)
    if len(reference) > n_subsample or len(current) > n_subsample:
        ref_sub = np.random.choice(reference, size=min(n_subsample, len(reference)), replace=False)
        cur_sub = np.random.choice(current, size=min(n_subsample, len(current)), replace=False)
        subs_stat = float(ks_2samp(ref_sub, cur_sub).statistic)
        subs_p = float(ks_2samp(ref_sub, cur_sub).pvalue)
    else:
        subs_stat = stat
        subs_p = float(ks_2samp(reference, current).pvalue)
    return stat, crit, subs_stat, subs_p


def mmd_two_sample_test(reference, current, gamma=1.0, n_permutations=200, seed=42):
    """Maximum Mean Discrepancy two-sample test with permutation null."""
    rng = np.random.default_rng(seed)

    def rbf_kernel(X, Y, gamma):
        XX = np.sum(X**2, axis=1, keepdims=True)
        YY = np.sum(Y**2, axis=1, keepdims=True)
        XY = X @ Y.T
        d2 = XX + YY.T - 2 * XY
        return np.exp(-gamma * d2)

    n_ref = min(2000, len(reference))
    n_cur = min(2000, len(current))
    ref_idx = rng.choice(len(reference), size=n_ref, replace=False)
    cur_idx = rng.choice(len(current), size=n_cur, replace=False)
    X = reference[ref_idx]
    Y = current[cur_idx]

    K_XX = rbf_kernel(X, X, gamma)
    K_YY = rbf_kernel(Y, Y, gamma)
    K_XY = rbf_kernel(X, Y, gamma)

    mmd_stat = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)

    combined = np.concatenate([X, Y])
    perm_stats = []
    n_x = len(X)
    for _ in range(n_permutations):
        perm = rng.permutation(len(combined))
        X_p = combined[perm[:n_x]]
        Y_p = combined[perm[n_x:]]
        K_XX_p = rbf_kernel(X_p, X_p, gamma)
        K_YY_p = rbf_kernel(Y_p, Y_p, gamma)
        K_XY_p = rbf_kernel(X_p, Y_p, gamma)
        perm_stats.append(np.mean(K_XX_p) + np.mean(K_YY_p) - 2 * np.mean(K_XY_p))

    pvalue = float(np.mean(np.array(perm_stats) >= mmd_stat))
    return float(mmd_stat), pvalue


def lof_detector(reference, current, contamination=0.001):
    """Local Outlier Factor detector trained on reference, applied to current."""
    from sklearn.neighbors import LocalOutlierFactor
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=True)
    lof.fit(reference)
    scores = lof.decision_function(current)
    predictions = lof.predict(current)
    outlier_fraction = float(np.mean(predictions == -1))
    return float(np.mean(scores)), outlier_fraction, scores


def run_adwin(errors, delta=0.002):
    """ADWIN drift detector on error stream."""
    from collections import deque
    window = deque()
    total = 0.0
    width = 0
    drift_point = None
    for i, error in enumerate(errors):
        window.append(error)
        total += error
        width += 1
        if width < 2:
            continue
        n1 = width // 2
        if n1 == 0:
            continue
        win1 = list(window)[:n1]
        win2 = list(window)[n1:]
        m1 = sum(win1) / len(win1)
        m2 = sum(win2) / len(win2)
        m = max(len(win1), len(win2))
        bound = np.sqrt(np.log(2.0 / delta) / (2 * m))
        if abs(m1 - m2) > bound:
            drift_point = i
            for _ in range(n1):
                total -= window.popleft()
                width -= 1
            break
    return drift_point is not None, drift_point


def run_ddm(errors, warning_level=2.0, drift_level=3.0):
    """Drift Detection Method (DDM) on error stream."""
    p = 0.0
    p_max = 0.0
    s_max = 0.0
    n = 0
    drift = False
    warning = False
    for error in errors:
        n += 1
        p = p + (error - p) / n
        s = np.sqrt(p * (1 - p) / n)
        if p + s > p_max + s_max:
            p_max = p + s
            s_max = s
        if p + s > p_max + drift_level * s_max:
            drift = True
            p = 0.0
            p_max = 0.0
            s_max = 0.0
            n = 0
        elif p + s > p_max + warning_level * s_max:
            warning = True
    return drift, warning


def inject_synthetic_drift(raw_chunks, drift_magnitude=0.1, drift_type="amount_shift", seed=42):
    """Inject synthetic drift into chunks for detection delay measurement."""
    rng = np.random.default_rng(seed)
    drifted = []
    for cnum, df in raw_chunks:
        df_drifted = df.copy()
        if drift_type == "amount_shift" and "Amount" in df_drifted.columns:
            df_drifted["Amount"] = df_drifted["Amount"] * (1 + drift_magnitude)
        elif drift_type == "label_flip" and "Is_laundering" in df_drifted.columns:
            n_flip = int(len(df_drifted) * drift_magnitude * 0.01)
            flip_idx = rng.choice(len(df_drifted), size=n_flip, replace=False)
            df_drifted.loc[flip_idx, "Is_laundering"] = 1 - df_drifted.loc[flip_idx, "Is_laundering"].values
        drifted.append((cnum, df_drifted))
    return drifted


def measure_detection_delay(detector_results, drift_start_chunk):
    """Measure detection delay for each detector."""
    delays = {}
    for method, detections in detector_results.items():
        delay = None
        for chunk, drifted in detections:
            if drifted and chunk >= drift_start_chunk:
                delay = chunk - drift_start_chunk
                break
        delays[method] = delay
    return delays


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

    bundle = load_baseline_bundle(args.model_path)
    model = bundle["model"]
    encoders = bundle.get("encoders")

    chunk_files = list_chunk_files(args.chunks_dir)

    # ---- pass 1: load everything we need -------------------------------
    raw = {}    # chunk -> raw DataFrame
    feats = {}  # chunk -> derived numeric features  [D1]
    probs = {}  # chunk -> model predicted probabilities [D2]
    for cnum, cf in chunk_files:
        path = os.path.join(args.chunks_dir, cf)
        raw[cnum] = pd.read_csv(path)
        feats[cnum] = derive_psi_features(raw[cnum])
        X, _, _ = load_chunk(path, encoders=encoders, fit_encoders=False)
        probs[cnum] = model.predict_proba(X)[:, 1]
        print(f"  loaded chunk {cnum:02d}: {len(raw[cnum]):,} rows")

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
    ref_domain = build_domain_matrix(raw[ref_chunk], encoders)
    dom_rows = []
    for cnum in sorted(raw):
        if cnum == ref_chunk:
            continue
        # feature engineering happens once per chunk pair, not once per seed
        cur_domain = build_domain_matrix(raw[cnum], encoders)
        aucs = np.array([run_domain_classifier(ref_domain, cur_domain, s)
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
        del cur_domain

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
    for ax in axes.flat:
        ax.set_xlabel("Chunk")
        ax.grid(alpha=0.3)

    # (1) PSI over numeric features (max across features)
    ax = axes[0][0]
    ax.bar(psi_summary["chunk"], psi_summary["psi_max"], color="steelblue")
    ax.axhline(PSI_THRESHOLD, color="red", linestyle="--", linewidth=1,
               label=f"PSI threshold ({PSI_THRESHOLD})")
    ax.set_title("[D1] PSI - max over derived numeric features")
    ax.set_ylabel("PSI (max)")
    ax.legend()

    # (2) KS on predicted probabilities
    ks_rows = drift_df[drift_df["method"] == "KS(predicted probabilities)"]
    ax = axes[0][1]
    ax.bar(ks_rows["chunk"], ks_rows["value"], color="darkorange")
    ax.plot(ks_rows["chunk"], ks_rows["threshold"], "r--", linewidth=1,
            label="KS critical value (alpha=0.05)")
    ax.set_title("[D2] KS on predicted probabilities")
    ax.set_ylabel("KS statistic")
    ax.legend()

    # (3) PSI on predicted probabilities
    psi_out = drift_df[drift_df["method"] == "PSI(predicted probabilities)"]
    ax = axes[1][0]
    ax.bar(psi_out["chunk"], psi_out["value"], color="seagreen")
    ax.axhline(PSI_THRESHOLD, color="red", linestyle="--", linewidth=1)
    ax.set_title("[D2] PSI on predicted probabilities")
    ax.set_ylabel("PSI (output scores)")

    # (4) domain classifier AUC with CI
    dom = drift_df[drift_df["method"] == "DomainClassifier(XGB)"]
    ax = axes[1][1]
    ax.bar(dom["chunk"], dom["value"], yerr=dom["auc_ci95"].fillna(0),
           capsize=3, color="indianred")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1,
               label="coin flip (0.5)")
    ax.axhline(DOMAIN_AUC_THRESHOLD, color="red", linestyle="--", linewidth=1,
               label=f"drift threshold ({DOMAIN_AUC_THRESHOLD})")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("[D3] Domain classifier AUC (mean +/- 95% CI, 3 seeds)")
    ax.set_ylabel("AUC")
    ax.legend()

    plt.suptitle("Label-free drift detection (input + output monitoring)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "drift_detectors_plot.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()