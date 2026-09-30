"""
common.py — Shared utilities for the AML Drift Detection project (SAML-D).

Single source of truth for everything that used to be copy-pasted across
phaseC/D/E/F: the label encoder, chunk loading, the deterministic stratified
split, evaluation metrics, PSI, the KS statistic, timestamp parsing and the
decision-threshold tuner.

Metrics are provided by scikit-learn (ROC-AUC, Average Precision, F1) and
scipy (two-sample KS test). PSI remains a custom implementation as there is
no standard library equivalent.

PICKLE COMPATIBILITY (important):
    results_phaseC/baseline_model.pkl stores SimpleLabelEncoder instances.
    Pickle stores only the class *name + module*, so every phase must have
    this exact class importable. Doing `from common import SimpleLabelEncoder`
    in each phase satisfies both cases:
      - NEW pickles: saved as "common.SimpleLabelEncoder" -> resolvable.
      - OLD pickles: phaseC ran as a script, so they were saved as
        "__main__.SimpleLabelEncoder" -> when phaseD/E/F run as scripts, the
        `from common import SimpleLabelEncoder` line puts the class into the
        script's namespace, i.e. into "__main__", so old pickles still load.
"""

import os
import pickle
import re

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score as sklearn_f1_score,
    roc_auc_score,
    average_precision_score,
)
# pyrefly: ignore [missing-import]
from scipy.stats import ks_2samp

# -----------------------------------------------------------------------
# Shared configuration (MUST stay identical across all phases)
# -----------------------------------------------------------------------
LABEL_COL = "Is_laundering"
LEAKAGE_COLS = ["Laundering_type"]           # direct giveaway -> never a feature
ID_COLS = ["Sender_account", "Receiver_account"]
RAW_TIME_COLS = ["Time", "Date"]
TIMESTAMP_COL = "__timestamp__"
CATEGORICAL_COLS = [
    "Payment_currency", "Received_currency",
    "Sender_bank_location", "Receiver_bank_location", "Payment_type",
]

# Feature-drop list used by every phase, in one place.
DROP_COLS = LEAKAGE_COLS + ID_COLS + RAW_TIME_COLS + [TIMESTAMP_COL, LABEL_COL]

# Seeds for the multi-seed experiments ([C2] baseline seeds, [D3] domain clf).
BASELINE_SEEDS = [42, 7, 123]
DETECTOR_SEEDS = [42, 7, 123]

# Decision-threshold tuning grid shared by phase C and phase F.
THRESHOLD_LO, THRESHOLD_HI, THRESHOLD_STEP = 0.01, 0.99, 0.01


# -----------------------------------------------------------------------
# Label encoder
# -----------------------------------------------------------------------

class SimpleLabelEncoder:
    """
    Minimal label encoder, no sklearn/scipy needed.

    NOTE: keep the class NAME identical to the one used when the baseline
    .pkl files were created (pickle stores the class path). transform() is
    vectorized via pandas map (much faster than a Python loop over rows) but
    returns exactly the same values as the original loop version.
    """

    def __init__(self):
        self.classes_ = None
        self.mapping = {}

    def fit(self, values):
        self.classes_ = sorted(set(values))
        self.mapping = {v: i for i, v in enumerate(self.classes_)}
        return self

    def transform(self, values):
        s = pd.Series(values)
        unseen_fill = self.mapping.get("__UNSEEN__", -1)
        return s.map(self.mapping).fillna(unseen_fill).astype(int).values

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    def add_unseen_bucket(self):
        """Reserve one extra id for categories that appear after training."""
        if "__UNSEEN__" not in self.mapping:
            new_id = len(self.mapping)
            self.mapping["__UNSEEN__"] = new_id
            self.classes_.append("__UNSEEN__")


# -----------------------------------------------------------------------
# Data loading / preprocessing / splitting
# -----------------------------------------------------------------------

def fit_imputer(df, feature_cols=None):
    """
    Calculate imputation values strictly on the training era (Chunk 1).
    Avoids future-to-past data leakage across temporal chunks.
    - Numerics: median
    - Categoricals / strings: mode
    """
    stats = {}
    cols = feature_cols if feature_cols is not None else df.columns
    for c in cols:
        if c in DROP_COLS or c == LABEL_COL:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            med = df[c].dropna().median()
            stats[c] = float(med) if pd.notna(med) else 0.0
        else:
            mode_vals = df[c].dropna().mode()
            stats[c] = str(mode_vals.iloc[0]) if not mode_vals.empty else "unknown"
    return stats


def apply_imputer(df, imputer_stats):
    """Fill missing values using precomputed training-era statistics."""
    df_out = df.copy()
    for c, fill_val in imputer_stats.items():
        if c in df_out.columns:
            df_out[c] = df_out[c].fillna(fill_val)
    return df_out


def load_chunk(path, encoders=None, fit_encoders=False):
    """
    Load a chunk CSV, drop leakage/ID/raw-time columns, integer-encode the
    categorical features.

    fit_encoders=True  -> fit new encoders on THIS chunk (phase C, chunk 1).
    fit_encoders=False -> reuse the provided encoders; unseen categories map
                          to the reserved __UNSEEN__ bucket id.
    Returns (X, y, encoders).
    """
    df = pd.read_csv(path)
    y = df[LABEL_COL].astype(int)

    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    if encoders is None:
        encoders = {}

    for col in CATEGORICAL_COLS:
        if col not in X.columns:
            continue
        if fit_encoders:
            le = SimpleLabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str).tolist())
            le.add_unseen_bucket()
            encoders[col] = le
        else:
            le = encoders[col]
            X[col] = le.transform(X[col].astype(str).tolist())
            # any -1 (truly unmapped value) -> map to the unseen bucket id
            unseen_id = le.mapping["__UNSEEN__"]
            X[col] = np.where(X[col] == -1, unseen_id, X[col])

    return X, y, encoders


def simple_train_test_split(X, y, test_size=0.2, seed=42):
    """
    Stratified-ish split without sklearn: shuffles indices per class, then
    splits. Fully deterministic for a given seed.

    MUST stay byte-identical in behaviour across phases: phase D/E/F rely on
    reproducing the exact held-out slice phase C validated on (same seed =>
    same split). `y` may be a pandas Series or 1-D array.
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(np.asarray(y))
    rng = np.random.default_rng(seed)
    idx_pos = np.where(y.values == 1)[0]
    idx_neg = np.where(y.values == 0)[0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)

    n_pos_test = max(1, int(len(idx_pos) * test_size))
    n_neg_test = max(1, int(len(idx_neg) * test_size))
    test_idx = np.concatenate([idx_pos[:n_pos_test], idx_neg[:n_neg_test]])
    train_idx = np.concatenate([idx_pos[n_pos_test:], idx_neg[n_neg_test:]])
    rng.shuffle(test_idx)
    rng.shuffle(train_idx)

    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def load_baseline_bundle(model_path):
    """
    Load the phase C artifact (model + encoders + tuned threshold).
    Tolerates the older format where the bare model was pickled directly.
    """
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    if isinstance(bundle, dict) and "model" in bundle:
        return bundle
    return {"model": bundle, "encoders": None, "threshold": 0.5}


# -----------------------------------------------------------------------
# Metrics (powered by scikit-learn)
# -----------------------------------------------------------------------

def precision_recall_f1(y_true, y_pred):
    """Returns (precision, recall, f1) using sklearn."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if np.sum(y_pred == 1) == 0 and np.sum(y_true == 1) == 0:
        return 0.0, 0.0, 0.0
    p = float(precision_score(y_true, y_pred, zero_division=0))
    r = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(sklearn_f1_score(y_true, y_pred, zero_division=0))
    return p, r, f1


def f1_score_manual(y_true, y_pred):
    """F1 score via sklearn (kept name for backward compatibility)."""
    _, _, f1 = precision_recall_f1(y_true, y_pred)
    return f1


def roc_auc_manual(y_true, y_score):
    """ROC-AUC via sklearn.metrics.roc_auc_score."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def average_precision_manual(y_true, y_score):
    """Average Precision via sklearn.metrics.average_precision_score."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = np.sum(y_true == 1)
    if n_pos == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def evaluate(y_true, y_pred, y_proba):
    """Standard evaluation metrics dict."""
    precision, recall, f1 = precision_recall_f1(y_true, y_pred)
    ap = average_precision_manual(y_true, y_proba)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc_manual(y_true, y_proba),
        "pr_auc": ap,
        "average_precision": ap,
        "n_positives": int(np.asarray(y_true).sum()),
        "n_rows": len(y_true),
    }


# -----------------------------------------------------------------------
# Model helpers
# -----------------------------------------------------------------------

def tune_threshold(y_val, proba, lo=THRESHOLD_LO, hi=THRESHOLD_HI,
                   step=THRESHOLD_STEP):
    """
    Find the F1-maximizing decision threshold on already-computed scores
    (default 0.5 is a bad choice under extreme imbalance).

    Takes the scores rather than the model so callers that already predicted
    do not pay for a second predict_proba pass.
    """
    y_true = np.asarray(y_val)
    proba = np.asarray(proba, dtype=float)
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(lo, hi, step):
        f1 = f1_score_manual(y_true, (proba >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


# -----------------------------------------------------------------------
# Drift statistics
# -----------------------------------------------------------------------

def calculate_psi(reference, current, bins=10):
    """Population Stability Index over quantile bins of the reference."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if len(reference) == 0 or len(current) == 0:
        return float("nan")

    quantiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(reference, quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    bin_edges = np.unique(bin_edges)

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    ref_pct = ref_counts / max(len(reference), 1)
    cur_pct = cur_counts / max(len(current), 1)

    eps = 1e-6
    ref_pct = np.where(ref_pct == 0, eps, ref_pct)
    cur_pct = np.where(cur_pct == 0, eps, cur_pct)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def calculate_categorical_psi(reference, current):
    """
    PSI adapted for categorical features: each category is a 'bin'; compares
    the proportion of each category between reference and current samples.
    """
    ref_series = pd.Series(reference).astype(str)
    cur_series = pd.Series(current).astype(str)

    categories = sorted(set(ref_series.unique()) | set(cur_series.unique()))

    ref_pct = ref_series.value_counts(normalize=True).reindex(categories, fill_value=0).values
    cur_pct = cur_series.value_counts(normalize=True).reindex(categories, fill_value=0).values

    eps = 1e-6
    ref_pct = np.where(ref_pct == 0, eps, ref_pct)
    cur_pct = np.where(cur_pct == 0, eps, cur_pct)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def ks_statistic(reference, current):
    """
    Two-sample Kolmogorov-Smirnov statistic via scipy.stats.ks_2samp.
    Returns the KS statistic (maximum absolute CDF gap).
    Used by the output-distribution detector ([D2]).
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    stat, _pvalue = ks_2samp(ref, cur)
    return float(stat)


# -----------------------------------------------------------------------
# Timestamp parsing + derived (D1) PSI monitoring features
# -----------------------------------------------------------------------

def parse_chunk_timestamp(df):
    """
    Parse the TRUE transaction timestamp from a chunk's raw columns.

    FIX (chronology bug): the original phaseB built __timestamp__ as
    pd.to_datetime(Time + " " + Date) — Time FIRST — which pandas mangled
    into values that do not follow real chronology (e.g. Date=2022-11-01,
    Time=01:00:23 -> 2022-01-01 11:00:23), so the "temporal" chunks had
    overlapping date ranges. We now combine Date FIRST with an explicit
    format, falling back to a generic parse only if the strict parse fails
    on >20% of rows.
    """
    if {"Date", "Time"} <= set(df.columns):
        s = (df["Date"].astype(str).str.strip() + " "
             + df["Time"].astype(str).str.strip())
        ts = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S", errors="coerce")
        if ts.isna().mean() > 0.2:  # unexpected format -> generic fallback
            ts = pd.to_datetime(s, errors="coerce")
        return ts
    if TIMESTAMP_COL in df.columns:
        return pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")


# Numeric features monitored by the PSI detector ([D1]: not just "Amount").
# All are computable from a single chunk in isolation -> no cross-chunk leak.
PSI_NUMERIC_FEATURES = [
    "Amount",
    "Amount_log",
    "hour_of_day",
    "day_of_week",
    "tx_per_sender",
    "amount_per_sender_mean",
]


def derive_psi_features(df):
    """
    Build the numeric monitoring features used by the PSI detector ([D1]).

    Rationale: monitoring only raw "Amount" made the 'PSI saw nothing' claim
    arguable. We add derived, leakage-free behaviour features:
      - Amount        raw amount
      - Amount_log    log1p(amount) — tames the heavy tail so quantile bins
                      resolve the low-amount bulk of transactions
      - hour_of_day   time-of-day pattern from the raw Date/Time strings
      - day_of_week   weekly pattern from the raw Date string
      - tx_per_sender / amount_per_sender_mean: per-sender behaviour,
        computed WITHIN the chunk (relative quantities, never account IDs)

    Returns a float DataFrame indexed like df with PSI_NUMERIC_FEATURES.
    """
    out = pd.DataFrame(index=df.index)

    if "Amount" in df.columns:
        amount = pd.to_numeric(df["Amount"], errors="coerce")
    else:
        amount = pd.Series(np.nan, index=df.index)
    out["Amount"] = amount
    out["Amount_log"] = np.log1p(amount)

    ts = parse_chunk_timestamp(df)
    if ts.notna().any():
        out["hour_of_day"] = ts.dt.hour.fillna(-1).astype(float)
        out["day_of_week"] = ts.dt.dayofweek.fillna(-1).astype(float)
    else:
        out["hour_of_day"] = -1.0
        out["day_of_week"] = -1.0

    if "Sender_account" in df.columns:
        sender = df["Sender_account"]
        out["tx_per_sender"] = sender.groupby(sender).transform("count").astype(float)
        out["amount_per_sender_mean"] = amount.groupby(sender).transform("mean")
    else:
        out["tx_per_sender"] = 1.0
        out["amount_per_sender_mean"] = amount

    return out.fillna(0.0).astype(float)


# -----------------------------------------------------------------------
# Small reporting helpers ([X2] / [E2])
# -----------------------------------------------------------------------

_CHUNK_FILE_RE = re.compile(r"^chunk_\d+\.csv$")


def list_chunk_files(chunks_dir):
    """
    List only the data chunk files (chunk_01.csv, chunk_02.csv, ...) in a
    directory — excludes helper artifacts like chunk_summary.csv that also
    start with 'chunk_'. Returns [(chunk_number, filename), ...] sorted.
    """
    files = [f for f in os.listdir(chunks_dir) if _CHUNK_FILE_RE.match(f)]
    return sorted((int(f.split("_")[1].split(".")[0]), f) for f in files)


def add_seed_manifest(df, seeds):
    """[X2] Append a 'seeds' column noting which seeds produced this CSV."""
    out = df.copy()
    out["seeds"] = ";".join(str(s) for s in seeds)
    return out


def ci95(std, n):
    """Half-width of a 95% confidence interval: 1.96 * std / sqrt(n)."""
    if n is None or n <= 0 or std is None:
        return float("nan")
    if isinstance(std, float) and np.isnan(std):
        return float("nan")
    return 1.96 * float(std) / np.sqrt(n)