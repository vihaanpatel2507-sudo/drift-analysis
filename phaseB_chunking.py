"""
Phase B: Data Understanding, Preprocessing, and Temporal Chunking
for AML Concept Drift Research (SAML-D dataset)

Splits time-ordered transaction data into sequential chunks without
look-ahead data leakage. Missing value imputation statistics are fit
strictly on Chunk 1 (training era) and applied forward.

Usage:
    python phaseB_chunking.py --data_path path/to/saml_d.csv
"""

import argparse
import os
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

DEFAULT_CHUNKS_DIR = "chunks"
DEFAULT_N_CHUNKS = 10


def load_and_inspect(data_path):
    print("=" * 70)
    print("STEP 1: DATA LOADING & INSPECTION")
    print("=" * 70)

    print(f"\nReading sample (first 5,000 rows) from: {data_path}")
    sample = pd.read_csv(data_path, nrows=5000)

    print("\n--- Columns ---")
    print(sample.columns.tolist())

    print("\n--- First 3 rows ---")
    print(sample.head(3))

    label_candidates = [
        c for c in sample.columns
        if any(k in c.lower() for k in ["is_laundering", "laundering", "fraud", "label", "target"])
    ]
    time_candidates = [
        c for c in sample.columns
        if any(k in c.lower() for k in ["date", "time", "timestamp"])
    ]

    print(f"\nDetected label candidate(s): {label_candidates}")
    print(f"Detected timestamp candidate(s): {time_candidates}")

    # Count total rows safely
    total_rows = sum(1 for _ in open(data_path)) - 1
    print(f"Total rows in dataset: {total_rows:,}")

    return sample, label_candidates, time_candidates, total_rows


def parse_timestamps(df):
    """
    Parse true datetime with explicit format to avoid ambiguous day/month
    inversions during temporal sorting.
    """
    if "Date" in df.columns and "Time" in df.columns:
        ts_str = df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip()
        ts = pd.to_datetime(ts_str, format="%Y-%m-%d %H:%M:%S", errors="coerce")
        if ts.isna().mean() > 0.2:
            ts = pd.to_datetime(ts_str, errors="coerce")
        return ts

    for col in ["__timestamp__", "Timestamp", "timestamp", "Date", "date", "Time", "time"]:
        if col in df.columns:
            return pd.to_datetime(df[col], errors="coerce")

    return None


def preprocess_and_chunk(data_path, label_col, n_chunks=10, out_dir=DEFAULT_CHUNKS_DIR):
    print("\n" + "=" * 70)
    print("STEP 2: PREPROCESSING & TEMPORAL CHUNKING")
    print("=" * 70)

    print(f"\nLoading full dataset from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} rows with {df.shape[1]} columns.")

    # 1. Parse timestamps and sort chronologically
    ts = parse_timestamps(df)
    if ts is not None and ts.notna().any():
        df["__timestamp__"] = ts
        n_unparseable = int(df["__timestamp__"].isna().sum())
        if n_unparseable > 0:
            print(f"Dropping {n_unparseable:,} rows with unparseable timestamps.")
            df = df.dropna(subset=["__timestamp__"])
        df = df.sort_values("__timestamp__").reset_index(drop=True)
        print("Dataset sorted chronologically by __timestamp__.")
    else:
        print("No parseable datetime found; preserving existing file row order.")

    # 2. Clean label column if present
    if label_col and label_col in df.columns:
        n_missing_labels = int(df[label_col].isna().sum())
        if n_missing_labels > 0:
            df = df.dropna(subset=[label_col])
            print(f"Dropped {n_missing_labels:,} rows missing label '{label_col}'.")
        df[label_col] = df[label_col].astype(int)
        pos_rate = df[label_col].mean()
        print(f"Overall positive rate: {pos_rate:.5f} ({df[label_col].sum():,} positive rows)")

    # 3. Methodological improvement: fit imputation parameters STRICTLY on Chunk 1
    # This completely eliminates look-ahead leakage across temporal chunks.
    chunk_size = len(df) // n_chunks
    chunk1_ref = df.iloc[:chunk_size]

    imputer_stats = {}
    for col in df.columns:
        if col in ["__timestamp__", label_col, "Laundering_type"]:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            med = chunk1_ref[col].dropna().median()
            imputer_stats[col] = float(med) if pd.notna(med) else 0.0
        elif df[col].dtype == object:
            mode_vals = chunk1_ref[col].dropna().mode()
            imputer_stats[col] = mode_vals.iloc[0] if not mode_vals.empty else "unknown"

    print("\nFitting imputation statistics on Chunk 1 (training era) to prevent future-to-past leakage:")
    for col, fill_val in imputer_stats.items():
        n_na = int(df[col].isna().sum())
        if n_na > 0:
            print(f"  {col}: {n_na:,} missing rows filled with '{fill_val}'")
            df[col] = df[col].fillna(fill_val)

    # 4. Save sequential temporal chunks
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nWriting {n_chunks} sequential chunks to ./{out_dir}/...")

    for i in range(n_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < n_chunks - 1 else len(df)
        chunk = df.iloc[start:end].copy()

        chunk_filename = f"chunk_{i+1:02d}.csv"
        chunk_path = os.path.join(out_dir, chunk_filename)
        chunk.to_csv(chunk_path, index=False)

        info = f"rows {start:>8:,}:{end:>8:,} ({len(chunk):>7:,} rows)"
        if label_col and label_col in chunk.columns:
            pos_count = int(chunk[label_col].sum())
            pos_pct = (pos_count / len(chunk)) * 100
            info += f" | {pos_count:>4} positives ({pos_pct:.3f}%)"
        if "__timestamp__" in chunk.columns:
            t_min = str(chunk["__timestamp__"].min())[:10]
            t_max = str(chunk["__timestamp__"].max())[:10]
            info += f" | {t_min} to {t_max}"

        print(f"  {chunk_filename} -> {info}")

    print(f"\nDone. Chunk 1 represents the baseline training era; Chunks 2..{n_chunks} represent the streaming future.")
    return df


def main():
    parser = argparse.ArgumentParser(description="Temporal chunking for SAML-D AML drift experiment.")
    parser.add_argument("--data_path", type=str, default="SAML-D.csv", help="Path to raw transaction CSV")
    parser.add_argument("--n_chunks", type=int, default=DEFAULT_N_CHUNKS, help="Number of sequential chunks to create")
    parser.add_argument("--label_col", type=str, default="Is_laundering", help="Target label column name")
    parser.add_argument("--out_dir", type=str, default=DEFAULT_CHUNKS_DIR, help="Directory to save chunks")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Warning: Data file '{args.data_path}' was not found in current directory.")
        print("Ensure the dataset is downloaded or specify the correct path with --data_path.")
        return

    sample, label_candidates, time_candidates, total_rows = load_and_inspect(args.data_path)
    label_col = args.label_col or (label_candidates[0] if label_candidates else None)
    preprocess_and_chunk(args.data_path, label_col=label_col, n_chunks=args.n_chunks, out_dir=args.out_dir)


if __name__ == "__main__":
    main()