import pandas as pd
import glob

chunk_files = sorted(glob.glob("chunks/chunk_*.csv"))

print(f"Found {len(chunk_files)} chunk files")
print()
print(f"{"Chunk":<15}{"Rows":<10}{"Positives":<12}{"Positive %":<12}{"Date range"}")
print("-" * 80)

for f in chunk_files:
    df = pd.read_csv(f, usecols=["Is_laundering", "__timestamp__"])
    n = len(df)
    pos = df["Is_laundering"].sum()
    pct = 100 * pos / n if n > 0 else 0
    date_min = df["__timestamp__"].min()
    date_max = df["__timestamp__"].max()
    print(f"{f:<15}{n:<10}{pos:<12}{pct:<12.4f}{date_min} to {date_max}")
