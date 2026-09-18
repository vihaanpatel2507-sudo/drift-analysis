import pandas as pd

df = pd.read_csv("chunks/chunk_01.csv")

print("Columns:", df.columns.tolist())

print("Dtypes:")
print(df.dtypes)

print("Shape:", df.shape)

print("First 3 rows:")
print(df.head(3))

for col in df.columns:
    if df[col].dtype == object or df[col].nunique() < 15:
        print()
        print(f"--- {col} unique values ({df[col].nunique()}) ---")
        print(df[col].value_counts().head(10))
