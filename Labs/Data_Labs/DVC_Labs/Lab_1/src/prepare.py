"""Stage 1 - clean the raw data, engineer features and split train/test."""
import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

params = yaml.safe_load(Path("params.yaml").read_text())["prepare"]

RAW = Path("data/raw/wine.csv")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)
Path("metrics").mkdir(exist_ok=True)

df = pd.read_csv(RAW)
rows_raw = len(df)

# --- cleaning -------------------------------------------------------------
df = df.dropna()
rows_after_na = len(df)
df = df.drop_duplicates()
rows_after_dedup = len(df)

# --- feature engineering --------------------------------------------------
df["is_red"] = (df["wine_type"] == "red").astype(int)
df["free_so2_ratio"] = df["free_sulfur_dioxide"] / df["total_sulfur_dioxide"]
df["acidity_total"] = df["fixed_acidity"] + df["volatile_acidity"] + df["citric_acid"]
df["alcohol_x_sulphates"] = df["alcohol"] * df["sulphates"]

# Binary target: "good" wine if quality >= threshold
df["label"] = (df["quality"] >= params["quality_threshold"]).astype(int)
df = df.drop(columns=["wine_type", "quality"])

train, test = train_test_split(
    df, test_size=params["test_size"], random_state=params["seed"],
    stratify=df["label"])
# Clip extreme outliers (winsorize) using quantiles from the TRAIN split only,
# so no information from the test set leaks into preprocessing.
num_cols = [c for c in train.columns if c not in ("label", "is_red")]
lo, hi = params["clip_quantiles"]
bounds = train[num_cols].quantile([lo, hi])
for c in num_cols:
    train[c] = train[c].clip(bounds.loc[lo, c], bounds.loc[hi, c])
    test[c] = test[c].clip(bounds.loc[lo, c], bounds.loc[hi, c])

train.to_csv(OUT / "train.csv", index=False)
test.to_csv(OUT / "test.csv", index=False)

report = {
    "rows_raw": rows_raw,
    "rows_dropped_na": rows_raw - rows_after_na,
    "rows_dropped_duplicates": rows_after_na - rows_after_dedup,
    "rows_clean": rows_after_dedup,
    "n_features": df.shape[1] - 1,
    "positive_rate": round(float(df["label"].mean()), 4),
    "train_rows": len(train),
    "test_rows": len(test),
}
Path("metrics/data_report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
