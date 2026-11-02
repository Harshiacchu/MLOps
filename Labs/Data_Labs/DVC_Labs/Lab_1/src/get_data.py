"""Download the UCI Wine Quality data and build data/raw/wine.csv.

Two dataset versions are supported so DVC data versioning can be exercised
on a real change:
  v1 -> red wines only           (1,599 rows)
  v2 -> red + white wines        (6,497 rows)
"""
import argparse
import io
import ssl
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality"
SOURCES = {"red": f"{BASE_URL}/winequality-red.csv",
           "white": f"{BASE_URL}/winequality-white.csv"}
VERSIONS = {"v1": ["red"], "v2": ["red", "white"]}


def _ssl_context() -> ssl.SSLContext:
    # python.org macOS builds ship without system CA certs; prefer certifi.
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def load(color: str) -> pd.DataFrame:
    with urlopen(SOURCES[color], timeout=60, context=_ssl_context()) as resp:
        df = pd.read_csv(io.BytesIO(resp.read()), sep=";")
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    df.insert(0, "wine_type", color)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=VERSIONS, default="v2")
    parser.add_argument("--out", default="data/raw/wine.csv")
    args = parser.parse_args()

    df = pd.concat([load(c) for c in VERSIONS[args.version]], ignore_index=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[{args.version}] wrote {len(df)} rows -> {out}")
    print(df["wine_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
