"""Download the UCI Spambase dataset and write data/spambase.csv with named columns.

Usage: python src/get_data.py
"""
import io
import re
import ssl
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/94/spambase.zip"
OUT = Path(__file__).resolve().parent.parent / "data" / "spambase.csv"


def main() -> None:
    try:  # python.org macOS builds ship without a CA bundle; certifi fixes that
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    with urllib.request.urlopen(URL, timeout=60, context=ctx) as resp:
        archive = zipfile.ZipFile(io.BytesIO(resp.read()))

    names_txt = archive.read("spambase.names").decode()
    # Attribute lines look like "word_freq_make:         continuous."
    cols = re.findall(r"^([a-z_]+_[^:\s]+):\s+continuous\.", names_txt, flags=re.M)
    assert len(cols) == 57, f"expected 57 feature names, got {len(cols)}"

    df = pd.read_csv(io.BytesIO(archive.read("spambase.data")), header=None, names=cols + ["is_spam"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} -> {df.shape[0]} rows, {df.shape[1] - 1} features, spam rate {df.is_spam.mean():.2%}")


if __name__ == "__main__":
    main()
