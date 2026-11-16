"""Independent check of the extended lab's results, run outside the notebook.

1. Loads the saved serving model (models/feature_selected_hgb.pkl) in a fresh process.
2. Rebuilds the same locked test split and recomputes the test metrics.
3. Compares them with reports/final_result.md and the feature list in reports/selected_features.txt.
4. Reproduces the leakage demo (selecting on all rows inflates AUC on pure noise).

Usage: python src/verify.py        (exit code 0 = all checks passed)
"""
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))  # so pickle can import fs_transformers.ColumnSubset
failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(name)


# --- 1-3: saved model reproduces the reported test numbers -------------------------------
df = pd.read_csv(ROOT / "data/spambase.csv")
X, y = df.drop(columns="is_spam"), df["is_spam"]
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

with open(ROOT / "models/feature_selected_hgb.pkl", "rb") as f:
    model = pickle.load(f)
proba = model.predict_proba(X_test)[:, 1]
auc, f1, acc = roc_auc_score(y_test, proba), f1_score(y_test, proba >= 0.5), accuracy_score(y_test, proba >= 0.5)
print(f"saved model on locked test set: AUC {auc:.4f}  F1 {f1:.4f}  accuracy {acc:.4f}  ({len(model[0].columns)} features)")

report = (ROOT / "reports/final_result.md").read_text()
winner = re.search(r"\*\*(.+?)\*\* \(", report).group(1)
row = next(l for l in report.splitlines() if l.startswith(f"| {winner} |"))
rep_auc, rep_f1, rep_acc = map(float, row.split("|")[3:6])
check("test AUC matches report", abs(auc - rep_auc) < 1e-4, f"({auc:.4f} vs {rep_auc:.4f})")
check("test F1 matches report", abs(f1 - rep_f1) < 1e-4, f"({f1:.4f} vs {rep_f1:.4f})")
check("test accuracy matches report", abs(acc - rep_acc) < 1e-4, f"({acc:.4f} vs {rep_acc:.4f})")

listed = (ROOT / "reports/selected_features.txt").read_text().split()
check("model uses exactly the listed features", list(model[0].columns) == listed, f"({len(listed)} features)")
check("model works without unused columns", np.allclose(model.predict_proba(X_test[listed])[:, 1], proba))

# --- 4: leakage demo -------------------------------------------------------------------------
rng = np.random.default_rng(42)
Xn, yn = rng.normal(size=(200, 2000)), rng.integers(0, 2, size=200)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
clf = LogisticRegression(max_iter=2000)
leaky = cross_val_score(clf, SelectKBest(f_classif, k=20).fit_transform(Xn, yn), yn, cv=cv, scoring="roc_auc").mean()
clean = cross_val_score(make_pipeline(SelectKBest(f_classif, k=20), clf), Xn, yn, cv=cv, scoring="roc_auc").mean()
print(f"noise data: leaky AUC {leaky:.3f}, pipeline AUC {clean:.3f}")
check("leaky selection inflates AUC on pure noise", leaky > 0.8 and clean < 0.7)

print("\nALL CHECKS PASSED" if not failures else f"\n{len(failures)} CHECK(S) FAILED: {failures}")
sys.exit(1 if failures else 0)
