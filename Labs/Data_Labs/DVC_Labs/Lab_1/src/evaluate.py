"""Stage 3 - evaluate on the held-out split, write metrics and plot data."""
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)

params = yaml.safe_load(Path("params.yaml").read_text())
threshold = params["evaluate"]["decision_threshold"]

bundle = joblib.load("models/model.joblib")
model, features = bundle["model"], bundle["features"]
test = pd.read_csv("data/processed/test.csv")
X, y = test[features], test["label"]

proba = model.predict_proba(X)[:, 1]
pred = (proba >= threshold).astype(int)

scores = {
    "model": bundle["algo"],
    "accuracy": accuracy_score(y, pred),
    "precision": precision_score(y, pred, zero_division=0),
    "recall": recall_score(y, pred),
    "f1": f1_score(y, pred),
    "roc_auc": roc_auc_score(y, proba),
    "avg_precision": average_precision_score(y, proba),
}
scores = {k: round(v, 4) if isinstance(v, float) else v for k, v in scores.items()}

Path("metrics").mkdir(exist_ok=True)
plots = Path("plots")
plots.mkdir(exist_ok=True)
Path("metrics/scores.json").write_text(json.dumps(scores, indent=2))

# Plot data consumed by `dvc plots show`
pd.DataFrame({"actual": y, "predicted": pred}).to_csv(
    plots / "confusion.csv", index=False)
fpr, tpr, _ = roc_curve(y, proba)
pd.DataFrame({"fpr": fpr, "tpr": tpr}).round(4).to_csv(plots / "roc.csv", index=False)
prec, rec, _ = precision_recall_curve(y, proba)
pd.DataFrame({"recall": rec, "precision": prec}).round(4).to_csv(
    plots / "prc.csv", index=False)

imp = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=5,
                             random_state=params["train"]["seed"], n_jobs=-1)
(pd.DataFrame({"feature": features, "importance": imp.importances_mean})
   .sort_values("importance", ascending=False).round(4)
   .to_csv(plots / "feature_importance.csv", index=False))

print(json.dumps(scores, indent=2))
