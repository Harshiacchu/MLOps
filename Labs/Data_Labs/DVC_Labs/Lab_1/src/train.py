"""Stage 2 - train the classifier selected in params.yaml."""
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

params = yaml.safe_load(Path("params.yaml").read_text())["train"]
seed = params["seed"]
algo = params["model"]

train = pd.read_csv("data/processed/train.csv")
X, y = train.drop(columns="label"), train["label"]

if algo == "random_forest":
    p = params["random_forest"]
    model = RandomForestClassifier(
        n_estimators=p["n_estimators"], max_depth=p["max_depth"],
        min_samples_leaf=p["min_samples_leaf"], class_weight=p["class_weight"],
        n_jobs=-1, random_state=seed)
elif algo == "gradient_boosting":
    p = params["gradient_boosting"]
    model = GradientBoostingClassifier(
        n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
        max_depth=p["max_depth"], subsample=p["subsample"], random_state=seed)
elif algo == "logistic_regression":
    p = params["logistic_regression"]
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=p["C"], class_weight=p["class_weight"],
                           max_iter=2000, random_state=seed))
else:
    raise ValueError(f"Unknown model '{algo}'")

model.fit(X, y)
Path("models").mkdir(exist_ok=True)
joblib.dump({"model": model, "features": list(X.columns), "algo": algo},
            "models/model.joblib")
print(f"Trained {algo} on {len(X)} rows, {X.shape[1]} features")
