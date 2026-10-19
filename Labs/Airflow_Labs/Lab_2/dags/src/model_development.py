import os
import pickle
import json
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


WORKING_DIR = "/opt/airflow/working_data"
MODEL_DIR = "/opt/airflow/model"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data() -> str:
    """
    Load CSV and persist raw dataframe to a pickle file.
    Returns path to saved file.
    """
    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "advertising.csv",
    )
    df = pd.read_csv(csv_path)

    out_path = os.path.join(WORKING_DIR, "raw.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(df, f)
    return out_path

def data_preprocessing(file_path: str) -> str:
    """
    Load dataframe, split, scale, and save (X_train, X_test, y_train, y_test) to pickle.
    Returns path to saved file.
    """
    with open(file_path, "rb") as f:
        df = pickle.load(f)

    X = df.drop(
        ["Timestamp", "Clicked on Ad", "Ad Topic Line", "Country", "City"],
        axis=1,
    )
    y = df["Clicked on Ad"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    num_columns = [
        "Daily Time Spent on Site",
        "Age",
        "Area Income",
        "Daily Internet Usage",
        "Male",
    ]

    ct = make_column_transformer(
        (StandardScaler(), num_columns),
        remainder="passthrough",
    )

    X_train_tr = ct.fit_transform(X_train)
    X_test_tr = ct.transform(X_test)

    out_path = os.path.join(WORKING_DIR, "preprocessed.pkl")
    with open(out_path, "wb") as f:
        pickle.dump((X_train_tr, X_test_tr, y_train.values, y_test.values), f)
    return out_path

def separate_data_outputs(file_path: str) -> str:
    """
    Passthrough; kept so the DAG composes cleanly.
    """
    return file_path

def build_model(file_path: str, filename: str) -> str:
    """Train three models, select a champion using validation F1, and save results."""
    with open(file_path, "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train,
        test_size=0.25,
        random_state=42,
        stratify=y_train,
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        ),
    }

    leaderboard = []
    best_name = None
    best_f1 = -1

    for name, model in candidates.items():
        model.fit(X_fit, y_fit)
        predictions = model.predict(X_val)

        metrics = {
            "model": name,
            "accuracy": round(float(accuracy_score(y_val, predictions)), 4),
            "precision": round(float(precision_score(
                y_val, predictions, zero_division=0
            )), 4),
            "recall": round(float(recall_score(
                y_val, predictions, zero_division=0
            )), 4),
            "f1_score": round(float(f1_score(
                y_val, predictions, zero_division=0
            )), 4),
        }

        leaderboard.append(metrics)

        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_name = name

    leaderboard.sort(key=lambda item: item["f1_score"], reverse=True)

    champion = candidates[best_name]
    champion.fit(X_train, y_train)

    model_path = os.path.join(MODEL_DIR, filename)

    with open(model_path, "wb") as f:
        pickle.dump(champion, f)

    test_predictions = champion.predict(X_test)

    results = {
        "champion": best_name,
        "selection_metric": "validation F1-score",
        "leaderboard": leaderboard,
        "test_accuracy": round(float(
            accuracy_score(y_test, test_predictions)
        ), 4),
        "test_f1_score": round(float(
            f1_score(y_test, test_predictions, zero_division=0)
        ), 4),
    }

    with open(os.path.join(MODEL_DIR, "leaderboard.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    return model_path

def load_model(file_path: str, filename: str) -> int:
    """
    Load saved model and test set, print score, and return first prediction as int.
    """
    with open(file_path, "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)

    model_path = os.path.join(MODEL_DIR, filename)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1_score": f1_score(y_test, pred, zero_division=0),
    }

    print("Model Evaluation Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    return int(pred[0])
