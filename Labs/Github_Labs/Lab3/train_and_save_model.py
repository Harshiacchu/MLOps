"""
Machine Learning Training and Artifact Export Pipeline.

Automated pipeline for model training, evaluation, and artifact storage to GCS.
"""

import argparse
import io
import json
import os
from datetime import datetime
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
from google.cloud import storage
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data(
    test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
    """
    Load Breast Cancer Wisconsin dataset and split into train and test sets.
    """
    cancer = load_breast_cancer()
    X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y = pd.Series(cancer.target, name="target")

    dataset_info = {
        "dataset_name": "Breast Cancer Wisconsin (Diagnostic)",
        "num_samples": int(X.shape[0]),
        "num_features": int(X.shape[1]),
        "feature_names": list(cancer.feature_names),
        "target_names": list(cancer.target_names),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, dataset_info


def build_and_train_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    random_state: int = 42,
) -> Pipeline:
    """
    Construct a pipeline with feature scaling and GradientBoostingClassifier.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    random_state=random_state,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, Any]:
    """
    Evaluate trained pipeline on test set and return comprehensive metrics.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = (
        pipeline.predict_proba(X_test)[:, 1]
        if hasattr(pipeline, "predict_proba")
        else None
    )

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        ),
    }

    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))

    return metrics


def save_artifacts_to_gcs(
    pipeline: Pipeline,
    metadata: Dict[str, Any],
    bucket_name: str,
    model_blob_name: str,
    metadata_blob_name: str,
) -> bool:
    """
    Saves model binary and metadata JSON to Google Cloud Storage.
    """
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        # Upload model pipeline
        model_blob = bucket.blob(model_blob_name)
        model_buffer = io.BytesIO()
        joblib.dump(pipeline, model_buffer)
        model_buffer.seek(0)
        model_blob.upload_from_file(
            model_buffer, content_type="application/octet-stream"
        )
        print(f"✅ Model uploaded to gs://{bucket_name}/{model_blob_name}")

        # Upload metadata JSON
        meta_blob = bucket.blob(metadata_blob_name)
        meta_blob.upload_from_string(
            json.dumps(metadata, indent=2), content_type="application/json"
        )
        print(
            f"✅ Metadata uploaded to gs://{bucket_name}/{metadata_blob_name}"
        )
        return True
    except Exception as e:
        print(f"⚠️ Failed to upload artifacts to GCS: {e}")
        return False


def save_artifacts_locally(
    pipeline: Pipeline,
    metadata: Dict[str, Any],
    output_dir: str,
    timestamp: str,
) -> Tuple[str, str]:
    """
    Saves model and metadata locally as fallback or local artifact storage.
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(
        output_dir, f"breast_cancer_gb_pipeline_{timestamp}.joblib"
    )
    meta_path = os.path.join(output_dir, f"metadata_{timestamp}.json")

    joblib.dump(pipeline, model_path)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"💾 Local model artifact saved: {model_path}")
    print(f"💾 Local metadata artifact saved: {meta_path}")
    return model_path, meta_path


def main():
    parser = argparse.ArgumentParser(
        description="Train Gradient Boosting model and save artifacts."
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=os.getenv("GCS_BUCKET_NAME", "mlops-github-actions-harshitha"),
        help="Google Cloud Storage bucket name",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default="artifacts",
        help="Local directory to store fallback artifacts",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of boosting stages to perform",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate shrinks the contribution of each tree",
    )
    args = parser.parse_args()

    print("🚀 Starting ML Pipeline Execution...")
    X_train, X_test, y_train, y_test, dataset_info = load_and_prepare_data()

    print(f"📊 Dataset: {dataset_info['dataset_name']}")
    print(f"🔹 Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"🔹 Features count: {dataset_info['num_features']}")

    # Train model
    print("⚙️ Training GradientBoostingClassifier pipeline...")
    pipeline = build_and_train_pipeline(
        X_train,
        y_train,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
    )

    # Evaluate model
    metrics = evaluate_model(pipeline, X_test, y_test)
    print("\n📈 Model Evaluation Results:")
    print(f"  • Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  • Precision: {metrics['precision']:.4f}")
    print(f"  • Recall:    {metrics['recall']:.4f}")
    print(f"  • F1-Score:  {metrics['f1_score']:.4f}")
    if "roc_auc" in metrics:
        print(f"  • ROC-AUC:   {metrics['roc_auc']:.4f}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    metadata = {
        "timestamp": timestamp,
        "model_type": "Pipeline(StandardScaler + GradientBoostingClassifier)",
        "hyperparameters": {
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": 3,
            "random_state": 42,
        },
        "dataset_info": dataset_info,
        "metrics": metrics,
    }

    # Save locally
    save_artifacts_locally(pipeline, metadata, args.local_dir, timestamp)

    # Save to GCS
    model_blob_name = f"trained_models/breast_cancer_gb_{timestamp}.joblib"
    metadata_blob_name = (
        f"trained_models/breast_cancer_gb_meta_{timestamp}.json"
    )
    save_artifacts_to_gcs(
        pipeline,
        metadata,
        args.bucket,
        model_blob_name,
        metadata_blob_name,
    )
    print("✅ ML Pipeline Run Finished Successfully.")


if __name__ == "__main__":
    main()
