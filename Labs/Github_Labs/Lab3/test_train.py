"""
Unit Tests for Breast Cancer ML Pipeline (Lab 3).
"""

import os
import tempfile
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from train_and_save_model import (
    load_and_prepare_data,
    build_and_train_pipeline,
    evaluate_model,
    save_artifacts_locally,
)


def test_load_and_prepare_data():
    """Verify data loading returns proper shape and stratified split."""
    X_train, X_test, y_train, y_test, info = load_and_prepare_data(
        test_size=0.2, random_state=42
    )

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

    assert len(X_train) + len(X_test) == info["num_samples"]
    assert X_train.shape[1] == 30
    assert X_test.shape[1] == 30
    assert len(info["feature_names"]) == 30
    assert len(info["target_names"]) == 2


def test_build_and_train_pipeline():
    """Verify model training builds pipeline with high accuracy."""
    X_train, X_test, y_train, y_test, _ = load_and_prepare_data(
        random_state=42
    )

    pipeline = build_and_train_pipeline(
        X_train,
        y_train,
        n_estimators=50,
        learning_rate=0.1,
        random_state=42,
    )

    assert isinstance(pipeline, Pipeline)
    assert "scaler" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps

    metrics = evaluate_model(pipeline, X_test, y_test)
    assert metrics["accuracy"] >= 0.90, (
        f"Expected accuracy >= 0.90, got {metrics['accuracy']}"
    )
    assert metrics["f1_score"] >= 0.90, (
        f"Expected f1_score >= 0.90, got {metrics['f1_score']}"
    )
    assert "confusion_matrix" in metrics
    assert "roc_auc" in metrics


def test_model_serialization_and_prediction():
    """Verify saved pipeline reloads and gives identical predictions."""
    X_train, X_test, y_train, y_test, info = load_and_prepare_data(
        random_state=42
    )
    pipeline = build_and_train_pipeline(
        X_train, y_train, n_estimators=30, random_state=42
    )
    original_preds = pipeline.predict(X_test)

    metrics = evaluate_model(pipeline, X_test, y_test)
    metadata = {
        "timestamp": "20261004-test",
        "dataset_info": info,
        "metrics": metrics,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path, meta_path = save_artifacts_locally(
            pipeline, metadata, tmpdir, "20261004-test"
        )

        assert os.path.exists(model_path)
        assert os.path.exists(meta_path)

        loaded_pipeline = joblib.load(model_path)
        reloaded_preds = loaded_pipeline.predict(X_test)

        assert (original_preds == reloaded_preds).all()
