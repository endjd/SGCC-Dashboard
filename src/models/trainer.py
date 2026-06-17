"""
Model Training Module
=====================
Provides training functions for XGBoost and LightGBM classifiers
with cross-validation support, threshold optimization, and artifact saving.

Designed for the SGCC electricity theft detection binary classification task.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
)

from ..utils.logger import get_logger

logger = get_logger("sgcc.models.trainer")


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Train an XGBoost classifier for electricity theft detection.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels (0=normal, 1=theft).
        params: Optional dict of XGBoost hyperparameters.
                If None, uses tuned defaults from the graduation project.

    Returns:
        Trained XGBClassifier instance.
    """
    from xgboost import XGBClassifier

    default_params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": 3.0,
        "random_state": 42,
        "eval_metric": "logloss",
        "use_label_encoder": False,
    }

    if params:
        default_params.update(params)

    logger.info("Training XGBoost with params: %s", default_params)
    start_time = time.time()

    model = XGBClassifier(**default_params)
    model.fit(X_train, y_train)

    elapsed = time.time() - start_time
    logger.info("XGBoost training completed in %.2f seconds", elapsed)

    return model


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Train a LightGBM classifier for electricity theft detection.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels (0=normal, 1=theft).
        params: Optional dict of LightGBM hyperparameters.

    Returns:
        Trained LGBMClassifier instance.
    """
    from lightgbm import LGBMClassifier

    default_params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": 3.0,
        "random_state": 42,
        "verbose": -1,
    }

    if params:
        default_params.update(params)

    logger.info("Training LightGBM with params: %s", default_params)
    start_time = time.time()

    model = LGBMClassifier(**default_params)
    model.fit(X_train, y_train)

    elapsed = time.time() - start_time
    logger.info("LightGBM training completed in %.2f seconds", elapsed)

    return model


def cross_validate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    scoring: str = "f1",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Perform stratified K-fold cross-validation on a classifier.

    Args:
        model: Scikit-learn compatible classifier (unfitted or fitted).
        X: Feature matrix.
        y: Target labels.
        n_splits: Number of cross-validation folds.
        scoring: Scoring metric for evaluation.
        random_state: Random seed for reproducibility.

    Returns:
        Dict with keys: 'scores', 'mean', 'std', 'n_splits', 'scoring'.
    """
    logger.info(
        "Running %d-fold cross-validation with scoring='%s'",
        n_splits,
        scoring,
    )

    cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

    results = {
        "scores": scores.tolist(),
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "n_splits": n_splits,
        "scoring": scoring,
    }

    logger.info(
        "CV Results: %s = %.4f (+/- %.4f)",
        scoring,
        results["mean"],
        results["std"],
    )

    return results


def find_optimal_threshold(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Tuple[float, float]:
    """
    Find the optimal classification threshold by maximizing F1-score
    on a validation set using the precision-recall curve.

    Args:
        model: Trained classifier with predict_proba method.
        X_val: Validation feature matrix.
        y_val: Validation labels.

    Returns:
        Tuple[float, float]: (optimal_threshold, best_f1_score).
    """
    probabilities = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, probabilities)

    # Compute F1 for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

    best_idx = np.argmax(f1_scores)
    optimal_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    best_f1 = float(f1_scores[best_idx])

    logger.info(
        "Optimal threshold: %.6f (F1=%.4f)", optimal_threshold, best_f1
    )

    return optimal_threshold, best_f1


def fit_scaler(
    X_train: pd.DataFrame,
) -> MinMaxScaler:
    """
    Fit a MinMaxScaler on training features.

    Args:
        X_train: Training feature matrix.

    Returns:
        Fitted MinMaxScaler instance.
    """
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    logger.info("MinMaxScaler fitted on %d features", X_train.shape[1])
    return scaler


def save_training_artifacts(
    model: Any,
    scaler: MinMaxScaler,
    selected_features: List[str],
    threshold: float,
    output_dir: str = ".",
    version: str = "v1",
    metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, str]:
    """
    Save all training artifacts (model, scaler, features, threshold, metadata).

    Args:
        model: Trained model to serialize.
        scaler: Fitted MinMaxScaler.
        selected_features: List of feature column names.
        threshold: Optimal classification threshold.
        output_dir: Directory to save artifacts.
        version: Model version string (e.g., 'v1', 'v2').
        metrics: Optional dict of evaluation metrics.

    Returns:
        Dict[str, str]: Mapping of artifact names to file paths.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Save model
    model_path = output_path / "best_xgb_model.pkl"
    joblib.dump(model, model_path)
    paths["model"] = str(model_path)

    # Save scaler
    scaler_path = output_path / "min_max_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    paths["scaler"] = str(scaler_path)

    # Save features
    features_path = output_path / "selected_feature_columns.json"
    with open(features_path, "w") as f:
        json.dump(selected_features, f, indent=2)
    paths["features"] = str(features_path)

    # Save threshold
    threshold_path = output_path / "optimal_threshold_tuned.json"
    with open(threshold_path, "w") as f:
        json.dump(threshold, f)
    paths["threshold"] = str(threshold_path)

    # Save metadata
    metadata = {
        "version": version,
        "model_type": type(model).__name__,
        "timestamp": datetime.now().isoformat(),
        "n_features": len(selected_features),
        "features": selected_features,
        "threshold": threshold,
        "metrics": metrics or {},
        "artifacts": {k: os.path.basename(v) for k, v in paths.items()},
    }

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    paths["metadata"] = str(metadata_path)

    logger.info("Saved %d training artifacts to %s", len(paths), output_path)

    return paths
