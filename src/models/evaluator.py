"""
Model Evaluation Module
=======================
Provides comprehensive evaluation functions for binary classification models
in the SGCC electricity theft detection context.

Includes:
    - Standard metrics (Precision, Recall, F1, AUC-ROC, PR-AUC)
    - Confusion matrix analysis
    - Classification reports
    - Threshold-based evaluation
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
)

from ..utils.logger import get_logger

logger = get_logger("sgcc.models.evaluator")


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics for binary classification.

    Args:
        y_true: Ground truth labels (0 or 1).
        y_pred: Predicted labels (0 or 1).
        y_proba: Optional predicted probabilities for the positive class.
                 Required for AUC-ROC and PR-AUC calculation.
        threshold: Classification threshold used (for reporting purposes).

    Returns:
        Dict[str, float]: Dictionary of metric names to values.
    """
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": threshold,
    }

    if y_proba is not None:
        try:
            metrics["auc_roc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["auc_roc"] = 0.0
            logger.warning("Could not compute AUC-ROC (single class present)")

        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
        except ValueError:
            metrics["pr_auc"] = 0.0
            logger.warning("Could not compute PR-AUC (single class present)")

    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["true_positives"] = int(tp)
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)

    # Specificity (True Negative Rate)
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    logger.info(
        "Evaluation: Precision=%.4f, Recall=%.4f, F1=%.4f",
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
    )

    return metrics


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None,
    output_dict: bool = False,
) -> str | Dict:
    """
    Generate a detailed classification report.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        target_names: Names for each class label.
        output_dict: If True, returns a dict instead of a string.

    Returns:
        str or dict: Classification report in text or dict format.
    """
    if target_names is None:
        target_names = ["Normal (0)", "Theft (1)"]

    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=output_dict,
        zero_division=0,
    )

    if not output_dict:
        logger.info("Classification Report:\n%s", report)

    return report


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute the confusion matrix and derived statistics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Dict containing 'matrix' (2x2 list), 'labels', and derived rates.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    total = int(tn + fp + fn + tp)

    result = {
        "matrix": cm.tolist(),
        "labels": ["Normal", "Theft"],
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "total_samples": total,
        "true_positive_rate": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "true_negative_rate": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
    }

    logger.info(
        "Confusion Matrix: TP=%d, TN=%d, FP=%d, FN=%d",
        tp, tn, fp, fn,
    )

    return result


def evaluate_at_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """
    Evaluate model predictions at a specific probability threshold.

    Args:
        y_true: Ground truth labels.
        y_proba: Predicted probabilities for the positive class.
        threshold: Classification threshold.

    Returns:
        Dict[str, float]: Evaluation metrics at the given threshold.
    """
    y_pred = (y_proba >= threshold).astype(int)
    return evaluate_model(y_true, y_pred, y_proba, threshold=threshold)


def get_roc_curve_data(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, List[float]]:
    """
    Compute ROC curve data for plotting.

    Args:
        y_true: Ground truth labels.
        y_proba: Predicted probabilities for the positive class.

    Returns:
        Dict with 'fpr', 'tpr', 'thresholds' as lists.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
    }


def get_precision_recall_curve_data(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, List[float]]:
    """
    Compute Precision-Recall curve data for plotting.

    Args:
        y_true: Ground truth labels.
        y_proba: Predicted probabilities for the positive class.

    Returns:
        Dict with 'precision', 'recall', 'thresholds' as lists.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
    }


def compare_models(
    results: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """
    Create a comparison DataFrame of evaluation metrics across models.

    Args:
        results: Dict mapping model names to their metric dicts.
                 Example: {'XGBoost': {...}, 'LightGBM': {...}}

    Returns:
        pd.DataFrame: Comparison table with models as rows and metrics as columns.
    """
    comparison_metrics = [
        "precision", "recall", "f1_score", "auc_roc", "pr_auc", "accuracy"
    ]

    comparison_data = {}
    for model_name, metrics in results.items():
        comparison_data[model_name] = {
            k: metrics.get(k, np.nan) for k in comparison_metrics
        }

    df = pd.DataFrame(comparison_data).T
    df.index.name = "Model"

    logger.info("Model comparison:\n%s", df.to_string())

    return df
