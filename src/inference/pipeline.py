"""
Inference Pipeline Module
=========================
Provides the refactored inference pipeline that uses the modular src/
components for data preprocessing, feature engineering, and prediction.

This module is a drop-in replacement for the original inference_pipeline.py,
maintaining full backwards compatibility while leveraging the modular architecture.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from ..data.preprocessor import (
    detect_and_sort_date_columns,
    impute_customer_series,
    preprocess_consumption_data,
)
from ..features.engineer import engineer_features
from ..utils.logger import get_logger

logger = get_logger("sgcc.inference.pipeline")


class InferencePipeline:
    """
    End-to-end inference pipeline for SGCC electricity theft detection.

    Encapsulates model loading, preprocessing, feature engineering,
    scaling, and prediction into a single reusable class.

    Attributes:
        model: Loaded XGBoost classifier.
        scaler: Loaded MinMaxScaler.
        selected_features: List of feature column names.
        threshold: Optimal classification threshold.

    Example:
        >>> pipeline = InferencePipeline(model_dir=".")
        >>> results = pipeline.predict(raw_df)
        >>> flagged = results[results["Predicted_FLAG"] == 1]
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        features_path: Optional[str] = None,
        threshold_path: Optional[str] = None,
    ) -> None:
        """
        Initialize the inference pipeline by loading all artifacts.

        Args:
            model_dir: Directory containing all artifact files.
                       If provided, individual paths are constructed relative
                       to this directory.
            model_path: Path to the model pickle file.
            scaler_path: Path to the scaler pickle file.
            features_path: Path to the features JSON file.
            threshold_path: Path to the threshold JSON file.
        """
        if model_dir is not None:
            base = Path(model_dir)
            model_path = model_path or str(base / "best_xgb_model.pkl")
            scaler_path = scaler_path or str(base / "min_max_scaler.pkl")
            features_path = features_path or str(base / "selected_feature_columns.json")
            threshold_path = threshold_path or str(base / "optimal_threshold_tuned.json")

        # Load artifacts
        logger.info("Loading model artifacts...")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        with open(features_path, "r") as f:
            self.selected_features: List[str] = json.load(f)

        with open(threshold_path, "r") as f:
            self.threshold: float = json.load(f)

        logger.info(
            "Pipeline initialized: model=%s, %d features, threshold=%.6f",
            type(self.model).__name__,
            len(self.selected_features),
            self.threshold,
        )

    def predict(
        self,
        raw_df: pd.DataFrame,
        return_features: bool = False,
        id_column: str = "CONS_NO",
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Run the full inference pipeline on raw customer data.

        Args:
            raw_df: DataFrame with customer IDs and date columns.
            return_features: If True, also return the engineered features DataFrame.
            id_column: Name of the customer ID column.

        Returns:
            pd.DataFrame: Input DataFrame with 'Predicted_Probability' and
                          'Predicted_FLAG' columns appended.
            If return_features=True, returns (predictions_df, features_df).
        """
        processed_df = raw_df.copy()
        exclude_cols = [id_column]

        # Step 1: Detect date columns
        sorted_date_cols = detect_and_sort_date_columns(
            processed_df, exclude_cols
        )

        if not sorted_date_cols:
            logger.warning("No date columns found for feature engineering")
            processed_df["Predicted_Probability"] = np.nan
            processed_df["Predicted_FLAG"] = -1
            if return_features:
                return processed_df, pd.DataFrame()
            return processed_df

        logger.info(
            "Processing %d customers across %d dates",
            processed_df.shape[0],
            len(sorted_date_cols),
        )

        # Step 2: Preprocess and impute
        original_matrix, imputed_matrix, sorted_date_cols = (
            preprocess_consumption_data(
                processed_df,
                id_column=id_column,
                exclude_cols=exclude_cols,
            )
        )

        # Step 3: Engineer features
        features_df = engineer_features(
            df=processed_df,
            imputed_matrix=imputed_matrix,
            original_matrix=original_matrix,
            sorted_date_cols=sorted_date_cols,
            id_column=id_column,
        )

        # Step 4: Select and scale features
        X_processed = features_df[self.selected_features]
        X_scaled = self.scaler.transform(X_processed)
        X_scaled_df = pd.DataFrame(
            X_scaled,
            columns=self.selected_features,
            index=X_processed.index,
        )

        # Step 5: Predict
        probabilities = self.model.predict_proba(X_scaled_df)[:, 1]

        processed_df["Predicted_Probability"] = probabilities
        processed_df["Predicted_FLAG"] = (
            probabilities >= self.threshold
        ).astype(int)

        n_flagged = int(processed_df["Predicted_FLAG"].sum())
        logger.info(
            "Prediction complete: %d/%d flagged as theft (%.1f%%)",
            n_flagged,
            len(processed_df),
            100 * n_flagged / len(processed_df) if len(processed_df) > 0 else 0,
        )

        if return_features:
            return processed_df, features_df
        return processed_df

    def get_info(self) -> Dict[str, Any]:
        """
        Return metadata about the loaded pipeline configuration.

        Returns:
            Dict with model type, feature count, threshold, and feature names.
        """
        return {
            "model_type": type(self.model).__name__,
            "n_features": len(self.selected_features),
            "threshold": self.threshold,
            "features": self.selected_features,
        }


# =============================================================================
# Module-level convenience function (backwards compatible)
# =============================================================================

# Lazy-loaded singleton pipeline instance
_pipeline_instance: Optional[InferencePipeline] = None


def _get_default_project_root() -> Path:
    """Find the project root by looking for the model artifacts."""
    current = Path(__file__).resolve().parent
    for _ in range(5):
        if (current / "best_xgb_model.pkl").exists():
            return current
        current = current.parent
    # Fallback: 3 levels up from this file
    return Path(__file__).resolve().parent.parent.parent


def predict_customer_theft(
    raw_customer_df: pd.DataFrame,
    return_features: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Module-level convenience function for backwards compatibility with
    the original inference_pipeline.py.

    Lazily initializes an InferencePipeline singleton on first call.

    Args:
        raw_customer_df: Raw customer consumption DataFrame.
        return_features: If True, return (predictions_df, features_df).

    Returns:
        pd.DataFrame or Tuple: Prediction results.
    """
    global _pipeline_instance

    if _pipeline_instance is None:
        project_root = _get_default_project_root()
        _pipeline_instance = InferencePipeline(
            model_dir=str(project_root)
        )

    return _pipeline_instance.predict(
        raw_customer_df, return_features=return_features
    )
