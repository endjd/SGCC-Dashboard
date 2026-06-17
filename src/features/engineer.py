"""
Feature Engineering Module
==========================
Extracts engineered features from preprocessed consumption data for the
SGCC electricity theft detection model. Implements all feature categories:

    1. Missing & Zero Patterns
    2. Activity Measures
    3. Statistical Summary Features
    4. Rolling Window Features
    5. Volatility & Irregularity Features

This module refactors the feature engineering logic from inference_pipeline.py
into reusable, testable functions.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.stats
from scipy.signal import find_peaks

from ..utils.logger import get_logger

logger = get_logger("sgcc.features.engineer")


# =============================================================================
# Helper Functions
# =============================================================================

def longest_value_streak(arr: np.ndarray, value: float, greater_than: bool = False) -> int:
    """
    Compute the longest consecutive streak where elements equal (or exceed) a value.

    Args:
        arr: 1D array of numeric values.
        value: Target value for comparison.
        greater_than: If True, count streaks where elements > value.
                      If False, count streaks where elements == value.

    Returns:
        int: Length of the longest consecutive matching streak.
    """
    max_streak = 0
    current_streak = 0

    if greater_than:
        for x in arr:
            if x > value:
                current_streak += 1
            else:
                max_streak = max(max_streak, current_streak)
                current_streak = 0
    else:
        for x in arr:
            if x == value:
                current_streak += 1
            else:
                max_streak = max(max_streak, current_streak)
                current_streak = 0

    return max(max_streak, current_streak)


def calculate_entropy(data: np.ndarray) -> float:
    """
    Calculate the Shannon entropy of a consumption distribution.

    Uses histogram-based estimation of the probability distribution.

    Args:
        data: 1D array of consumption values.

    Returns:
        float: Shannon entropy value (0.0 for constant data).
    """
    if data.size == 0 or (data == data[0]).all():
        return 0.0
    counts, _ = np.histogram(data, bins="auto")
    counts = counts[counts > 0]
    probabilities = counts / counts.sum()
    return float(scipy.stats.entropy(probabilities))


def calculate_peaks(series: np.ndarray) -> int:
    """
    Count the number of local maxima (peaks) in a consumption series.

    Args:
        series: 1D array of consumption values.

    Returns:
        int: Number of detected peaks.
    """
    peaks, _ = find_peaks(series)
    return len(peaks)


def calculate_rolling_features(
    series: pd.Series,
    window_sizes: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Compute rolling window statistics for a consumption time series.

    For each window size, calculates the mean of rolling means and
    rolling standard deviations. Also computes the 7-day to 90-day
    rolling mean ratio.

    Args:
        series: Time series with DatetimeIndex.
        window_sizes: List of window sizes in days.

    Returns:
        Dict[str, float]: Dictionary of rolling feature name-value pairs.
    """
    if window_sizes is None:
        window_sizes = [7, 30, 90]

    features: Dict[str, float] = {}

    for window in window_sizes:
        rol_mean = series.rolling(window=window).mean()
        rol_std = series.rolling(window=window).std()

        mean_rol_mean = rol_mean.mean()
        mean_rol_std = rol_std.mean()

        features[f"mean_{window}d_rolling_mean"] = (
            mean_rol_mean if not pd.isna(mean_rol_mean) else 0.0
        )
        features[f"mean_{window}d_rolling_std"] = (
            mean_rol_std if not pd.isna(mean_rol_std) else 0.0
        )

    # Ratio of short-term to long-term trend
    mean_7d = features.get("mean_7d_rolling_mean", 0.0)
    mean_90d = features.get("mean_90d_rolling_mean", 0.0)
    features["ratio_7d_to_90d_mean"] = (
        mean_7d / mean_90d if mean_90d != 0 else 0.0
    )

    return features


def get_missing_features(row_data: np.ndarray) -> tuple:
    """
    Compute missing-value pattern features from raw (un-imputed) data.

    Args:
        row_data: 1D array of original consumption values (may contain NaNs).

    Returns:
        Tuple[int, int]: (longest_missing_streak, num_missing_blocks).
    """
    is_missing = pd.isna(row_data)

    # Longest consecutive missing streak
    max_streak = 0
    current_streak = 0
    for val in is_missing:
        if val:
            current_streak += 1
        else:
            max_streak = max(max_streak, current_streak)
            current_streak = 0
    longest_missing_streak = max(max_streak, current_streak)

    # Number of distinct missing blocks
    num_blocks = 0
    in_block = False
    for val in is_missing:
        if val and not in_block:
            num_blocks += 1
            in_block = True
        elif not val:
            in_block = False

    return longest_missing_streak, num_blocks


# =============================================================================
# Main Feature Engineering Function
# =============================================================================

def engineer_features(
    df: pd.DataFrame,
    imputed_matrix: np.ndarray,
    original_matrix: np.ndarray,
    sorted_date_cols: List[str],
    id_column: str = "CONS_NO",
    rolling_windows: Optional[List[int]] = None,
    missing_threshold: float = 0.60,
) -> pd.DataFrame:
    """
    Engineer all features from preprocessed consumption data.

    Generates 5 categories of features:
        1. Missing/Zero patterns (from original data)
        2. Activity measures (from imputed data)
        3. Statistical summary (from imputed data)
        4. Rolling window statistics (from imputed data)
        5. Volatility/irregularity (from imputed data)

    Args:
        df: Original DataFrame containing at least the ID column.
        imputed_matrix: Imputed consumption array, shape (n_customers, n_dates).
        original_matrix: Original consumption array (with NaNs).
        sorted_date_cols: Chronologically sorted date column names.
        id_column: Customer identifier column name.
        rolling_windows: Window sizes for rolling features.
        missing_threshold: Threshold for low_data_flag.

    Returns:
        pd.DataFrame: Feature DataFrame with one row per customer,
                       indexed to align with the input DataFrame.
    """
    if rolling_windows is None:
        rolling_windows = [7, 30, 90]

    n_customers = imputed_matrix.shape[0]
    n_dates = imputed_matrix.shape[1]

    logger.info("Engineering features for %d customers", n_customers)

    features_df = df[[id_column]].copy().reset_index(drop=True)

    # -----------------------------------------------------------------
    # 1. Missing/Zero Pattern Features (from original data)
    # -----------------------------------------------------------------
    original_raw = df[sorted_date_cols].values
    original_missing_rate = 1 - (
        (~pd.isna(original_raw)).sum(axis=1) / n_dates
    )
    features_df["initial_missing_ratio"] = original_missing_rate

    # Missing pattern features
    missing_pattern_features = np.apply_along_axis(
        get_missing_features, 1, original_raw
    )
    features_df["longest_missing_streak"] = missing_pattern_features[:, 0]
    features_df["num_missing_blocks"] = missing_pattern_features[:, 1]

    # -----------------------------------------------------------------
    # 2. Activity Measures (from imputed data)
    # -----------------------------------------------------------------
    features_df["num_active_days"] = np.sum(imputed_matrix > 0, axis=1)
    features_df["zero_days_count"] = np.sum(imputed_matrix == 0, axis=1)
    features_df["zero_days_ratio"] = features_df["zero_days_count"] / n_dates
    features_df["longest_active_streak"] = np.apply_along_axis(
        lambda x: longest_value_streak(x, value=0, greater_than=True),
        1,
        imputed_matrix,
    )
    features_df["longest_zero_streak"] = np.apply_along_axis(
        lambda x: longest_value_streak(x, value=0, greater_than=False),
        1,
        imputed_matrix,
    )

    # -----------------------------------------------------------------
    # 3. Statistical Summary Features (from imputed data)
    # -----------------------------------------------------------------
    features_df["consumption_variance"] = np.apply_along_axis(
        np.var, 1, imputed_matrix
    )
    features_df["consumption_cv"] = [
        np.std(row) / np.mean(row) if np.mean(row) != 0 else 0.0
        for row in imputed_matrix
    ]
    features_df["consumption_entropy"] = np.apply_along_axis(
        calculate_entropy, 1, imputed_matrix
    )
    features_df["consumption_mean"] = np.apply_along_axis(
        np.mean, 1, imputed_matrix
    )
    features_df["consumption_std"] = np.apply_along_axis(
        np.std, 1, imputed_matrix
    )
    features_df["consumption_min"] = np.apply_along_axis(
        np.min, 1, imputed_matrix
    )
    features_df["consumption_max"] = np.apply_along_axis(
        np.max, 1, imputed_matrix
    )
    features_df["consumption_median"] = np.apply_along_axis(
        np.median, 1, imputed_matrix
    )
    features_df["consumption_iqr"] = np.apply_along_axis(
        lambda x: np.percentile(x, 75) - np.percentile(x, 25),
        1,
        imputed_matrix,
    )
    features_df["consumption_slope"] = np.apply_along_axis(
        lambda x: (
            scipy.stats.linregress(np.arange(len(x)), x)[0]
            if np.std(x) != 0
            else 0.0
        ),
        1,
        imputed_matrix,
    )

    # Low data flag
    features_df["low_data_flag"] = (
        features_df["initial_missing_ratio"] > missing_threshold
    ).astype(int)

    # -----------------------------------------------------------------
    # 4. Rolling Window Features (from imputed data)
    # -----------------------------------------------------------------
    date_index = pd.to_datetime(sorted_date_cols)
    rolling_features_data = []

    for i in range(n_customers):
        customer_series = pd.Series(
            imputed_matrix[i, :], index=date_index
        )
        rolling_features_data.append(
            calculate_rolling_features(customer_series, rolling_windows)
        )

    rolling_features_df = pd.DataFrame(rolling_features_data)
    features_df = pd.concat([features_df, rolling_features_df], axis=1)

    # -----------------------------------------------------------------
    # 5. Volatility & Irregularity Features (from imputed data)
    # -----------------------------------------------------------------
    features_df["num_peaks"] = np.apply_along_axis(
        calculate_peaks, 1, imputed_matrix
    )

    logger.info(
        "Feature engineering complete: %d features generated",
        features_df.shape[1] - 1,  # Exclude ID column
    )

    return features_df
