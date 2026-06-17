"""
Data Preprocessing Module
=========================
Handles imputation and cleaning of raw consumption time-series data.
Implements the 3-stage imputation strategy from the training pipeline:
    1. Time-based interpolation (limit=3, bidirectional)
    2. Monthly median fill
    3. Overall median fill

This module extracts and refactors the preprocessing logic originally
found in inference_pipeline.py for reusability across training and inference.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger("sgcc.data.preprocessor")


def detect_and_sort_date_columns(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
) -> List[str]:
    """
    Identify date-parseable column names and return them in chronological order.

    Args:
        df: DataFrame whose columns may contain date-formatted names.
        exclude_cols: Column names to exclude from date detection.

    Returns:
        List[str]: Chronologically sorted list of date column names.
    """
    if exclude_cols is None:
        exclude_cols = []

    datetime_name_pairs = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        parsed_date = pd.to_datetime(col, errors="coerce")
        if pd.notna(parsed_date):
            datetime_name_pairs.append((parsed_date, col))

    datetime_name_pairs.sort(key=lambda x: x[0])
    sorted_date_cols = [name for _, name in datetime_name_pairs]

    logger.debug("Detected %d date columns", len(sorted_date_cols))
    return sorted_date_cols


def impute_customer_series(
    series: pd.Series,
    interpolation_method: str = "time",
    interpolation_limit: int = 3,
    interpolation_direction: str = "both",
) -> pd.Series:
    """
    Apply 3-stage imputation to a single customer's consumption time series.

    Stage 1: Time-based interpolation with bounded gap filling.
    Stage 2: Monthly median imputation for remaining NaNs.
    Stage 3: Overall median imputation as final fallback.

    Args:
        series: A pandas Series with a DatetimeIndex containing the
                customer's daily consumption values (may contain NaNs).
        interpolation_method: Interpolation method for stage 1.
        interpolation_limit: Maximum consecutive NaN gap to interpolate.
        interpolation_direction: Direction for interpolation.

    Returns:
        pd.Series: Fully imputed series with no NaN values.
    """
    # Stage 1: Time interpolation
    series_imputed = series.interpolate(
        method=interpolation_method,
        limit=interpolation_limit,
        limit_direction=interpolation_direction,
    )

    # Stage 2: Monthly median
    def _fill_monthly_median(s: pd.Series) -> pd.Series:
        if s.isna().all():
            return s
        return s.fillna(s.median())

    series_imputed = series_imputed.groupby(
        series_imputed.index.month
    ).transform(_fill_monthly_median)

    # Stage 3: Overall median
    overall_median = series_imputed.median()
    series_imputed = series_imputed.fillna(overall_median)

    return series_imputed


def preprocess_consumption_data(
    df: pd.DataFrame,
    id_column: str = "CONS_NO",
    exclude_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Preprocess the raw consumption DataFrame: detect dates, convert to
    numeric, and apply imputation for each customer row.

    Args:
        df: Raw input DataFrame with customer IDs and date columns.
        id_column: Column name for customer identifiers.
        exclude_cols: Additional columns to exclude from date detection.

    Returns:
        Tuple containing:
            - original_matrix (np.ndarray): Original consumption values
              (with NaNs), shape (n_customers, n_dates).
            - imputed_matrix (np.ndarray): Imputed consumption values,
              shape (n_customers, n_dates).
            - sorted_date_cols (List[str]): Chronologically sorted date
              column names.

    Raises:
        ValueError: If no date columns are found in the DataFrame.
    """
    if exclude_cols is None:
        exclude_cols = [id_column]
    elif id_column not in exclude_cols:
        exclude_cols = [id_column] + list(exclude_cols)

    # Detect and sort date columns
    sorted_date_cols = detect_and_sort_date_columns(df, exclude_cols)

    if not sorted_date_cols:
        raise ValueError(
            "No date columns found in the DataFrame. "
            "Ensure column names are in a date-parseable format."
        )

    logger.info(
        "Preprocessing %d customers across %d date columns",
        df.shape[0],
        len(sorted_date_cols),
    )

    # Convert to numeric matrix
    original_matrix = (
        df[sorted_date_cols]
        .apply(pd.to_numeric, errors="coerce")
        .values.astype(float)
    )

    # Apply per-customer imputation
    date_index = pd.to_datetime(sorted_date_cols)
    imputed_rows = []

    for i in range(original_matrix.shape[0]):
        customer_series = pd.Series(original_matrix[i, :], index=date_index)
        imputed_series = impute_customer_series(customer_series)
        imputed_rows.append(imputed_series.values)

    imputed_matrix = np.array(imputed_rows)

    n_original_nans = np.isnan(original_matrix).sum()
    n_remaining_nans = np.isnan(imputed_matrix).sum()

    logger.info(
        "Imputation complete: %d NaNs resolved, %d remaining",
        n_original_nans - n_remaining_nans,
        n_remaining_nans,
    )

    return original_matrix, imputed_matrix, sorted_date_cols
