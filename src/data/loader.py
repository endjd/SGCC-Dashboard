"""
Data Loading Module
===================
Provides utilities for loading and validating SGCC consumption data.
Supports CSV input with schema validation and basic integrity checks.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np

from ..utils.logger import get_logger

logger = get_logger("sgcc.data.loader")


def load_csv(
    file_path: str,
    id_column: str = "CONS_NO",
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load a CSV file containing customer consumption data.

    Args:
        file_path: Path to the CSV file.
        id_column: Name of the customer ID column.
        nrows: Optional number of rows to read (useful for testing).

    Returns:
        pd.DataFrame: Loaded DataFrame with validated structure.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the CSV is empty or missing the ID column.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    logger.info("Loading data from: %s", file_path)

    df = pd.read_csv(file_path, nrows=nrows)

    if df.empty:
        raise ValueError(f"CSV file is empty: {file_path}")

    if id_column not in df.columns:
        raise ValueError(
            f"Required column '{id_column}' not found in CSV. "
            f"Available columns: {list(df.columns[:5])}..."
        )

    logger.info(
        "Loaded %d rows x %d columns from %s",
        df.shape[0],
        df.shape[1],
        file_path.name,
    )

    return df


def load_sample_data(
    sample_path: str = "sample_input.csv",
    id_column: str = "CONS_NO",
) -> Optional[pd.DataFrame]:
    """
    Load the bundled sample input CSV for demonstration purposes.

    Args:
        sample_path: Path to the sample CSV file.
        id_column: Name of the customer ID column.

    Returns:
        pd.DataFrame or None: Sample DataFrame if file exists, else None.
    """
    try:
        return load_csv(sample_path, id_column=id_column)
    except FileNotFoundError:
        logger.warning("Sample data file not found at: %s", sample_path)
        return None


def validate_dataframe(
    df: pd.DataFrame,
    id_column: str = "CONS_NO",
    min_date_columns: int = 10,
) -> Tuple[bool, List[str]]:
    """
    Validate the structure and quality of the input DataFrame.

    Performs the following checks:
        1. ID column exists
        2. Minimum number of date columns present
        3. No duplicate customer IDs
        4. Data types are parseable as numeric consumption values

    Args:
        df: Input DataFrame to validate.
        id_column: Name of the customer ID column.
        min_date_columns: Minimum number of date columns expected.

    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_issues).
            is_valid is True if all checks pass.
    """
    issues: List[str] = []

    # Check 1: ID column
    if id_column not in df.columns:
        issues.append(f"Missing required column: '{id_column}'")

    # Check 2: Date columns
    date_columns = _detect_date_columns(df, exclude_cols=[id_column, "FLAG"])
    if len(date_columns) < min_date_columns:
        issues.append(
            f"Found only {len(date_columns)} date columns "
            f"(minimum required: {min_date_columns})"
        )

    # Check 3: Duplicate IDs
    if id_column in df.columns:
        n_duplicates = df[id_column].duplicated().sum()
        if n_duplicates > 0:
            issues.append(f"Found {n_duplicates} duplicate customer IDs")

    # Check 4: Empty DataFrame
    if df.shape[0] == 0:
        issues.append("DataFrame has no rows")

    # Check 5: Sample numeric check on date columns
    if len(date_columns) > 0:
        sample_col = date_columns[0]
        numeric_vals = pd.to_numeric(df[sample_col], errors="coerce")
        non_numeric_ratio = numeric_vals.isna().sum() / len(df)
        if non_numeric_ratio > 0.5:
            issues.append(
                f"Column '{sample_col}' has {non_numeric_ratio:.0%} "
                f"non-numeric values"
            )

    is_valid = len(issues) == 0

    if is_valid:
        logger.info("Data validation passed: %d customers, %d date columns",
                     df.shape[0], len(date_columns))
    else:
        logger.warning("Data validation failed with %d issue(s):", len(issues))
        for issue in issues:
            logger.warning("  - %s", issue)

    return is_valid, issues


def _detect_date_columns(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
) -> List[str]:
    """
    Detect columns whose names can be parsed as dates.

    Args:
        df: Input DataFrame.
        exclude_cols: Column names to skip.

    Returns:
        List[str]: Column names that are valid dates.
    """
    if exclude_cols is None:
        exclude_cols = []

    date_cols = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        parsed = pd.to_datetime(col, errors="coerce")
        if pd.notna(parsed):
            date_cols.append(col)

    return date_cols
