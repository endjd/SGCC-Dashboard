"""
Data loading and preprocessing modules for the SGCC pipeline.
"""

from .loader import load_csv, load_sample_data, validate_dataframe
from .preprocessor import (
    detect_and_sort_date_columns,
    impute_customer_series,
    preprocess_consumption_data,
)

__all__ = [
    "load_csv",
    "load_sample_data",
    "validate_dataframe",
    "detect_and_sort_date_columns",
    "impute_customer_series",
    "preprocess_consumption_data",
]
