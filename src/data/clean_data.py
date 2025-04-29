"""Data cleaning and validation module."""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

# Configure logging
logger = logging.getLogger(__name__)

# Define data validation ranges
VALID_RANGES = {
    'pH': (0, 14),
    'Iron': (0, float('inf')),
    'Nitrate': (0, float('inf')),
    'Chloride': (0, float('inf')),
    'Lead': (0, float('inf')),
    'Zinc': (0, float('inf')),
    'Turbidity': (0, float('inf')),
    'Fluoride': (0, float('inf')),
    'Copper': (0, float('inf')),
    'Sulfate': (0, float('inf')),
    'Conductivity': (0, float('inf')),
    'Chlorine': (0, float('inf')),
    'Total Dissolved Solids': (0, float('inf')),
    'Water Temperature': (-50, 100),  # Reasonable range in Celsius
    'Air Temperature': (-50, 100)     # Reasonable range in Celsius
}


def validate_data_ranges(df: pd.DataFrame) -> pd.Series:
    """
    Validate that all values in the dataframe are within acceptable ranges.

    Args:
        df: Input dataframe with water quality parameters

    Returns:
        Boolean mask indicating which rows are valid

    Raises:
        ValueError: If dataframe is empty or missing required columns
    """
    if df.empty:
        raise ValueError("Input dataframe is empty")

    required_columns = list(VALID_RANGES.keys())
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    valid_mask = pd.Series(True, index=df.index)

    for column, (min_val, max_val) in VALID_RANGES.items():
        column_mask = (df[column] >= min_val) & (df[column] <= max_val)
        valid_mask &= column_mask
        invalid_count = (~column_mask).sum()
        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} invalid values in {column}")

    return valid_mask


def clean_dataset(input_path: str, output_path: str) -> Tuple[int, int]:
    """
    Clean the dataset by removing invalid entries and saving the result.

    Args:
        input_path: Path to input CSV file
        output_path: Path to save cleaned CSV file

    Returns:
        Tuple of (total_rows, valid_rows)

    Raises:
        FileNotFoundError: If input file doesn't exist
        PermissionError: If cannot write to output path
    """
    try:
        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)
        total_rows = len(df)

        # Remove unnamed columns
        unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)
            logger.info(f"Removed {len(unnamed_cols)} unnamed columns")

        # Validate data ranges
        valid_mask = validate_data_ranges(df)
        valid_rows = valid_mask.sum()

        # Keep only valid rows
        df = df[valid_mask]

        # Save cleaned data
        df.to_csv(output_path, index=False)
        logger.info(f"Saved cleaned data to {output_path}")

        return total_rows, valid_rows

    except FileNotFoundError:
        logger.error(f"Input file not found: {input_path}")
        raise
    except PermissionError:
        logger.error(f"Permission denied when writing to: {output_path}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during data cleaning: {str(e)}")
        raise
