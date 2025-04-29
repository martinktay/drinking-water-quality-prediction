import pytest
import pandas as pd
import numpy as np
from src.data.clean_data import validate_data_ranges


def test_validate_data_ranges():
    # Create test data
    test_data = pd.DataFrame({
        'pH': [6.5, 7.0, 14.5],  # pH should be between 0-14
        'Iron': [0.5, 1.0, -0.1],  # Iron should be non-negative
        'Nitrate': [10.0, 20.0, 30.0],
        'Target': [0, 1, 0]
    })

    # Test validation
    valid_mask = validate_data_ranges(test_data)
    assert isinstance(valid_mask, pd.Series)
    assert valid_mask.dtype == bool
    assert len(valid_mask) == len(test_data)
    assert valid_mask.sum() == 1  # Only the second row should be valid


def test_empty_dataframe():
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError):
        validate_data_ranges(empty_df)


def test_missing_required_columns():
    df_missing_cols = pd.DataFrame({
        'pH': [7.0],
        'Iron': [0.5]
    })
    with pytest.raises(ValueError):
        validate_data_ranges(df_missing_cols)
