import pandas as pd
import numpy as np
from pathlib import Path
import os


def clean_data():
    # Create necessary directories
    Path('data/processed').mkdir(parents=True, exist_ok=True)

    # Load the raw data
    print("Loading raw data...")
    df = pd.read_csv('data/raw/water_quality_dataset_100k.csv')

    # Remove columns with high percentage of null values (threshold: 30%)
    null_percentages = df.isnull().sum() / len(df) * 100
    columns_to_drop = null_percentages[null_percentages > 30].index
    df = df.drop(columns=columns_to_drop)

    # Remove rows with any remaining null values
    df = df.dropna()

    # Keep only important features based on domain knowledge
    important_features = [
        'pH', 'Iron', 'Nitrate', 'Chloride', 'Lead', 'Zinc',
        'Turbidity', 'Fluoride', 'Copper', 'Sulfate', 'Conductivity',
        'Chlorine', 'Total Dissolved Solids', 'Water Temperature',
        'Air Temperature', 'Target'
    ]

    # Filter columns to keep only important features
    df = df[important_features]

    # Save the cleaned data
    print("Saving cleaned data...")
    df.to_csv('data/processed/processed_data.csv', index=False)

    print(f"Original shape: {df.shape}")
    print(f"Removed columns: {list(columns_to_drop)}")
    print("Data cleaning completed successfully!")


if __name__ == "__main__":
    # Delete old processed data if it exists
    if os.path.exists('data/processed/processed_data.csv'):
        os.remove('data/processed/processed_data.csv')
        print("Deleted old processed data file")

    clean_data()
