import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os


def prepare_data():
    # Read the dataset
    print("Reading dataset...")
    df = pd.read_csv('../water_quality_dataset_100k.csv')

    # Handle categorical variables
    print("Processing categorical variables...")

    # Convert 'Color' to one-hot encoding
    color_dummies = pd.get_dummies(df['Color'], prefix='Color')
    df = pd.concat([df, color_dummies], axis=1)
    df.drop('Color', axis=1, inplace=True)

    # Convert 'Source' to one-hot encoding
    source_dummies = pd.get_dummies(df['Source'], prefix='Source')
    df = pd.concat([df, source_dummies], axis=1)
    df.drop('Source', axis=1, inplace=True)

    # Convert 'Month' to one-hot encoding
    month_dummies = pd.get_dummies(df['Month'], prefix='Month')
    df = pd.concat([df, month_dummies], axis=1)
    df.drop('Month', axis=1, inplace=True)

    # Drop the Index column if it exists
    if 'Index' in df.columns:
        df.drop('Index', axis=1, inplace=True)

    # Handle missing values
    print("Handling missing values...")
    df = df.fillna(df.mean())

    # Split features and target
    X = df.drop('Target', axis=1)
    y = df['Target']

    # Split into train and test sets
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create data directory if it doesn't exist
    os.makedirs('../data', exist_ok=True)

    # Save the splits
    print("Saving data splits...")
    X_train.to_csv('../data/X_train.csv', index=False)
    X_test.to_csv('../data/X_test.csv', index=False)
    y_train.to_csv('../data/y_train.csv', index=False)
    y_test.to_csv('../data/y_test.csv', index=False)

    print("\nData preparation completed successfully!")
    print(f"Training set size: {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples")
    print(f"Number of features: {X_train.shape[1]}")
    print("\nFeature names:")
    print(X_train.columns.tolist())


if __name__ == "__main__":
    prepare_data()
