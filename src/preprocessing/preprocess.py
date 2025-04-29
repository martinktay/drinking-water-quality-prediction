import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os
from pathlib import Path
import pickle
import logging
from typing import Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/preprocessing.log'),
        logging.StreamHandler()
    ]
)


def validate_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate the input data for required columns and data types."""
    required_columns = [
        'pH', 'Iron', 'Nitrate', 'Chloride', 'Lead', 'Zinc',
        'Turbidity', 'Fluoride', 'Copper', 'Sulfate', 'Conductivity',
        'Chlorine', 'Total Dissolved Solids', 'Water Temperature',
        'Air Temperature', 'Target'
    ]

    issues = []

    # Check for required columns
    missing_columns = [
        col for col in required_columns if col not in df.columns]
    if missing_columns:
        issues.append(f"Missing required columns: {missing_columns}")

    # Check data types
    numeric_columns = [col for col in required_columns if col != 'Target']
    for col in numeric_columns:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            issues.append(
                f"Column {col} should be numeric but has type {df[col].dtype}")

    # Check for negative values in numeric columns (except temperature)
    for col in numeric_columns:
        if col not in ['Air Temperature', 'Water Temperature'] and col in df.columns and (df[col] < 0).any():
            issues.append(f"Column {col} contains negative values")

    # Check target column
    if 'Target' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['Target']):
            issues.append("Target column should be numeric")
        elif not set(df['Target'].unique()).issubset({0, 1}):
            issues.append("Target column should contain only 0 and 1 values")

    return len(issues) == 0, issues


def load_data():
    """Load the raw dataset."""
    df = pd.read_csv('data/raw/water_quality_dataset_100k.csv')
    print(f"Loaded data shape: {df.shape}")
    return df


def plot_feature_distributions(df: pd.DataFrame, title: str, output_dir: str) -> None:
    """Plot distributions of numeric features."""
    logging.info(f"Creating distribution plots for {title}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get numeric columns (excluding target)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if 'Target' in numeric_cols:
        numeric_cols = numeric_cols.drop('Target')

    # Take a sample of 10000 rows for visualization
    df_sample = df.sample(n=min(10000, len(df)), random_state=42)

    # Set up the figure
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()

    # Plot each feature
    for i, col in enumerate(numeric_cols):
        ax = axes[i]

        # Plot histogram with KDE
        sns.histplot(data=df_sample, x=col, kde=True, ax=ax)

        # Add title and labels
        ax.set_title(f'{col} Distribution')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')

        # Add skewness and kurtosis
        skewness = df[col].skew()
        kurtosis = df[col].kurtosis()
        ax.text(0.05, 0.95, f'Skew: {skewness:.2f}\nKurt: {kurtosis:.2f}',
                transform=ax.transAxes, verticalalignment='top')

    # Remove empty subplots
    for i in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[i])

    # Adjust layout and save
    plt.tight_layout()
    output_path = os.path.join(
        output_dir, f'{title.lower().replace(" ", "_")}_distributions.png')
    plt.savefig(output_path)
    plt.close()

    logging.info(f"Distribution plots saved to {output_path}")


def plot_missing_values(df: pd.DataFrame, title: str, output_dir: str) -> None:
    """Plot missing values heatmap."""
    logging.info(f"Creating missing values plot for {title}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Calculate missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if len(missing) > 0:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=missing.index, y=missing.values)
        plt.title(f'Missing Values - {title}')
        plt.xticks(rotation=45)
        plt.ylabel('Number of Missing Values')
        plt.tight_layout()
        plt.savefig(os.path.join(
            output_dir, f'{title.lower().replace(" ", "_")}_missing_values.png'))
        plt.close()

        logging.info(f"Missing values plot saved to {output_dir}")
    else:
        logging.info("No missing values found")


def plot_outliers(df: pd.DataFrame, title: str, output_dir: str) -> None:
    """Plot boxplots to show outliers."""
    logging.info(f"Creating outlier plots for {title}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Take a sample of 10000 rows for visualization
    df_sample = df.sample(n=min(10000, len(df)), random_state=42)

    # Get numeric columns (excluding target)
    numeric_cols = df_sample.select_dtypes(
        include=['float64', 'int64']).columns
    if 'Target' in numeric_cols:
        numeric_cols = numeric_cols.drop('Target')

    # Set up the figure
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()

    # Plot each feature
    for i, col in enumerate(numeric_cols):
        ax = axes[i]

        # Plot boxplot
        sns.boxplot(y=df_sample[col], ax=ax)

        # Add title and labels
        ax.set_title(f'{col} Boxplot')
        ax.set_ylabel('Value')

        # Calculate and display outlier count
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        ax.text(0.05, 0.95, f'Outliers: {outliers}',
                transform=ax.transAxes, verticalalignment='top')

    # Remove empty subplots
    for i in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[i])

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(
        output_dir, f'{title.lower().replace(" ", "_")}_outliers.png'))
    plt.close()

    logging.info(f"Outlier plots saved to {output_dir}")


def plot_correlations(df: pd.DataFrame, title: str, output_dir: str) -> None:
    """Plot correlation matrix and feature correlations with target."""
    logging.info(f"Creating correlation plots for {title}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Take a sample of 10000 rows for visualization
    df_sample = df.sample(n=min(10000, len(df)), random_state=42)

    # Get numeric columns
    numeric_cols = df_sample.select_dtypes(
        include=['float64', 'int64']).columns

    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    correlation_matrix = df_sample[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True,
                cmap='coolwarm', center=0, fmt='.2f')
    plt.title(f'Correlation Matrix - {title}')
    plt.tight_layout()
    plt.savefig(os.path.join(
        output_dir, f'{title.lower().replace(" ", "_")}_correlation_matrix.png'))
    plt.close()

    # Plot correlations with target if available
    if 'Target' in numeric_cols:
        plt.figure(figsize=(10, 6))
        target_correlations = correlation_matrix['Target'].drop('Target')
        target_correlations.sort_values().plot(kind='bar')
        plt.title(f'Feature Correlations with Target - {title}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(
            output_dir, f'{title.lower().replace(" ", "_")}_target_correlations.png'))
        plt.close()

    logging.info(f"Correlation plots saved to {output_dir}")


def plot_target_distribution(df: pd.DataFrame, title: str, output_dir: str) -> None:
    """Plot target variable distribution."""
    logging.info(f"Creating target distribution plot for {title}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    if 'Target' in df.columns:
        plt.figure(figsize=(8, 6))
        target_counts = df['Target'].value_counts()
        sns.barplot(x=target_counts.index, y=target_counts.values)
        plt.title(f'Target Distribution - {title}')
        plt.xlabel('Target (0: Not Potable, 1: Potable)')
        plt.ylabel('Count')

        # Add percentage labels
        total = len(df)
        for i, v in enumerate(target_counts.values):
            plt.text(i, v, f'{v/total:.1%}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(
            output_dir, f'{title.lower().replace(" ", "_")}_target_distribution.png'))
        plt.close()

        logging.info(f"Target distribution plot saved to {output_dir}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the dataset."""
    logging.info("Starting data cleaning...")

    # Make a copy to avoid modifying the original dataframe
    df = df.copy()

    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Log initial statistics
    logging.info(f"Initial shape: {df.shape}")
    logging.info("Initial missing values:")
    logging.info(df.isnull().sum())

    # Fill missing values
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in numeric_cols:
        if col != 'Target':  # Don't fill missing values in target
            median = df[col].median()
            missing_count = df[col].isnull().sum()
            df[col] = df[col].fillna(median)
            logging.info(
                f"Filled {missing_count} missing values in {col} with median {median:.2f}")

    for col in categorical_cols:
        if col == 'Source':
            df[col] = df[col].fillna('Unknown')
            logging.info(f"Filled missing values in Source with 'Unknown'")
        else:
            mode = df[col].mode()[0]
            missing_count = df[col].isnull().sum()
            df[col] = df[col].fillna(mode)
            logging.info(
                f"Filled {missing_count} missing values in {col} with mode {mode}")

    # Handle outliers using IQR method for numeric columns
    for col in numeric_cols:
        # Don't handle outliers in target and temperatures
        if col != 'Target' and col not in ['Air Temperature', 'Water Temperature']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = ((df[col] < lower_bound) |
                        (df[col] > upper_bound)).sum()
            if outliers > 0:
                logging.info(
                    f"Found {outliers} outliers in {col} (IQR method)")
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    # Log final statistics
    logging.info(f"Final shape: {df.shape}")
    logging.info("Remaining missing values:")
    logging.info(df.isnull().sum())

    logging.info("Data cleaning completed successfully")
    return df


def create_preprocessor():
    """Create and return a preprocessor for the numeric features."""
    numeric_features = ['pH', 'Iron', 'Nitrate', 'Chloride', 'Lead', 'Zinc',
                        'Turbidity', 'Fluoride', 'Copper', 'Sulfate', 'Conductivity',
                        'Chlorine', 'Total Dissolved Solids', 'Water Temperature',
                        'Air Temperature']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features)
        ])

    return preprocessor


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """Save the processed dataset with proper formatting."""
    logging.info(f"Saving processed data to {output_path}...")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Round float columns to 6 decimal places
    float_cols = df.select_dtypes(include=['float64']).columns
    for col in float_cols:
        df[col] = df[col].round(6)

    # Save with proper formatting
    df.to_csv(
        output_path,
        index=False,
        float_format='%.6f',
        na_rep='NA',
        quoting=1
    )

    logging.info(f"Data saved successfully to {output_path}")


def preprocess_data(input_path: str, output_path: str) -> pd.DataFrame:
    """Main function to preprocess the water quality dataset."""
    logging.info(f"Starting preprocessing pipeline for {input_path}")

    try:
        # Read the CSV file
        df = pd.read_csv(input_path)
        logging.info(f"Successfully loaded data with shape {df.shape}")

        # Validate data
        is_valid, issues = validate_data(df)
        if not is_valid:
            error_msg = "Data validation failed:\n" + "\n".join(issues)
            logging.error(error_msg)
            raise ValueError(error_msg)

        # Clean the data
        df_cleaned = clean_data(df)

        # Save the processed data
        save_processed_data(df_cleaned, output_path)

        logging.info("Preprocessing completed successfully")
        return df_cleaned

    except Exception as e:
        logging.error(f"Error during preprocessing: {str(e)}")
        raise


if __name__ == "__main__":
    # Define input and output paths
    input_path = "data/raw/water_quality_dataset_100k.csv"
    output_path = "data/processed/processed_data.csv"

    # Run preprocessing
    df_processed = preprocess_data(input_path, output_path)
