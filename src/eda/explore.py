import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os


def load_data(file_path):
    """Load the dataset from the specified path."""
    return pd.read_csv(file_path)


def create_visualizations(df, output_dir):
    """Create and save various visualizations of the dataset."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # 1. Target Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Target', data=df)
    plt.title('Distribution of Safe vs Unsafe Water')
    plt.savefig(f'{output_dir}/target_distribution.png')
    plt.close()

    # 2. Feature Distributions
    numeric_features = ['pH', 'Iron', 'Nitrate', 'Chloride', 'Lead', 'Zinc',
                        'Color', 'Turbidity', 'Fluoride', 'Copper', 'Odor',
                        'Sulfate', 'Conductivity', 'Chlorine']

    for feature in numeric_features:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=feature, hue='Target', kde=True)
        plt.title(f'Distribution of {feature} by Target')
        plt.savefig(f'{output_dir}/{feature}_distribution.png')
        plt.close()

    # 3. Correlation Matrix
    plt.figure(figsize=(15, 12))
    correlation_matrix = df[numeric_features + ['Target']].corr()
    sns.heatmap(correlation_matrix, annot=True,
                cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_matrix.png')
    plt.close()

    # 4. Box Plots
    for feature in numeric_features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Target', y=feature, data=df)
        plt.title(f'Box Plot of {feature} by Target')
        plt.savefig(f'{output_dir}/{feature}_boxplot.png')
        plt.close()

    # 5. Interactive Scatter Plots (using plotly)
    for feature in numeric_features:
        fig = px.scatter(df, x=feature, y='Target',
                         color='Target', title=f'{feature} vs Target')
        fig.write_html(f'{output_dir}/{feature}_scatter.html')

    # 6. Pair Plot (sample 1000 points for performance)
    sample_df = df.sample(n=min(1000, len(df)), random_state=42)
    plt.figure(figsize=(20, 20))
    sns.pairplot(sample_df, hue='Target', vars=numeric_features)
    plt.savefig(f'{output_dir}/pairplot.png')
    plt.close()


def generate_summary_statistics(df, output_path):
    """Generate and save summary statistics of the dataset."""
    summary = df.describe().T
    summary['missing_values'] = df.isnull().sum()
    summary['unique_values'] = df.nunique()
    summary.to_csv(output_path)


def main():
    # Create necessary directories
    os.makedirs('reports/figures', exist_ok=True)

    # Load data
    print("Loading data...")
    df = load_data('data/processed/water_quality_cleaned.csv')

    # Generate visualizations
    print("Creating visualizations...")
    create_visualizations(df, 'reports/figures')

    # Generate summary statistics
    print("Generating summary statistics...")
    generate_summary_statistics(df, 'reports/summary_statistics.csv')

    print("EDA completed successfully!")


if __name__ == "__main__":
    main()
