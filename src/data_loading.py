"""
Data loading and initial exploration module for the Water Quality Classification project.
This module handles data loading, basic exploration, and data quality checks.
"""

from utils import setup_environment
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


def load_data(file_path=None):
    """
    Load the water quality dataset.

    Args:
        file_path (str, optional): Path to the dataset file. If None, uses default path.

    Returns:
        pandas.DataFrame: The loaded dataset
    """
    if file_path is None:
        # Use relative path from project root
        file_path = os.path.join('..', 'water_quality_dataset_100k.csv')

    print("Loading dataset...")
    data = pd.read_csv(file_path)

    # Drop the Index column if it exists
    if 'Index' in data.columns:
        data = data.drop(columns=['Index'])

    print(
        f"Dataset loaded successfully with {len(data)} rows and {len(data.columns)} columns")
    return data


def get_data_summary(df):
    """
    Generate a comprehensive summary of the dataset.

    Args:
        df (pandas.DataFrame): The dataset to analyze

    Returns:
        pandas.DataFrame: Summary statistics including non-null counts, null counts, and data types
    """
    total = len(df)
    missing_pct = ((total - df.count()) / total) * 100

    summary = pd.DataFrame({
        "Non-Null Count": df.count(),
        "Null Count": df.isnull().sum(),
        "Total Rows": total,
        "Data Type": df.dtypes,
        "Unique Values": df.nunique(),
        "Missing Percentage": missing_pct.round(2)
    })

    return summary


def analyze_memory_usage(df):
    """
    Analyze and display memory usage information for the dataset.

    Args:
        df (pandas.DataFrame): The dataset to analyze
    """
    print("\n=== Memory Usage Analysis ===")
    print("\nDetailed DataFrame Information:")
    df.info(verbose=True, show_counts=True, memory_usage='deep')

    # Calculate memory usage per column
    memory_usage = df.memory_usage(deep=True)
    total_memory = memory_usage.sum()

    print(f"\nTotal Memory Usage: {total_memory / 1024**2:.2f} MB")
    print("\nMemory Usage by Column (MB):")
    for col, usage in memory_usage.items():
        if col != 'Index':  # Skip the index
            print(f"{col}: {usage / 1024**2:.2f} MB")


def get_numerical_features(df, target_column='Target'):
    """
    Identify and return numerical features in the dataset.

    Args:
        df (pandas.DataFrame): The dataset to analyze
        target_column (str): Name of the target column to exclude

    Returns:
        list: List of numerical feature column names
    """
    # Get all numerical columns
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove target column if it exists
    if target_column in numerical_features:
        numerical_features.remove(target_column)

    return numerical_features


def plot_numerical_distributions(df, numerical_features, target_column=None, figsize=(20, 15)):
    """
    Create distribution plots for numerical features.

    Args:
        df (pandas.DataFrame): The dataset
        numerical_features (list): List of numerical feature names
        target_column (str, optional): Name of the target column for conditional plots
        figsize (tuple): Figure size (width, height)
    """
    n_features = len(numerical_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    plt.figure(figsize=figsize)

    for i, feature in enumerate(numerical_features, 1):
        plt.subplot(n_rows, n_cols, i)

        if target_column is not None and target_column in df.columns:
            # Create conditional distributions
            for target_value in sorted(df[target_column].unique()):
                sns.histplot(data=df[df[target_column] == target_value],
                             x=feature,
                             kde=True,
                             alpha=0.5,
                             label=f'Target={target_value}')
            plt.legend()
        else:
            # Create simple histogram
            sns.histplot(data=df, x=feature, kde=True)

        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Count')

    plt.tight_layout()
    plt.show()


def plot_numerical_boxplots(df, numerical_features, target_column=None, figsize=(20, 15), horizontal=False):
    """
    Create box plots for numerical features.

    Args:
        df (pandas.DataFrame): The dataset
        numerical_features (list): List of numerical feature names
        target_column (str, optional): Name of the target column for conditional plots
        figsize (tuple): Figure size (width, height)
        horizontal (bool): Whether to create horizontal box plots
    """
    if horizontal:
        # Create a single horizontal box plot for all features
        plt.figure(figsize=figsize)
        sns.boxplot(data=df[numerical_features], orient='h')
        plt.title('Boxplots of Numerical Features')
        plt.xlabel('Value')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()
    else:
        # Create individual box plots in a grid
        n_features = len(numerical_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        plt.figure(figsize=figsize)

        for i, feature in enumerate(numerical_features, 1):
            plt.subplot(n_rows, n_cols, i)

            if target_column is not None and target_column in df.columns:
                # Create conditional box plots
                sns.boxplot(data=df, x=target_column, y=feature)
                plt.title(f'Box Plot of {feature} by Target')
            else:
                # Create simple box plot
                sns.boxplot(data=df, y=feature)
                plt.title(f'Box Plot of {feature}')

        plt.tight_layout()
        plt.show()


def plot_feature_pairs(df, pairs, target_column='Target', figsize=(20, 15)):
    """
    Create scatter plots for pairs of numerical features.

    Args:
        df (pandas.DataFrame): The dataset
        pairs (list): List of tuples containing pairs of feature names
        target_column (str): Name of the target column for coloring
        figsize (tuple): Figure size (width, height)
    """
    print("\n=== Scatter Plots of Feature Pairs ===")

    n_pairs = len(pairs)
    n_cols = 3
    n_rows = (n_pairs + n_cols - 1) // n_cols

    plt.figure(figsize=figsize)

    for i, (x_feature, y_feature) in enumerate(pairs, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.scatterplot(x=x_feature, y=y_feature, data=df,
                        alpha=0.5, hue=target_column)

        plt.xlabel(x_feature)
        plt.ylabel(y_feature)
        plt.title(f'{x_feature} vs {y_feature}')

    plt.suptitle('Scatter Plots of Selected Feature Pairs', fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df, numerical_features, target_column='Target', figsize=(20, 8)):
    """
    Create a correlation matrix heatmap for numerical features.

    Args:
        df (pandas.DataFrame): The dataset
        numerical_features (list): List of numerical feature names
        target_column (str): Name of the target column
        figsize (tuple): Figure size (width, height)
    """
    print("\n=== Correlation Matrix of Numerical Features ===")

    # Calculate correlation matrix
    corr_matrix = df[numerical_features + [target_column]].corr()

    # Create the heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f',
                square=True, linewidths=.5, cbar_kws={"shrink": .8})

    plt.title('Correlation Matrix of Numerical Features', pad=20, fontsize=16)
    plt.tight_layout()
    plt.show()

    # Print strong correlations (absolute value > 0.5)
    print("\nStrong Correlations (|r| > 0.5):")
    strong_correlations = corr_matrix.unstack().sort_values(ascending=False)
    # Remove self-correlations
    strong_correlations = strong_correlations[strong_correlations != 1]
    strong_correlations = strong_correlations[abs(strong_correlations) > 0.5]
    print(strong_correlations)


def analyze_numerical_features(df, target_column='Target', plot=True):
    """
    Analyze numerical features and their relationship with the target variable.

    Args:
        df (pandas.DataFrame): The dataset to analyze
        target_column (str): Name of the target column
        plot (bool): Whether to create visualizations

    Returns:
        list: List of numerical feature column names
    """
    # Get numerical features
    numerical_features = get_numerical_features(df, target_column)

    if not numerical_features:
        print("\nNo numerical features found in the dataset.")
        return

    print(f"\n=== Numerical Features Analysis ===")
    print(f"Found {len(numerical_features)} numerical features:")
    print(numerical_features)

    # Calculate basic statistics
    print("\nBasic Statistics for Numerical Features:")
    stats = df[numerical_features].describe().T
    stats['count'] = stats['count'].astype(int)
    for col in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
        if col in stats.columns:
            stats[col] = stats[col].round(4)
    print(stats)

    # Check for correlation with target if it's numerical
    if target_column in df.columns and df[target_column].dtype in [np.int64, np.float64]:
        print("\nCorrelation with Target:")
        correlations = df[numerical_features +
                          [target_column]].corr()[target_column]
        correlations = correlations.drop(target_column)
        print(correlations.sort_values(ascending=False))

    # Create visualizations if requested
    if plot:
        print("\nCreating distribution plots...")
        plot_numerical_distributions(df, numerical_features, target_column)

        print("\nCreating individual box plots...")
        plot_numerical_boxplots(df, numerical_features,
                                target_column, horizontal=False)

        print("\nCreating horizontal box plot comparison...")
        plot_numerical_boxplots(df, numerical_features,
                                target_column, horizontal=True)

        # Define interesting feature pairs for scatter plots
        pairs = [
            ('pH', 'Iron'),
            ('Nitrate', 'Chloride'),
            ('Lead', 'Zinc'),
            ('Fluoride', 'Copper'),
            ('Total Dissolved Solids', 'Water Temperature'),
            ('Water Temperature', 'Air Temperature')
        ]

        # Filter pairs to only include features that exist in the dataset
        valid_pairs = [
            (x, y) for x, y in pairs if x in numerical_features and y in numerical_features]

        if valid_pairs:
            print("\nCreating scatter plots of feature pairs...")
            plot_feature_pairs(df, valid_pairs, target_column)

        # Create correlation matrix
        print("\nCreating correlation matrix...")
        plot_correlation_matrix(df, numerical_features, target_column)

    return numerical_features


def get_categorical_features(df):
    """
    Identify and return categorical features in the dataset.

    Args:
        df (pandas.DataFrame): The dataset to analyze

    Returns:
        list: List of categorical feature column names
    """
    # Get all categorical columns (object type)
    categorical_features = df.select_dtypes(include=[object]).columns.tolist()

    return categorical_features


def plot_categorical_countplots(df, categorical_features, target_column='Target', figsize=(10, 4)):
    """
    Create count plots for categorical features with target variable.

    Args:
        df (pandas.DataFrame): The dataset
        categorical_features (list): List of categorical feature names
        target_column (str): Name of the target column
        figsize (tuple): Figure size (width, height)
    """
    print("\n=== Count Plots of Categorical Features ===")

    for feature in categorical_features:
        plt.figure(figsize=figsize)
        sns.countplot(data=df, x=feature, hue=target_column)
        plt.title(f'Count Plot of {feature} with {target_column}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


def analyze_categorical_features(df, target_column='Target', plot=True):
    """
    Analyze categorical features and their relationship with the target variable.

    Args:
        df (pandas.DataFrame): The dataset to analyze
        target_column (str): Name of the target column
        plot (bool): Whether to create visualizations
    """
    # Get categorical features
    categorical_features = get_categorical_features(df)

    if not categorical_features:
        print("\nNo categorical features found in the dataset.")
        return

    print(f"\n=== Categorical Features Analysis ===")
    print(f"Found {len(categorical_features)} categorical features:")
    print(categorical_features)

    # Analyze each categorical feature
    for feature in categorical_features:
        print(f"\n--- Analysis of {feature} ---")

        # Get value counts and percentages
        value_counts = df[feature].value_counts()
        percentages = (value_counts / len(df) * 100).round(2)

        # Create summary DataFrame
        summary = pd.DataFrame({
            'Count': value_counts,
            'Percentage': percentages
        })

        print("\nValue Distribution:")
        print(summary)

        # If target is categorical, analyze relationship
        if target_column in df.columns:
            print("\nRelationship with Target:")
            # Create contingency table
            contingency = pd.crosstab(
                df[feature], df[target_column], normalize='columns') * 100
            print(contingency.round(2))

            # Calculate chi-square test if appropriate
            if len(value_counts) > 1 and len(df[target_column].unique()) > 1:
                from scipy.stats import chi2_contingency
                chi2, p, dof, expected = chi2_contingency(
                    pd.crosstab(df[feature], df[target_column]))
                print(f"\nChi-square test:")
                print(f"Chi2: {chi2:.4f}")
                print(f"P-value: {p:.4f}")
                print(f"Degrees of freedom: {dof}")

    # Create visualizations if requested
    if plot:
        plot_categorical_countplots(df, categorical_features, target_column)

    return categorical_features


def plot_target_distribution(counts, target_column='Target', class_labels=None, plot_type='bar'):
    """
    Create a visualization of the target variable distribution.

    Args:
        counts (pandas.Series): Value counts of the target variable
        target_column (str): Name of the target column
        class_labels (dict, optional): Mapping of class values to their labels
        plot_type (str): Type of plot to create ('bar' or 'pie')
    """
    plt.figure(figsize=(10, 6))

    # Prepare labels
    if class_labels:
        labels = [class_labels.get(x, str(x)) for x in counts.index]
    else:
        labels = [str(x) for x in counts.index]

    if plot_type == 'bar':
        # Create the bar plot
        ax = counts.plot(
            kind='bar', color=sns.color_palette("husl", len(counts)))

        # Customize the plot
        plt.title(f"{target_column} Class Distribution", pad=20, fontsize=14)
        plt.xticks(rotation=0)
        plt.xticks(range(len(counts)), labels)
        plt.xlabel(f"{target_column} Class", labelpad=10)
        plt.ylabel("Count", labelpad=10)

        # Add value labels on top of bars
        for i, v in enumerate(counts):
            ax.text(i, v, str(v), ha='center', va='bottom')

        # Add percentage labels
        percentages = (counts / counts.sum() * 100).round(1)
        for i, p in enumerate(percentages):
            ax.text(i, counts[i]/2, f'{p}%',
                    ha='center', va='center', color='white')

    elif plot_type == 'pie':
        # Create the pie chart
        plt.pie(counts, labels=labels, autopct='%1.1f%%',
                colors=sns.color_palette("husl", len(counts)),
                startangle=90, shadow=True)
        plt.title(f"{target_column} Class Distribution", pad=20, fontsize=14)
        # Equal aspect ratio ensures that pie is drawn as a circle
        plt.axis('equal')

    plt.tight_layout()
    plt.show()


def analyze_target_distribution(df, target_column='Target', class_labels=None, plot=True, plot_type='bar'):
    """
    Analyze and display the distribution of the target variable.

    Args:
        df (pandas.DataFrame): The dataset to analyze
        target_column (str): Name of the target column
        class_labels (dict, optional): Mapping of class values to their labels
        plot (bool): Whether to create a visualization
        plot_type (str): Type of plot to create ('bar' or 'pie')
    """
    if target_column not in df.columns:
        print(
            f"\nWarning: Target column '{target_column}' not found in dataset")
        return

    print(f"\n=== Target Variable Distribution ({target_column}) ===")

    # Get value counts and percentages
    counts = df[target_column].value_counts()
    percentages = (counts / counts.sum() * 100).round(2)

    # Create a summary DataFrame with formatted output
    distribution = pd.DataFrame({
        'Class': counts.index,
        'Count': counts.values,
        'Percentage': percentages.values
    })

    # Format the output for better readability
    print("\nClass Distribution:")
    print("=" * 50)
    print(f"{'Class':<15} {'Count':<15} {'Percentage':<15}")
    print("-" * 50)
    for _, row in distribution.iterrows():
        class_label = class_labels.get(row['Class'], str(
            row['Class'])) if class_labels else str(row['Class'])
        print(
            f"{class_label:<15} {row['Count']:<15} {row['Percentage']:<15.2f}%")
    print("=" * 50)

    # Print total
    print(f"\nTotal Samples: {counts.sum()}")

    # Print imbalance ratio for binary classification
    if len(counts) == 2:
        imbalance_ratio = counts.max() / counts.min()
        print(f"Class Imbalance Ratio: {imbalance_ratio:.2f}:1")

        # Print majority and minority class information
        majority_class = counts.idxmax()
        minority_class = counts.idxmin()
        majority_label = class_labels.get(majority_class, str(
            majority_class)) if class_labels else str(majority_class)
        minority_label = class_labels.get(minority_class, str(
            minority_class)) if class_labels else str(minority_class)
        print(
            f"\nMajority Class: {majority_label} ({percentages[majority_class]:.2f}%)")
        print(
            f"Minority Class: {minority_label} ({percentages[minority_class]:.2f}%)")

    # Create visualization if requested
    if plot:
        plot_target_distribution(
            counts, target_column, class_labels, plot_type)


def analyze_missing_patterns(df):
    """
    Analyze patterns of missing values in the dataset.

    Args:
        df (pandas.DataFrame): The dataset to analyze
    """
    print("\n=== Missing Values Pattern Analysis ===")

    # Calculate missing percentage per row
    missing_per_row = df.isna().sum(axis=1) / df.shape[1] * 100

    # Create summary of missing patterns
    pattern_summary = missing_per_row.value_counts().sort_index()

    # Print summary
    print("\nMissing Values Pattern Summary:")
    print("=" * 50)
    print(f"{'Missing %':<15} {'Row Count':<15} {'Percentage of Dataset':<20}")
    print("-" * 50)
    for missing_pct, count in pattern_summary.items():
        dataset_pct = (count / len(df) * 100).round(2)
        print(f"{missing_pct:.2f}%{'':<10} {count:<15} {dataset_pct}%{'':<15}")
    print("=" * 50)

    # Create visualization
    plt.figure(figsize=(12, 6))
    sns.histplot(data=missing_per_row, bins=20)
    plt.title('Distribution of Missing Values Percentage per Row')
    plt.xlabel('Percentage of Missing Values')
    plt.ylabel('Number of Rows')
    plt.axvline(x=missing_per_row.mean(), color='r', linestyle='--',
                label=f'Mean: {missing_per_row.mean():.2f}%')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print additional statistics
    print("\nAdditional Statistics:")
    print(f"Mean missing percentage per row: {missing_per_row.mean():.2f}%")
    print(
        f"Median missing percentage per row: {missing_per_row.median():.2f}%")
    print(f"Maximum missing percentage per row: {missing_per_row.max():.2f}%")
    print(
        f"Number of complete rows: {len(df) - missing_per_row[missing_per_row > 0].count()}")
    print(
        f"Number of rows with any missing values: {missing_per_row[missing_per_row > 0].count()}")


def visualize_missing_values(df, figsize=(12, 6)):
    """
    Visualize missing values in the dataset using a heatmap.

    Args:
        df (pandas.DataFrame): The dataset to analyze
        figsize (tuple): Figure size (width, height)
    """
    print("\n=== Missing Values Visualization ===")

    # Create heatmap of missing values
    plt.figure(figsize=figsize)
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title('Missing Values Heatmap', pad=20, fontsize=14)
    plt.xlabel('Features')
    plt.ylabel('Samples')
    plt.tight_layout()
    plt.show()

    # Calculate and display missing values count
    missing_values = df.isnull().sum()
    # Only show columns with missing values
    missing_values = missing_values[missing_values > 0]

    if len(missing_values) > 0:
        print("\nMissing Values Count:")
        print("=" * 50)
        print(f"{'Feature':<30} {'Missing Count':<15}")
        print("-" * 50)
        for feature, count in missing_values.items():
            print(f"{feature:<30} {count:<15}")
        print("=" * 50)
    else:
        print("\nNo missing values found in the dataset.")


def analyze_missing_values(df):
    """
    Analyze and display missing values in the dataset.

    Args:
        df (pandas.DataFrame): The dataset to analyze
    """
    print("\n=== Missing Values Analysis ===")

    # Calculate missing values
    missing_counts = df.isna().sum()
    missing_percentages = (missing_counts / len(df) * 100).round(2)

    # Create summary DataFrame
    missing_summary = pd.DataFrame({
        'Missing Count': missing_counts,
        'Missing Percentage': missing_percentages
    })

    # Sort by missing count in ascending order
    missing_summary = missing_summary.sort_values('Missing Count')

    # Print summary
    print("\nMissing Values Summary (Sorted by Count):")
    print("=" * 50)
    print(f"{'Column':<30} {'Missing Count':<15} {'Missing %':<15}")
    print("-" * 50)
    for idx, row in missing_summary.iterrows():
        if row['Missing Count'] > 0:  # Only show columns with missing values
            print(
                f"{idx:<30} {row['Missing Count']:<15} {row['Missing Percentage']:<15.2f}%")
    print("=" * 50)

    # Print total missing values
    total_missing = missing_counts.sum()
    total_percentage = (
        total_missing / (len(df) * len(df.columns)) * 100).round(2)
    print(f"\nTotal Missing Values: {total_missing}")
    print(f"Total Missing Percentage: {total_percentage}%")

    # Create visualization if there are missing values
    if total_missing > 0:
        # Bar plot of missing percentages
        plt.figure(figsize=(12, 6))
        sns.barplot(x=missing_percentages.index, y=missing_percentages.values)
        plt.xticks(rotation=45, ha='right')
        plt.title('Missing Values Percentage by Column')
        plt.ylabel('Missing Percentage (%)')
        plt.tight_layout()
        plt.show()

        # Heatmap of missing values
        visualize_missing_values(df)

    # Analyze missing patterns
    analyze_missing_patterns(df)

    # Print columns with no missing values
    complete_columns = missing_counts[missing_counts == 0].index.tolist()
    if complete_columns:
        print("\nColumns with No Missing Values:")
        print("=" * 50)
        for col in complete_columns:
            print(f"- {col}")
        print("=" * 50)


def drop_missing_rows(df, subset=None, how='any', thresh=None):
    """
    Drop rows with missing values from the dataset.

    Args:
        df (pandas.DataFrame): The dataset to process
        subset (list, optional): List of columns to consider when dropping rows
        how (str): How to determine if a row should be dropped ('any' or 'all')
        thresh (int, optional): Minimum number of non-NA values required to keep a row

    Returns:
        pandas.DataFrame: Dataset with rows containing missing values dropped
    """
    print("\n=== Dropping Rows with Missing Values ===")

    # Make a copy of the dataframe
    df_processed = df.copy()

    # Calculate initial missing values
    initial_missing = df_processed.isnull().sum().sum()
    initial_rows = len(df_processed)

    # Drop rows with missing values
    df_processed = df_processed.dropna(subset=subset, how=how, thresh=thresh)

    # Calculate final missing values
    final_missing = df_processed.isnull().sum().sum()
    final_rows = len(df_processed)
    rows_dropped = initial_rows - final_rows

    # Print summary
    print("\nSummary of Row Dropping:")
    print("=" * 50)
    print(f"Initial number of rows: {initial_rows}")
    print(f"Final number of rows: {final_rows}")
    print(f"Number of rows dropped: {rows_dropped}")
    print(
        f"Percentage of rows dropped: {(rows_dropped/initial_rows*100):.2f}%")
    print(f"Initial missing values: {initial_missing}")
    print(f"Final missing values: {final_missing}")
    print("=" * 50)

    # Print columns that still have missing values
    remaining_missing = df_processed.isnull().sum()
    remaining_missing = remaining_missing[remaining_missing > 0]
    if len(remaining_missing) > 0:
        print("\nColumns that still have missing values:")
        print("=" * 50)
        for col, count in remaining_missing.items():
            print(f"- {col}: {count} missing values")
        print("=" * 50)
    else:
        print("\nNo missing values remain in the dataset.")

    return df_processed


def impute_numerical_features(df, strategy='mean', target_column='Target'):
    """
    Impute missing values for numerical features using specified strategy.

    Args:
        df (pandas.DataFrame): The dataset to process
        strategy (str): Imputation strategy ('mean', 'median', 'mode')
        target_column (str): Name of the target column to exclude

    Returns:
        pandas.DataFrame: Dataset with imputed numerical features
    """
    print("\n=== Imputing Missing Values for Numerical Features ===")

    # Make a copy of the dataframe
    df_processed = df.copy()

    # Get numerical features
    numerical_features = get_numerical_features(df_processed, target_column)

    if not numerical_features:
        print("No numerical features found in the dataset.")
        return df_processed

    # Calculate initial missing values
    initial_missing = df_processed[numerical_features].isnull().sum()
    total_initial_missing = initial_missing.sum()

    if total_initial_missing == 0:
        print("No missing values found in numerical features.")
        return df_processed

    # Print initial missing values
    print("\nInitial Missing Values:")
    print("=" * 50)
    print(f"{'Feature':<30} {'Missing Count':<15}")
    print("-" * 50)
    for feature, count in initial_missing[initial_missing > 0].items():
        print(f"{feature:<30} {count:<15}")
    print("=" * 50)

    # Impute missing values
    for feature in numerical_features:
        if df_processed[feature].isnull().sum() > 0:
            if strategy == 'mean':
                value = df_processed[feature].mean()
                strategy_name = 'mean'
            elif strategy == 'median':
                value = df_processed[feature].median()
                strategy_name = 'median'
            else:  # mode
                value = df_processed[feature].mode()[0]
                strategy_name = 'mode'

            missing_count = df_processed[feature].isnull().sum()
            df_processed[feature] = df_processed[feature].fillna(value)
            print(
                f"\nFilled {missing_count} missing values in {feature} with {strategy_name}: {value:.4f}")

    # Verify imputation
    final_missing = df_processed[numerical_features].isnull().sum()
    total_final_missing = final_missing.sum()

    print("\nVerification of Imputation:")
    print("=" * 50)
    print(f"Total missing values before imputation: {total_initial_missing}")
    print(f"Total missing values after imputation: {total_final_missing}")

    if total_final_missing == 0:
        print("\nAll missing values have been successfully imputed.")
    else:
        print("\nWarning: Some missing values remain after imputation:")
        for feature, count in final_missing[final_missing > 0].items():
            print(f"- {feature}: {count} missing values")

    # Print summary statistics before and after imputation
    print("\nSummary Statistics Comparison:")
    print("=" * 50)
    print(f"{'Feature':<30} {'Before Mean':<15} {'After Mean':<15} {'Change':<15}")
    print("-" * 50)
    for feature in numerical_features:
        if initial_missing[feature] > 0:
            before_mean = df[feature].mean()
            after_mean = df_processed[feature].mean()
            change = ((after_mean - before_mean) / before_mean * 100).round(2)
            print(
                f"{feature:<30} {before_mean:<15.4f} {after_mean:<15.4f} {change:<15.2f}%")
    print("=" * 50)

    return df_processed


def impute_categorical_features(df, strategy='mode', target_column='Target'):
    """
    Impute missing values for categorical features using specified strategy.

    Args:
        df (pandas.DataFrame): The dataset to process
        strategy (str): Imputation strategy ('mode', 'unknown', 'most_frequent')
        target_column (str): Name of the target column to exclude

    Returns:
        pandas.DataFrame: Dataset with imputed categorical features
    """
    print("\n=== Imputing Missing Values for Categorical Features ===")

    # Make a copy of the dataframe
    df_processed = df.copy()

    # Get categorical features
    categorical_features = get_categorical_features(df_processed)

    if not categorical_features:
        print("No categorical features found in the dataset.")
        return df_processed

    # Calculate initial missing values
    initial_missing = df_processed[categorical_features].isnull().sum()
    total_initial_missing = initial_missing.sum()

    if total_initial_missing == 0:
        print("No missing values found in categorical features.")
        return df_processed

    # Print initial missing values
    print("\nInitial Missing Values:")
    print("=" * 50)
    print(f"{'Feature':<30} {'Missing Count':<15} {'Missing %':<15}")
    print("-" * 50)
    for feature, count in initial_missing[initial_missing > 0].items():
        missing_pct = (count / len(df_processed) * 100).round(2)
        print(f"{feature:<30} {count:<15} {missing_pct:<15.2f}%")
    print("=" * 50)

    # Impute missing values
    if strategy == 'unknown':
        # Directly fill all missing values with 'Unknown'
        df_processed[categorical_features] = df_processed[categorical_features].fillna(
            'Unknown')
        print("\nFilled all missing values in categorical features with 'Unknown'")
    else:
        # Impute feature by feature
        for feature in categorical_features:
            if df_processed[feature].isnull().sum() > 0:
                if strategy == 'mode' or strategy == 'most_frequent':
                    value = df_processed[feature].mode()[0]
                    strategy_name = 'mode'
                else:
                    value = 'Unknown'
                    strategy_name = 'Unknown'

                missing_count = df_processed[feature].isnull().sum()
                df_processed[feature] = df_processed[feature].fillna(value)
                print(
                    f"\nFilled {missing_count} missing values in {feature} with {strategy_name}: {value}")

    # Verify imputation
    final_missing = df_processed[categorical_features].isnull().sum()
    total_final_missing = final_missing.sum()

    print("\nVerification of Imputation:")
    print("=" * 50)
    print(f"Total missing values before imputation: {total_initial_missing}")
    print(f"Total missing values after imputation: {total_final_missing}")

    if total_final_missing == 0:
        print("\nAll missing values have been successfully imputed.")
    else:
        print("\nWarning: Some missing values remain after imputation:")
        for feature, count in final_missing[final_missing > 0].items():
            print(f"- {feature}: {count} missing values")

    # Print value distribution before and after imputation
    print("\nValue Distribution Comparison:")
    print("=" * 50)
    for feature in categorical_features:
        if initial_missing[feature] > 0:
            print(f"\n--- {feature} ---")
            print("\nBefore Imputation:")
            before_counts = df[feature].value_counts()
            before_pct = (before_counts / len(df) * 100).round(2)
            print(pd.DataFrame({
                'Count': before_counts,
                'Percentage': before_pct
            }))

            print("\nAfter Imputation:")
            after_counts = df_processed[feature].value_counts()
            after_pct = (after_counts / len(df_processed) * 100).round(2)
            print(pd.DataFrame({
                'Count': after_counts,
                'Percentage': after_pct
            }))
    print("=" * 50)

    return df_processed


def handle_missing_values(df, threshold=0.3, numerical_strategy='mean', categorical_strategy='mode', drop_rows=False):
    """
    Handle missing values in the dataset using different strategies.

    Args:
        df (pandas.DataFrame): The dataset to process
        threshold (float): Threshold for dropping columns (if missing percentage > threshold)
        numerical_strategy (str): Strategy for numerical features ('mean', 'median', 'mode')
        categorical_strategy (str): Strategy for categorical features ('mode', 'unknown')
        drop_rows (bool): Whether to drop rows with missing values instead of imputing

    Returns:
        pandas.DataFrame: Processed dataset with handled missing values
    """
    print("\n=== Handling Missing Values ===")

    # Make a copy of the dataframe
    df_processed = df.copy()

    # Calculate missing percentages
    missing_percentages = df_processed.isnull().sum() / len(df_processed) * 100

    # Identify columns to drop
    columns_to_drop = missing_percentages[missing_percentages >
                                          threshold * 100].index.tolist()

    if columns_to_drop:
        print(f"\nDropping columns with >{threshold*100}% missing values:")
        for col in columns_to_drop:
            print(f"- {col} ({missing_percentages[col]:.2f}% missing)")
        df_processed = df_processed.drop(columns=columns_to_drop)

    if drop_rows:
        # Drop rows with missing values
        df_processed = drop_missing_rows(df_processed)
    else:
        # Impute numerical features
        df_processed = impute_numerical_features(
            df_processed, strategy=numerical_strategy)

        # Impute categorical features
        df_processed = impute_categorical_features(
            df_processed, strategy=categorical_strategy)

    # Print summary of changes
    print("\n=== Missing Values Summary After Processing ===")
    remaining_missing = df_processed.isnull().sum().sum()
    if remaining_missing == 0:
        print("All missing values have been handled.")
    else:
        print(
            f"Warning: {remaining_missing} missing values remain in the dataset.")
        print("\nRemaining missing values by column:")
        remaining_missing_counts = df_processed.isnull().sum()
        remaining_missing_counts = remaining_missing_counts[remaining_missing_counts > 0]
        for feature, count in remaining_missing_counts.items():
            print(f"- {feature}: {count} missing values")

    return df_processed


def display_categorical_missing_counts(df, target_column='Target'):
    """
    Display missing value counts for categorical features.

    Args:
        df (pandas.DataFrame): The dataset to analyze
        target_column (str): Name of the target column to exclude
    """
    print("\n=== Missing Values in Categorical Features ===")

    # Get categorical features
    categorical_features = get_categorical_features(df)

    if not categorical_features:
        print("No categorical features found in the dataset.")
        return

    # Calculate missing values
    missing_counts = df[categorical_features].isnull().sum()
    missing_percentages = (missing_counts / len(df) * 100).round(2)

    # Create summary DataFrame
    missing_summary = pd.DataFrame({
        'Missing Count': missing_counts,
        'Missing Percentage': missing_percentages
    })

    # Sort by missing count in descending order
    missing_summary = missing_summary.sort_values(
        'Missing Count', ascending=False)

    # Print summary
    print("\nMissing Values Summary:")
    print("=" * 50)
    print(f"{'Feature':<30} {'Missing Count':<15} {'Missing %':<15}")
    print("-" * 50)
    for idx, row in missing_summary.iterrows():
        if row['Missing Count'] > 0:  # Only show columns with missing values
            print(
                f"{idx:<30} {row['Missing Count']:<15} {row['Missing Percentage']:<15.2f}%")
    print("=" * 50)

    # Print total missing values
    total_missing = missing_counts.sum()
    total_percentage = (
        total_missing / (len(df) * len(categorical_features)) * 100).round(2)
    print(f"\nTotal Missing Values: {total_missing}")
    print(f"Total Missing Percentage: {total_percentage}%")

    # Print columns with no missing values
    complete_columns = missing_counts[missing_counts == 0].index.tolist()
    if complete_columns:
        print("\nColumns with No Missing Values:")
        print("=" * 50)
        for col in complete_columns:
            print(f"- {col}")
        print("=" * 50)

    return missing_summary


def display_missing_counts(df, target_column='Target'):
    """
    Display missing value counts for all features in the dataset.

    Args:
        df (pandas.DataFrame): The dataset to analyze
        target_column (str): Name of the target column to exclude
    """
    print("\n=== Missing Values in All Features ===")

    # Calculate missing values
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df) * 100).round(2)

    # Create summary DataFrame
    missing_summary = pd.DataFrame({
        'Missing Count': missing_counts,
        'Missing Percentage': missing_percentages
    })

    # Sort by missing count in descending order
    missing_summary = missing_summary.sort_values(
        'Missing Count', ascending=False)

    # Print summary
    print("\nMissing Values Summary:")
    print("=" * 50)
    print(f"{'Feature':<30} {'Missing Count':<15} {'Missing %':<15}")
    print("-" * 50)
    for idx, row in missing_summary.iterrows():
        if row['Missing Count'] > 0:  # Only show columns with missing values
            print(
                f"{idx:<30} {row['Missing Count']:<15} {row['Missing Percentage']:<15.2f}%")
    print("=" * 50)

    # Print total missing values
    total_missing = missing_counts.sum()
    total_percentage = (
        total_missing / (len(df) * len(df.columns)) * 100).round(2)
    print(f"\nTotal Missing Values: {total_missing}")
    print(f"Total Missing Percentage: {total_percentage}%")

    # Print columns with no missing values
    complete_columns = missing_counts[missing_counts == 0].index.tolist()
    if complete_columns:
        print("\nColumns with No Missing Values:")
        print("=" * 50)
        for col in complete_columns:
            print(f"- {col}")
        print("=" * 50)

    # Print data types of columns with missing values
    print("\nData Types of Columns with Missing Values:")
    print("=" * 50)
    for idx, row in missing_summary.iterrows():
        if row['Missing Count'] > 0:
            print(f"{idx:<30} {df[idx].dtype}")
    print("=" * 50)

    return missing_summary


def encode_categorical_features(df, target_column='Target'):
    """
    Encode categorical features using appropriate encoding methods.

    Args:
        df (pandas.DataFrame): The dataset to process
        target_column (str): Name of the target column to exclude

    Returns:
        pandas.DataFrame: Dataset with encoded categorical features
    """
    print("\n=== Encoding Categorical Features ===")

    # Make a copy of the dataframe
    df_encoded = df.copy()

    # Get categorical features
    categorical_features = get_categorical_features(df_encoded)

    if not categorical_features:
        print("No categorical features found in the dataset.")
        return df_encoded

    print(f"\nFound {len(categorical_features)} categorical features:")
    print(categorical_features)

    # Dictionary to store encoding mappings
    encoding_mappings = {}

    # Process each categorical feature
    for feature in categorical_features:
        print(f"\n--- Processing {feature} ---")

        # Check if feature is ordinal (has inherent order)
        is_ordinal = False
        # Add your logic here to determine if feature is ordinal
        # For example, if feature name contains 'level', 'grade', etc.
        if any(term in feature.lower() for term in ['level', 'grade', 'rank', 'score']):
            is_ordinal = True

        if is_ordinal:
            # Use label encoding for ordinal features
            print(f"Using label encoding for ordinal feature: {feature}")
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df_encoded[feature] = le.fit_transform(
                df_encoded[feature].astype(str))
            encoding_mappings[feature] = {
                'type': 'label',
                'mapping': dict(zip(le.classes_, le.transform(le.classes_)))
            }
        else:
            # Use one-hot encoding for nominal features
            print(f"Using one-hot encoding for nominal feature: {feature}")
            # Get dummy variables
            dummies = pd.get_dummies(
                df_encoded[feature], prefix=feature, drop_first=True)
            # Drop original column
            df_encoded = df_encoded.drop(columns=[feature])
            # Add dummy variables
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            encoding_mappings[feature] = {
                'type': 'one-hot',
                'categories': dummies.columns.tolist()
            }

    # Print encoding summary
    print("\n=== Encoding Summary ===")
    print("=" * 50)
    for feature, mapping in encoding_mappings.items():
        print(f"\nFeature: {feature}")
        print(f"Encoding Type: {mapping['type']}")
        if mapping['type'] == 'label':
            print("Value Mappings:")
            for value, code in mapping['mapping'].items():
                print(f"  {value} -> {code}")
        else:  # one-hot
            print(f"Created {len(mapping['categories'])} dummy variables:")
            for category in mapping['categories']:
                print(f"  {category}")
    print("=" * 50)

    # Print new feature list
    print("\nNew Features After Encoding:")
    print("=" * 50)
    new_features = [col for col in df_encoded.columns if col not in df.columns]
    print(f"Added {len(new_features)} new features:")
    for feature in new_features:
        print(f"- {feature}")
    print("=" * 50)

    return df_encoded, encoding_mappings


def identify_features(df, target_column='Target', non_feature_cols=None):
    """
    Identify numerical and categorical features in the dataset.

    Args:
        df (pandas.DataFrame): The dataset to analyze
        target_column (str): Name of the target column
        non_feature_cols (list): List of columns to exclude from features

    Returns:
        tuple: (numerical_features, categorical_features)
    """
    print("\n=== Feature Identification ===")

    # Default non-feature columns
    if non_feature_cols is None:
        non_feature_cols = ['Target', 'Month', 'Day', 'Time of Day']

    # Identify numerical features
    numerical_features = df.select_dtypes(
        include=['int64', 'float64']).columns.tolist()
    numerical_features = [
        col for col in numerical_features if col not in non_feature_cols]

    # Identify categorical features
    categorical_features = df.select_dtypes(
        include=['object', 'category']).columns.tolist()
    categorical_features = [
        col for col in categorical_features if col not in non_feature_cols]

    # Print feature summary
    print("\nFeature Summary:")
    print("=" * 50)
    print(
        f"Total Features: {len(numerical_features) + len(categorical_features)}")
    print(f"Numerical Features: {len(numerical_features)}")
    print(f"Categorical Features: {len(categorical_features)}")
    print("\nNumerical Features:")
    for feature in numerical_features:
        print(f"- {feature} ({df[feature].dtype})")
    print("\nCategorical Features:")
    for feature in categorical_features:
        print(f"- {feature} ({df[feature].dtype})")
    print("=" * 50)

    return numerical_features, categorical_features


def display_initial_dataframe(df):
    """
    Display the initial dataframe with basic information.

    Args:
        df (pandas.DataFrame): The dataset to display
    """
    print("\n=== Initial DataFrame ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    print(f"Rows: {len(df)}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nLast 5 rows:")
    print(df.tail())
    print("\nData Types:")
    print(df.dtypes)
    print("\nMemory Usage:")
    print(f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


def plot_numerical_histograms(df, numerical_features, figsize=(15, 12), bins=15, title='Histograms of Numerical Features'):
    """
    Create histograms for numerical features.

    Args:
        df (pandas.DataFrame): The dataset
        numerical_features (list): List of numerical feature names
        figsize (tuple): Figure size (width, height)
        bins (int): Number of bins for histograms
        title (str): Title for the plot
    """
    print("\n=== Numerical Features Distribution ===")

    if not numerical_features:
        print("No numerical features found in the dataset.")
        return

    # Calculate number of rows and columns for subplots
    n_features = len(numerical_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    # Create figure and subplots
    plt.figure(figsize=figsize)

    # Plot histograms
    for i, feature in enumerate(numerical_features, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(data=df, x=feature, bins=bins, kde=True)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Count')

    # Add main title
    plt.suptitle(title, fontsize=16, y=1.02)

    # Adjust layout
    plt.tight_layout()
    plt.show()

    # Print basic statistics
    print("\nBasic Statistics:")
    print("=" * 50)
    stats = df[numerical_features].describe().T
    stats['count'] = stats['count'].astype(int)
    for col in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
        if col in stats.columns:
            stats[col] = stats[col].round(4)
    print(stats)
    print("=" * 50)


def create_numerical_transformer(imputation_strategy='mean', scaling_method='minmax'):
    """
    Create a pipeline for numerical feature transformation.

    Args:
        imputation_strategy (str): Strategy for imputing missing values ('mean', 'median', 'mode')
        scaling_method (str): Method for scaling features ('minmax', 'standard', 'robust')

    Returns:
        sklearn.pipeline.Pipeline: Pipeline for numerical feature transformation
    """
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

    # Create imputer
    imputer = SimpleImputer(strategy=imputation_strategy)

    # Create scaler based on method
    if scaling_method == 'minmax':
        scaler = MinMaxScaler()
    elif scaling_method == 'standard':
        scaler = StandardScaler()
    else:  # robust
        scaler = RobustScaler()

    # Create pipeline
    numerical_transformer = Pipeline(steps=[
        ('imputer', imputer),
        ('scaler', scaler)
    ])

    return numerical_transformer


def transform_numerical_features(df, numerical_features, transformer=None, imputation_strategy='mean', scaling_method='minmax'):
    """
    Transform numerical features using the specified transformer.

    Args:
        df (pandas.DataFrame): The dataset
        numerical_features (list): List of numerical feature names
        transformer (sklearn.pipeline.Pipeline, optional): Pre-created transformer
        imputation_strategy (str): Strategy for imputing missing values
        scaling_method (str): Method for scaling features

    Returns:
        tuple: (transformed_df, transformer)
    """
    print("\n=== Transforming Numerical Features ===")

    if not numerical_features:
        print("No numerical features found in the dataset.")
        return df, None

    # Make a copy of the dataframe
    df_transformed = df.copy()

    # Create transformer if not provided
    if transformer is None:
        transformer = create_numerical_transformer(
            imputation_strategy, scaling_method)

    # Print transformation details
    print(f"\nTransformation Details:")
    print(f"- Imputation Strategy: {imputation_strategy}")
    print(f"- Scaling Method: {scaling_method}")
    print(f"\nFeatures to Transform:")
    for feature in numerical_features:
        print(f"- {feature}")

    # Transform features
    print("\nTransforming features...")
    df_transformed[numerical_features] = transformer.fit_transform(
        df_transformed[numerical_features])

    # Print summary statistics before and after transformation
    print("\nSummary Statistics Comparison:")
    print("=" * 50)
    print(f"{'Feature':<30} {'Before Mean':<15} {'After Mean':<15} {'Before Std':<15} {'After Std':<15}")
    print("-" * 50)
    for feature in numerical_features:
        before_mean = df[feature].mean()
        after_mean = df_transformed[feature].mean()
        before_std = df[feature].std()
        after_std = df_transformed[feature].std()
        print(f"{feature:<30} {before_mean:<15.4f} {after_mean:<15.4f} {before_std:<15.4f} {after_std:<15.4f}")
    print("=" * 50)

    return df_transformed, transformer


def create_categorical_transformer(imputation_strategy='most_frequent', handle_unknown='ignore'):
    """
    Create a pipeline for categorical feature transformation.

    Args:
        imputation_strategy (str): Strategy for imputing missing values ('most_frequent', 'constant')
        handle_unknown (str): How to handle unknown categories ('ignore', 'error')

    Returns:
        sklearn.pipeline.Pipeline: Pipeline for categorical feature transformation
    """
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder

    # Create imputer
    imputer = SimpleImputer(strategy=imputation_strategy)

    # Create encoder
    encoder = OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False)

    # Create pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', imputer),
        ('onehot', encoder)
    ])

    return categorical_transformer


def transform_categorical_features(df, categorical_features, transformer=None, imputation_strategy='most_frequent', handle_unknown='ignore'):
    """
    Transform categorical features using the specified transformer.

    Args:
        df (pandas.DataFrame): The dataset
        categorical_features (list): List of categorical feature names
        transformer (sklearn.pipeline.Pipeline, optional): Pre-created transformer
        imputation_strategy (str): Strategy for imputing missing values
        handle_unknown (str): How to handle unknown categories

    Returns:
        tuple: (transformed_df, transformer, feature_names)
    """
    print("\n=== Transforming Categorical Features ===")

    if not categorical_features:
        print("No categorical features found in the dataset.")
        return df, None, []

    # Make a copy of the dataframe
    df_transformed = df.copy()

    # Create transformer if not provided
    if transformer is None:
        transformer = create_categorical_transformer(
            imputation_strategy, handle_unknown)

    # Print transformation details
    print(f"\nTransformation Details:")
    print(f"- Imputation Strategy: {imputation_strategy}")
    print(f"- Handle Unknown: {handle_unknown}")
    print(f"\nFeatures to Transform:")
    for feature in categorical_features:
        print(f"- {feature}")

    # Transform features
    print("\nTransforming features...")
    transformed_data = transformer.fit_transform(
        df_transformed[categorical_features])

    # Get feature names after one-hot encoding
    feature_names = []
    for i, feature in enumerate(categorical_features):
        categories = transformer.named_steps['onehot'].categories_[i]
        for category in categories:
            feature_names.append(f"{feature}_{category}")

    # Create new dataframe with transformed features
    df_transformed = df_transformed.drop(columns=categorical_features)
    df_transformed = pd.concat([
        df_transformed,
        pd.DataFrame(transformed_data, columns=feature_names, index=df.index)
    ], axis=1)

    # Print summary of transformation
    print("\nTransformation Summary:")
    print("=" * 50)
    print(f"Original Features: {len(categorical_features)}")
    print(f"New Features: {len(feature_names)}")
    print("\nNew Feature Names:")
    for name in feature_names:
        print(f"- {name}")
    print("=" * 50)

    return df_transformed, transformer, feature_names


def create_preprocessor(numerical_features, categorical_features,
                        numerical_imputation_strategy='mean', numerical_scaling_method='minmax',
                        categorical_imputation_strategy='most_frequent', categorical_handle_unknown='ignore'):
    """
    Create a ColumnTransformer to preprocess both numerical and categorical features.

    Args:
        numerical_features (list): List of numerical feature names
        categorical_features (list): List of categorical feature names
        numerical_imputation_strategy (str): Strategy for imputing missing values in numerical features
        numerical_scaling_method (str): Method for scaling numerical features
        categorical_imputation_strategy (str): Strategy for imputing missing values in categorical features
        categorical_handle_unknown (str): How to handle unknown categories

    Returns:
        sklearn.compose.ColumnTransformer: Combined preprocessor for all features
    """
    from sklearn.compose import ColumnTransformer

    # Create numerical transformer
    numerical_transformer = create_numerical_transformer(
        imputation_strategy=numerical_imputation_strategy,
        scaling_method=numerical_scaling_method
    )

    # Create categorical transformer
    categorical_transformer = create_categorical_transformer(
        imputation_strategy=categorical_imputation_strategy,
        handle_unknown=categorical_handle_unknown
    )

    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'  # Keep other columns unchanged
    )

    return preprocessor


def display_preprocessed_data(df_transformed, all_features, non_feature_cols=None, n_rows=5):
    """
    Display the preprocessed DataFrame in a clear and informative way.

    Args:
        df_transformed (pandas.DataFrame): The preprocessed DataFrame
        all_features (list): List of all feature names after transformation
        non_feature_cols (list): List of non-feature columns
        n_rows (int): Number of rows to display
    """
    print("\n=== Preprocessed DataFrame ===")
    print("=" * 50)

    # Set display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.4f}'.format)

    # Display shape information
    print(f"\nShape: {df_transformed.shape}")
    print(f"Number of Samples: {df_transformed.shape[0]}")
    print(f"Number of Features: {df_transformed.shape[1]}")

    # Display column information
    print("\nColumns:")
    print("-" * 50)
    print("Transformed Features:")
    for feature in all_features:
        print(f"- {feature}")

    if non_feature_cols:
        print("\nNon-Feature Columns:")
        for col in non_feature_cols:
            if col in df_transformed.columns:
                print(f"- {col}")

    # Display data preview
    print("\nData Preview:")
    print("-" * 50)
    print(df_transformed.head(n_rows))

    # Display data types
    print("\nData Types:")
    print("-" * 50)
    print(df_transformed.dtypes)

    # Display basic statistics
    print("\nBasic Statistics:")
    print("-" * 50)
    stats = df_transformed.describe().T
    stats['count'] = stats['count'].astype(int)
    print(stats)

    print("=" * 50)


def preprocess_features(df, numerical_features, categorical_features, non_feature_cols=None, preprocessor=None,
                        numerical_imputation_strategy='mean', numerical_scaling_method='minmax',
                        categorical_imputation_strategy='most_frequent', categorical_handle_unknown='ignore'):
    """
    Preprocess both numerical and categorical features using a ColumnTransformer.

    Args:
        df (pandas.DataFrame): The dataset
        numerical_features (list): List of numerical feature names
        categorical_features (list): List of categorical feature names
        non_feature_cols (list): List of columns to keep but not transform
        preprocessor (sklearn.compose.ColumnTransformer, optional): Pre-created preprocessor
        numerical_imputation_strategy (str): Strategy for imputing missing values in numerical features
        numerical_scaling_method (str): Method for scaling numerical features
        categorical_imputation_strategy (str): Strategy for imputing missing values in categorical features
        categorical_handle_unknown (str): How to handle unknown categories

    Returns:
        tuple: (transformed_df, preprocessor, all_features)
    """
    print("\n=== Preprocessing All Features ===")

    # Set default non-feature columns if not provided
    if non_feature_cols is None:
        non_feature_cols = ['Target', 'Month', 'Day', 'Time of Day']

    # Print initial shape and types
    print(f"\nInitial DataFrame Shape: {df.shape}")
    print(f"Initial Features: {len(df.columns)}")
    print("\nInitial Data Types:")
    print(df.dtypes)

    # Create preprocessor if not provided
    if preprocessor is None:
        preprocessor = create_preprocessor(
            numerical_features, categorical_features,
            numerical_imputation_strategy, numerical_scaling_method,
            categorical_imputation_strategy, categorical_handle_unknown
        )

    # Print preprocessing details
    print(f"\nPreprocessing Details:")
    print(f"Numerical Features ({len(numerical_features)}):")
    for feature in numerical_features:
        print(f"- {feature}")
    print(f"\nCategorical Features ({len(categorical_features)}):")
    for feature in categorical_features:
        print(f"- {feature}")
    print(f"\nNon-Feature Columns ({len(non_feature_cols)}):")
    for feature in non_feature_cols:
        print(f"- {feature}")
    print(f"\nNumerical Imputation: {numerical_imputation_strategy}")
    print(f"Numerical Scaling: {numerical_scaling_method}")
    print(f"Categorical Imputation: {categorical_imputation_strategy}")
    print(f"Handle Unknown Categories: {categorical_handle_unknown}")

    # Transform features
    print("\nTransforming features...")
    transformed_data = preprocessor.fit_transform(df)

    # Print shape and type information
    print(f"\nShape of Transformed Data: {transformed_data.shape}")
    print(f"Number of Samples: {transformed_data.shape[0]}")
    print(
        f"Number of Features After Transformation: {transformed_data.shape[1]}")
    print(f"\nType of Transformed Data: {type(transformed_data)}")
    print(f"Data Type of Transformed Array: {transformed_data.dtype}")

    # Display transformed data
    print("\nTransformed Data Preview:")
    print("=" * 50)
    # Convert to DataFrame for better display
    preview_df = pd.DataFrame(transformed_data[:5])  # Show first 5 rows
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', None)  # Auto-detect display width
    # Format float values
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(preview_df)
    print("=" * 50)

    # Get feature names after transformation
    print("\n=== Feature Names After Transformation ===")

    # Get numerical feature names
    num_features = numerical_features
    print(f"\nNumerical Features ({len(num_features)}):")
    for feature in num_features:
        print(f"- {feature}")

    # Get categorical feature names after one-hot encoding
    try:
        cat_features = preprocessor.named_transformers_[
            'cat']['onehot'].get_feature_names_out(categorical_features)
        print(
            f"\nCategorical Features (One-Hot Encoded, {len(cat_features)}):")
        for feature in cat_features:
            print(f"- {feature}")
    except AttributeError:
        # Fallback for older scikit-learn versions
        cat_features = []
        for i, feature in enumerate(categorical_features):
            categories = preprocessor.named_transformers_[
                'cat'].named_steps['onehot'].categories_[i]
            for category in categories:
                cat_features.append(f"{feature}_{category}")
        print(
            f"\nCategorical Features (One-Hot Encoded, {len(cat_features)}):")
        for feature in cat_features:
            print(f"- {feature}")

    # Combine all feature names
    all_features = num_features + list(cat_features)
    print(f"\nTotal Features After Transformation: {len(all_features)}")

    # Create new dataframe with transformed features
    df_transformed = pd.DataFrame(
        transformed_data, columns=all_features, index=df.index)

    # Add back non-transformed columns
    if non_feature_cols:
        print("\nAdding back non-transformed columns...")
        for col in non_feature_cols:
            if col in df.columns:
                df_transformed[col] = df[col].values
                print(f"- Added {col}")

    # Print summary of preprocessing
    print("\nPreprocessing Summary:")
    print("=" * 50)
    print(
        f"Original Features: {len(numerical_features) + len(categorical_features)}")
    print(f"New Features: {len(all_features)}")
    print(f"Non-Feature Columns: {len(non_feature_cols)}")
    print("\nFeature Names:")
    for name in all_features:
        print(f"- {name}")
    print("\nNon-Feature Columns:")
    for name in non_feature_cols:
        print(f"- {name}")
    print("=" * 50)

    # Print final shape and type information
    print("\nFinal DataFrame Information:")
    print(f"Shape: {df_transformed.shape}")
    print(f"Type: {type(df_transformed)}")
    print("\nFinal Data Types:")
    print(df_transformed.dtypes)
    print(
        f"\nMemory Usage: {df_transformed.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Display final transformed DataFrame
    display_preprocessed_data(df_transformed, all_features, non_feature_cols)

    return df_transformed, preprocessor, all_features


def analyze_feature_statistics(df, numerical_features=None, categorical_features=None, target_column='Target'):
    """
    Perform comprehensive statistical analysis of features.

    Args:
        df (pandas.DataFrame): The dataset
        numerical_features (list): List of numerical feature names
        categorical_features (list): List of categorical feature names
        target_column (str): Name of the target column
    """
    print("\n=== Feature Statistics Analysis ===")
    print("=" * 50)

    # Identify features if not provided
    if numerical_features is None or categorical_features is None:
        numerical_features, categorical_features = identify_features(
            df, target_column)

    # Analyze numerical features
    if numerical_features:
        print("\nNumerical Features Statistics:")
        print("-" * 50)

        # Calculate basic statistics
        stats = df[numerical_features].describe().T
        stats['count'] = stats['count'].astype(int)

        # Add additional statistics
        stats['skewness'] = df[numerical_features].skew()
        stats['kurtosis'] = df[numerical_features].kurtosis()
        stats['missing_values'] = df[numerical_features].isnull().sum()
        stats['missing_pct'] = (
            stats['missing_values'] / len(df) * 100).round(2)
        stats['unique_values'] = df[numerical_features].nunique()

        # Format the statistics
        for col in ['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skewness', 'kurtosis']:
            if col in stats.columns:
                stats[col] = stats[col].round(4)

        print(stats)

        # Analyze correlation with target if it's numerical
        if target_column in df.columns and df[target_column].dtype in [np.int64, np.float64]:
            print("\nCorrelation with Target:")
            correlations = df[numerical_features +
                              [target_column]].corr()[target_column]
            correlations = correlations.drop(target_column)
            print(correlations.sort_values(ascending=False))

    # Analyze categorical features
    if categorical_features:
        print("\nCategorical Features Statistics:")
        print("-" * 50)

        for feature in categorical_features:
            print(f"\n--- {feature} ---")

            # Calculate value counts and percentages
            value_counts = df[feature].value_counts()
            percentages = (value_counts / len(df) * 100).round(2)

            # Create summary DataFrame
            summary = pd.DataFrame({
                'Count': value_counts,
                'Percentage': percentages
            })

            print("\nValue Distribution:")
            print(summary)

            # If target is categorical, analyze relationship
            if target_column in df.columns:
                print("\nRelationship with Target:")
                # Create contingency table
                contingency = pd.crosstab(
                    df[feature], df[target_column], normalize='columns') * 100
                print(contingency.round(2))

                # Calculate chi-square test if appropriate
                if len(value_counts) > 1 and len(df[target_column].unique()) > 1:
                    from scipy.stats import chi2_contingency
                    chi2, p, dof, expected = chi2_contingency(
                        pd.crosstab(df[feature], df[target_column]))
                    print(f"\nChi-square test:")
                    print(f"Chi2: {chi2:.4f}")
                    print(f"P-value: {p:.4f}")
                    print(f"Degrees of freedom: {dof}")

    # Analyze target variable
    if target_column in df.columns:
        print("\nTarget Variable Analysis:")
        print("-" * 50)

        # Get value counts and percentages
        counts = df[target_column].value_counts()
        percentages = (counts / counts.sum() * 100).round(2)

        # Create summary DataFrame
        target_summary = pd.DataFrame({
            'Count': counts,
            'Percentage': percentages
        })

        print("\nClass Distribution:")
        print(target_summary)

        # Print imbalance ratio for binary classification
        if len(counts) == 2:
            imbalance_ratio = counts.max() / counts.min()
            print(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}:1")

            # Print majority and minority class information
            majority_class = counts.idxmax()
            minority_class = counts.idxmin()
            print(
                f"\nMajority Class: {majority_class} ({percentages[majority_class]:.2f}%)")
            print(
                f"Minority Class: {minority_class} ({percentages[minority_class]:.2f}%)")

    print("=" * 50)


def display_dataframe_statistics(df, numerical_features=None, categorical_features=None, target_column='Target'):
    """
    Display comprehensive statistics of the DataFrame using describe() with additional formatting.

    Args:
        df (pandas.DataFrame): The dataset
        numerical_features (list): List of numerical feature names
        categorical_features (list): List of categorical feature names
        target_column (str): Name of the target column
    """
    print("\n=== DataFrame Statistics ===")
    print("=" * 50)

    # Set display options for better formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.4f}'.format)

    # Identify features if not provided
    if numerical_features is None or categorical_features is None:
        numerical_features, categorical_features = identify_features(
            df, target_column)

    # Display basic DataFrame information
    print(f"\nShape: {df.shape}")
    print(f"Number of Samples: {df.shape[0]}")
    print(f"Number of Features: {df.shape[1]}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Display numerical features statistics
    if numerical_features:
        print("\nNumerical Features Statistics:")
        print("-" * 50)

        # Get basic statistics
        stats = df[numerical_features].describe().T

        # Add additional statistics
        stats['skewness'] = df[numerical_features].skew()
        stats['kurtosis'] = df[numerical_features].kurtosis()
        stats['missing_values'] = df[numerical_features].isnull().sum()
        stats['missing_pct'] = (
            stats['missing_values'] / len(df) * 100).round(2)
        stats['unique_values'] = df[numerical_features].nunique()

        # Format the statistics
        stats['count'] = stats['count'].astype(int)
        for col in ['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skewness', 'kurtosis']:
            if col in stats.columns:
                stats[col] = stats[col].round(4)

        print(stats)

    # Display categorical features statistics
    if categorical_features:
        print("\nCategorical Features Statistics:")
        print("-" * 50)

        for feature in categorical_features:
            print(f"\n--- {feature} ---")

            # Calculate value counts and percentages
            value_counts = df[feature].value_counts()
            percentages = (value_counts / len(df) * 100).round(2)

            # Create summary DataFrame
            summary = pd.DataFrame({
                'Count': value_counts,
                'Percentage': percentages
            })

            print("\nValue Distribution:")
            print(summary)

    # Display target variable statistics if present
    if target_column in df.columns:
        print("\nTarget Variable Statistics:")
        print("-" * 50)

        # Get value counts and percentages
        counts = df[target_column].value_counts()
        percentages = (counts / counts.sum() * 100).round(2)

        # Create summary DataFrame
        target_summary = pd.DataFrame({
            'Count': counts,
            'Percentage': percentages
        })

        print("\nClass Distribution:")
        print(target_summary)

        # Print imbalance ratio for binary classification
        if len(counts) == 2:
            imbalance_ratio = counts.max() / counts.min()
            print(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}:1")

    print("=" * 50)

    # Reset display options
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')


def visualize_distributions(df, numerical_features=None, categorical_features=None, target_column='Target', figsize=(20, 15)):
    """
    Visualize distributions of features using various plot types.

    Args:
        df (pandas.DataFrame): The dataset
        numerical_features (list): List of numerical feature names
        categorical_features (list): List of categorical feature names
        target_column (str): Name of the target column
        figsize (tuple): Figure size (width, height)
    """
    print("\n=== Visualizing Data Distributions ===")
    print("=" * 50)

    # Identify features if not provided
    if numerical_features is None or categorical_features is None:
        numerical_features, categorical_features = identify_features(
            df, target_column)

    # Create subplots for numerical features
    if numerical_features:
        print("\nVisualizing Numerical Features:")
        print("-" * 50)

        # Calculate number of rows and columns for subplots
        n_features = len(numerical_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        # Create figure for histograms and box plots
        plt.figure(figsize=figsize)

        # Plot histograms and box plots
        for i, feature in enumerate(numerical_features, 1):
            # Histogram
            plt.subplot(n_rows, n_cols, i)
            sns.histplot(data=df, x=feature, kde=True, bins=30)
            plt.title(f'Distribution of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Count')

        plt.tight_layout()
        plt.show()

        # Create figure for box plots
        plt.figure(figsize=figsize)

        for i, feature in enumerate(numerical_features, 1):
            # Box plot
            plt.subplot(n_rows, n_cols, i)
            sns.boxplot(data=df, y=feature)
            plt.title(f'Box Plot of {feature}')

        plt.tight_layout()
        plt.show()

        # Create correlation heatmap
        print("\nCorrelation Heatmap:")
        plt.figure(figsize=(15, 12))
        corr_matrix = df[numerical_features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f',
                    square=True, linewidths=.5, cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix of Numerical Features')
        plt.tight_layout()
        plt.show()

        # Create scatter plots for highly correlated features
        print("\nScatter Plots for Highly Correlated Features:")
        high_corr = corr_matrix[abs(corr_matrix) > 0.5].stack().reset_index()
        high_corr = high_corr[high_corr['level_0'] != high_corr['level_1']]
        high_corr = high_corr.sort_values(0, ascending=False).drop_duplicates()

        if not high_corr.empty:
            n_pairs = min(len(high_corr), 6)  # Show top 6 pairs
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            axes = axes.flatten()

            for i, (_, row) in enumerate(high_corr.head(n_pairs).iterrows()):
                x, y = row['level_0'], row['level_1']
                sns.scatterplot(data=df, x=x, y=y, hue=target_column if target_column in df.columns else None,
                                alpha=0.5, ax=axes[i])
                axes[i].set_title(f'{x} vs {y}\nCorrelation: {row[0]:.2f}')

            plt.tight_layout()
            plt.show()

    # Create plots for categorical features
    if categorical_features:
        print("\nVisualizing Categorical Features:")
        print("-" * 50)

        # Calculate number of rows and columns for subplots
        n_features = len(categorical_features)
        n_cols = 2
        n_rows = (n_features + n_cols - 1) // n_cols

        # Create figure for count plots
        plt.figure(figsize=figsize)

        for i, feature in enumerate(categorical_features, 1):
            plt.subplot(n_rows, n_cols, i)
            sns.countplot(
                data=df, x=feature, hue=target_column if target_column in df.columns else None)
            plt.title(f'Count Plot of {feature}')
            plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.show()

    print("=" * 50)


def visualize_trends(df, numerical_features=None, time_column=None, target_column='Target', figsize=(20, 15)):
    """
    Visualize trends in numerical features over time.

    Args:
        df (pandas.DataFrame): The dataset
        numerical_features (list): List of numerical feature names
        time_column (str): Name of the time column
        target_column (str): Name of the target column
        figsize (tuple): Figure size (width, height)
    """
    print("\n=== Visualizing Trends ===")
    print("=" * 50)

    # Call analyze_trends for visualization
    analyze_trends(df, numerical_features, time_column, target_column, figsize)

    print("=" * 50)


def analyze_trends(df, numerical_features=None, time_column=None, target_column='Target',
                   figsize=(20, 15), threshold=3):
    """
    Analyze trends and detect anomalies in the data.

    Args:
        df (pandas.DataFrame): The dataset
        numerical_features (list): List of numerical feature names
        time_column (str): Name of the time column
        target_column (str): Name of the target column
        figsize (tuple): Figure size (width, height)
        threshold (float): Threshold for anomaly detection (number of standard deviations)
    """
    print("\n=== Trend and Anomaly Analysis ===")
    print("=" * 50)

    # Select numerical columns if not provided
    if numerical_features is None:
        numerical_features = df.select_dtypes(
            include=['int64', 'float64']).columns.tolist()

    if not numerical_features:
        print("No numerical features found for trend analysis.")
        return

    # If time column is not provided, try to find one
    if time_column is None:
        time_columns = ['Month', 'Day', 'Time of Day', 'Date', 'Timestamp']
        for col in time_columns:
            if col in df.columns:
                time_column = col
                break

    # Trend Analysis
    if time_column and time_column in df.columns:
        print(f"\nTrend Analysis Over {time_column}:")
        print("-" * 50)

        # Calculate number of rows and columns for subplots
        n_features = len(numerical_features)
        n_cols = 2
        n_rows = (n_features + n_cols - 1) // n_cols

        # Create figure for trend plots
        plt.figure(figsize=figsize)

        for i, feature in enumerate(numerical_features, 1):
            plt.subplot(n_rows, n_cols, i)

            if target_column in df.columns:
                # Group by time and target, calculate mean
                trend_data = df.groupby([time_column, target_column])[
                    feature].mean().unstack()
                trend_data.plot(ax=plt.gca(), marker='o')
            else:
                # Group by time, calculate mean
                trend_data = df.groupby(time_column)[feature].mean()
                trend_data.plot(ax=plt.gca(), marker='o')

            plt.title(f'Trend of {feature} Over {time_column}')
            plt.xlabel(time_column)
            plt.ylabel(feature)
            plt.grid(True)

        plt.tight_layout()
        plt.show()

        # Print trend statistics
        print("\nTrend Statistics:")
        print("-" * 50)
        for feature in numerical_features:
            if target_column in df.columns:
                trend_stats = df.groupby([time_column, target_column])[
                    feature].agg(['mean', 'std', 'min', 'max'])
                print(
                    f"\n{feature} Statistics by {time_column} and {target_column}:")
                print(trend_stats)
            else:
                trend_stats = df.groupby(time_column)[feature].agg(
                    ['mean', 'std', 'min', 'max'])
                print(f"\n{feature} Statistics by {time_column}:")
                print(trend_stats)
    else:
        print("\nNo time column found for trend analysis.")

    # Anomaly Detection
    print("\nAnomaly Detection:")
    print("-" * 50)

    for feature in numerical_features:
        # Calculate mean and standard deviation
        mean = df[feature].mean()
        std = df[feature].std()

        # Identify anomalies
        anomalies = df[abs(df[feature] - mean) > threshold * std]

        if not anomalies.empty:
            print(
                f"\nAnomalies in {feature} (> {threshold} standard deviations):")
            print(f"Number of anomalies: {len(anomalies)}")
            print(f"Percentage of data: {(len(anomalies)/len(df)*100):.2f}%")

            # Create box plot with anomalies highlighted
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, y=feature)
            plt.scatter(x=[0]*len(anomalies), y=anomalies[feature],
                        color='red', alpha=0.5, label='Anomalies')
            plt.title(f'Box Plot of {feature} with Anomalies Highlighted')
            plt.legend()
            plt.show()

            # Print anomaly details
            print("\nAnomaly Details:")
            print(anomalies[[feature, time_column, target_column]
                            if time_column and target_column in df.columns
                            else [feature]].describe())
        else:
            print(
                f"\nNo anomalies found in {feature} (> {threshold} standard deviations)")

    print("=" * 50)


def plot_horizontal_boxplots(df, exclude_cols=None, figsize=(15, 10)):
    """
    Create horizontal boxplots of numerical features, excluding specified columns.

    Args:
        df (pandas.DataFrame): The dataset
        exclude_cols (list): List of columns to exclude from the plot
        figsize (tuple): Figure size (width, height)
    """
    print("\n=== Horizontal Boxplots of Numerical Features ===")
    print("=" * 50)

    # Default columns to exclude
    if exclude_cols is None:
        exclude_cols = ['Time of Day', 'Day', 'Month']

    # Select numerical columns and exclude specified columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    plot_cols = [col for col in numerical_cols if col not in exclude_cols]

    if not plot_cols:
        print("No numerical features to plot after exclusions.")
        return

    print(f"\nPlotting {len(plot_cols)} numerical features:")
    for col in plot_cols:
        print(f"- {col}")

    # Create horizontal boxplot
    plt.figure(figsize=figsize)
    sns.boxplot(data=df[plot_cols], orient='h')
    plt.title('Horizontal Boxplots of Numerical Features', fontsize=16, pad=20)
    plt.xlabel('Value')
    plt.ylabel('Feature')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Print basic statistics
    print("\nBasic Statistics:")
    print("-" * 50)
    stats = df[plot_cols].describe().T
    stats['count'] = stats['count'].astype(int)
    for col in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
        if col in stats.columns:
            stats[col] = stats[col].round(4)
    print(stats)
    print("=" * 50)


def plot_correlation_heatmap(df, numerical_features=None, figsize=(14, 12),
                             cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5,
                             threshold=0.1, include_target=True):
    """
    Create a correlation heatmap for numerical features and identify significant correlations.

    Args:
        df (pandas.DataFrame): The dataset
        numerical_features (list): List of numerical feature names
        figsize (tuple): Figure size (width, height)
        cmap (str): Color map for the heatmap
        annot (bool): Whether to annotate the heatmap
        fmt (str): Format string for annotations
        linewidths (float): Width of the lines between cells
        threshold (float): Minimum correlation value to consider significant
        include_target (bool): Whether to include target column in correlation analysis
    """
    print("\n=== Correlation Analysis ===")
    print("=" * 50)

    # Select numerical columns if not provided
    if numerical_features is None:
        numerical_features = df.select_dtypes(
            include=['int64', 'float64']).columns.tolist()

    if not numerical_features:
        print("No numerical features found for correlation analysis.")
        return

    # Include target column if specified
    if include_target and 'Target' in df.columns and 'Target' not in numerical_features:
        numerical_features.append('Target')

    print(
        f"\nAnalyzing correlations between {len(numerical_features)} features:")
    for feature in numerical_features:
        print(f"- {feature}")

    # Calculate correlation matrix
    correlation_matrix = df[numerical_features].corr()

    # Create full correlation heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix,
                annot=annot,
                cmap=cmap,
                fmt=fmt,
                linewidths=linewidths,
                square=True,
                cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix of Numerical Features',
              fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()

    # Identify significant correlations
    significant_correlations = correlation_matrix[
        (correlation_matrix > threshold) & (correlation_matrix != 1.0)
    ].dropna(how='all').dropna(axis=1, how='all')

    # Create heatmap of significant correlations
    if not significant_correlations.empty:
        plt.figure(figsize=figsize)
        sns.heatmap(significant_correlations,
                    annot=annot,
                    cmap=cmap,
                    fmt=fmt,
                    linewidths=linewidths,
                    square=True,
                    cbar_kws={"shrink": .8})
        plt.title('Significant Correlations in Numerical Data',
                  fontsize=16, pad=20)
        plt.tight_layout()
        plt.show()

        # Print significant correlations
        print(f"\nSignificant Correlations (|r| > {threshold}):")
        print("-" * 50)
        print(significant_correlations)

        # Print feature pairs with significant correlations
        print("\nFeature Pairs with Significant Correlations:")
        print("-" * 50)
        for col in significant_correlations.columns:
            for idx in significant_correlations.index:
                if not pd.isna(significant_correlations.loc[idx, col]):
                    print(
                        f"{idx} & {col}: {significant_correlations.loc[idx, col]:.3f}")

        # Print correlations with target if included
        if include_target and 'Target' in numerical_features:
            print("\nCorrelations with Target:")
            print("-" * 50)
            target_correlations = correlation_matrix['Target'].drop('Target')
            target_correlations = target_correlations[abs(
                target_correlations) > threshold]
            print(target_correlations.sort_values(ascending=False))
    else:
        print(
            f"\nNo significant correlations found above threshold {threshold}")

    # Print correlation summary
    print("\nStrong Correlations (|r| > 0.5):")
    strong_correlations = correlation_matrix.unstack().sort_values(ascending=False)
    # Remove self-correlations
    strong_correlations = strong_correlations[strong_correlations != 1]
    strong_correlations = strong_correlations[abs(strong_correlations) > 0.5]
    print(strong_correlations)

    # Print feature pairs with highest correlations
    if not strong_correlations.empty:
        print("\nTop 5 Feature Pairs by Correlation:")
        for (feature1, feature2), corr in strong_correlations.head(5).items():
            print(f"{feature1} & {feature2}: {corr:.3f}")

    print("=" * 50)


def visualize_transformed_distributions(df_transformed, numerical_features=None, figsize=(20, 15), bins=15):
    """
    Visualize distributions of numerical features after preprocessing.

    Args:
        df_transformed (pandas.DataFrame): The preprocessed dataset
        numerical_features (list): List of numerical feature names
        figsize (tuple): Figure size (width, height)
        bins (int): Number of bins for histograms
    """
    print("\n=== Visualizing Transformed Distributions ===")
    print("=" * 50)

    # Select numerical columns if not provided
    if numerical_features is None:
        numerical_features = df_transformed.select_dtypes(
            include=['int64', 'float64']).columns.tolist()

    if not numerical_features:
        print("No numerical features found in the transformed dataset.")
        return

    print(f"\nVisualizing {len(numerical_features)} numerical features:")
    print("-" * 50)

    # Create histograms
    plt.figure(figsize=figsize)
    df_transformed[numerical_features].hist(bins=bins)
    plt.suptitle('Histograms of Numerical Features After Transformation',
                 fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    # Create horizontal boxplots
    plot_horizontal_boxplots(df_transformed)

    # Create correlation heatmap
    plot_correlation_heatmap(df_transformed, numerical_features)

    print("=" * 50)


def save_preprocessed_data(df, file_path="data/preprocessed.csv", index=False):
    """
    Save the preprocessed DataFrame to a CSV file.

    Args:
        df (pandas.DataFrame): The preprocessed DataFrame to save
        file_path (str): Path where the CSV file will be saved
        index (bool): Whether to include the index in the saved file
    """
    print("\n=== Saving Preprocessed Data ===")
    print("=" * 50)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    try:
        # Save the DataFrame to CSV
        df.to_csv(file_path, index=index)
        print(f"\nPreprocessed data successfully saved to: {file_path}")
        print(f"Shape of saved data: {df.shape}")
        print(f"Number of features: {len(df.columns)}")
        print(f"Number of samples: {len(df)}")

        # Print first few rows of the saved data
        print("\nFirst few rows of saved data:")
        print(df.head())

    except Exception as e:
        print(f"\nError saving preprocessed data: {str(e)}")

    print("=" * 50)


def create_interaction_features(df, numerical_features=None, target_column='Target'):
    """
    Create new features by combining existing numerical features.

    Args:
        df (pandas.DataFrame): The dataset
        numerical_features (list): List of numerical feature names
        target_column (str): Name of the target column to exclude

    Returns:
        pandas.DataFrame: Dataset with new interaction features
    """
    print("\n=== Creating Interaction Features ===")
    print("=" * 50)

    # Make a copy of the dataframe
    df_interactions = df.copy()

    # Select numerical columns if not provided
    if numerical_features is None:
        numerical_features = df.select_dtypes(
            include=['int64', 'float64']).columns.tolist()
        if target_column in numerical_features:
            numerical_features.remove(target_column)

    if not numerical_features:
        print("No numerical features found for creating interactions.")
        return df_interactions

    print(
        f"\nCreating interactions between {len(numerical_features)} features:")
    for feature in numerical_features:
        print(f"- {feature}")

    # Create interaction features
    new_features = []

    # 1. Multiplicative interactions
    for i, feat1 in enumerate(numerical_features):
        for feat2 in numerical_features[i+1:]:
            interaction_name = f"{feat1}_x_{feat2}"
            df_interactions[interaction_name] = df_interactions[feat1] * \
                df_interactions[feat2]
            new_features.append(interaction_name)
            print(f"Created {interaction_name}")

    # 2. Ratio interactions
    for i, feat1 in enumerate(numerical_features):
        for feat2 in numerical_features[i+1:]:
            # Avoid division by zero
            if (df_interactions[feat2] != 0).all():
                ratio_name = f"{feat1}_ratio_{feat2}"
                df_interactions[ratio_name] = df_interactions[feat1] / \
                    df_interactions[feat2]
                new_features.append(ratio_name)
                print(f"Created {ratio_name}")

    # 3. Sum and difference interactions
    for i, feat1 in enumerate(numerical_features):
        for feat2 in numerical_features[i+1:]:
            sum_name = f"{feat1}_plus_{feat2}"
            diff_name = f"{feat1}_minus_{feat2}"
            df_interactions[sum_name] = df_interactions[feat1] + \
                df_interactions[feat2]
            df_interactions[diff_name] = df_interactions[feat1] - \
                df_interactions[feat2]
            new_features.extend([sum_name, diff_name])
            print(f"Created {sum_name} and {diff_name}")

    # Print summary of new features
    print(f"\nCreated {len(new_features)} new interaction features:")
    for feature in new_features:
        print(f"- {feature}")

    # Print basic statistics of new features
    print("\nBasic Statistics of New Features:")
    print(df_interactions[new_features].describe().T)

    print("=" * 50)
    return df_interactions


def create_aggregate_features(df, numerical_features=None, categorical_features=None,
                              target_column='Target', time_column=None):
    """
    Create aggregate features from existing features.

    Args:
        df (pandas.DataFrame): The dataset
        numerical_features (list): List of numerical feature names
        categorical_features (list): List of categorical feature names
        target_column (str): Name of the target column
        time_column (str): Name of the time column

    Returns:
        pandas.DataFrame: Dataset with new aggregate features
    """
    print("\n=== Creating Aggregate Features ===")
    print("=" * 50)

    # Make a copy of the dataframe
    df_aggregates = df.copy()

    # Select numerical columns if not provided
    if numerical_features is None:
        numerical_features = df.select_dtypes(
            include=['int64', 'float64']).columns.tolist()
        if target_column in numerical_features:
            numerical_features.remove(target_column)

    # Select categorical columns if not provided
    if categorical_features is None:
        categorical_features = df.select_dtypes(
            include=['object', 'category']).columns.tolist()

    if not numerical_features and not categorical_features:
        print("No features found for creating aggregates.")
        return df_aggregates

    new_features = []

    # 1. Statistical aggregates for numerical features
    if numerical_features:
        print("\nCreating statistical aggregates for numerical features:")
        for feature in numerical_features:
            # Mean, median, std, min, max
            df_aggregates[f"{feature}_mean"] = df_aggregates[feature].mean()
            df_aggregates[f"{feature}_median"] = df_aggregates[feature].median()
            df_aggregates[f"{feature}_std"] = df_aggregates[feature].std()
            df_aggregates[f"{feature}_min"] = df_aggregates[feature].min()
            df_aggregates[f"{feature}_max"] = df_aggregates[feature].max()
            new_features.extend([
                f"{feature}_mean", f"{feature}_median", f"{feature}_std",
                f"{feature}_min", f"{feature}_max"
            ])
            print(f"Created statistical aggregates for {feature}")

    # 2. Time-based aggregates if time column exists
    if time_column and time_column in df.columns:
        print("\nCreating time-based aggregates:")
        for feature in numerical_features:
            # Group by time and calculate statistics
            time_stats = df_aggregates.groupby(time_column)[feature].agg([
                'mean', 'median', 'std', 'min', 'max'
            ]).add_prefix(f"{feature}_time_")

            # Merge statistics back to original dataframe
            df_aggregates = df_aggregates.merge(
                time_stats, left_on=time_column, right_index=True
            )
            new_features.extend(time_stats.columns.tolist())
            print(f"Created time-based aggregates for {feature}")

    # 3. Target-based aggregates if target column exists
    if target_column in df.columns:
        print("\nCreating target-based aggregates:")
        for feature in numerical_features:
            # Group by target and calculate statistics
            target_stats = df_aggregates.groupby(target_column)[feature].agg([
                'mean', 'median', 'std', 'min', 'max'
            ]).add_prefix(f"{feature}_target_")

            # Merge statistics back to original dataframe
            df_aggregates = df_aggregates.merge(
                target_stats, left_on=target_column, right_index=True
            )
            new_features.extend(target_stats.columns.tolist())
            print(f"Created target-based aggregates for {feature}")

    # 4. Categorical feature aggregates
    if categorical_features:
        print("\nCreating categorical feature aggregates:")
        for feature in categorical_features:
            # Count of each category
            category_counts = df_aggregates[feature].value_counts()
            df_aggregates[f"{feature}_count"] = df_aggregates[feature].map(
                category_counts)
            new_features.append(f"{feature}_count")
            print(f"Created count aggregate for {feature}")

    # Print summary of new features
    print(f"\nCreated {len(new_features)} new aggregate features:")
    for feature in new_features:
        print(f"- {feature}")

    # Print basic statistics of new features
    print("\nBasic Statistics of New Features:")
    print(df_aggregates[new_features].describe().T)

    print("=" * 50)
    return df_aggregates


def create_domain_features(df, numerical_features=None, target_column='Target'):
    """
    Create domain-specific features for water quality analysis.

    Args:
        df (pandas.DataFrame): The dataset
        numerical_features (list): List of numerical feature names
        target_column (str): Name of the target column to exclude

    Returns:
        tuple: (df_with_features, updated_numerical_features)
    """
    print("\n=== Creating Domain-Specific Features ===")
    print("=" * 50)

    # Make a copy of the dataframe
    df_domain = df.copy()

    # Select numerical columns if not provided
    if numerical_features is None:
        numerical_features = df.select_dtypes(
            include=['int64', 'float64']).columns.tolist()
        if target_column in numerical_features:
            numerical_features.remove(target_column)

    if not numerical_features:
        print("No numerical features found for creating domain features.")
        return df_domain, numerical_features

    print("\nCreating water quality specific features:")

    # 1. Temperature-based features
    if 'Water Temperature' in df.columns and 'Air Temperature' in df.columns:
        # Temperature ratio (avoiding division by zero)
        df_domain['Water_Temp_to_Air_Temp_Ratio'] = df_domain['Water Temperature'] / \
            (df_domain['Air Temperature'] + 1)
        print("Created Water_Temp_to_Air_Temp_Ratio")

        # Temperature difference
        df_domain['Temp_Difference'] = df_domain['Water Temperature'] - \
            df_domain['Air Temperature']
        print("Created Temp_Difference")

        numerical_features.extend(
            ['Water_Temp_to_Air_Temp_Ratio', 'Temp_Difference'])

    # 2. Metal-related features
    metal_columns = ['Iron', 'Lead', 'Zinc', 'Copper', 'Manganese']
    if all(col in df.columns for col in metal_columns):
        # Total metals
        df_domain['Total_Metals'] = df_domain[metal_columns].sum(axis=1)
        print("Created Total_Metals")

        # Heavy metals ratio
        df_domain['Heavy_Metals_Ratio'] = df_domain['Lead'] / \
            (df_domain['Iron'] + 1)
        print("Created Heavy_Metals_Ratio")

        # Metal concentration index
        df_domain['Metal_Concentration_Index'] = df_domain[metal_columns].mean(
            axis=1)
        print("Created Metal_Concentration_Index")

        numerical_features.extend(
            ['Total_Metals', 'Heavy_Metals_Ratio', 'Metal_Concentration_Index'])

    # 3. Water quality indices
    if 'pH' in df.columns and 'Total Dissolved Solids' in df.columns:
        # pH-TDS interaction
        df_domain['pH_TDS_Interaction'] = df_domain['pH'] * \
            df_domain['Total Dissolved Solids']
        print("Created pH_TDS_Interaction")

        # Water quality score
        df_domain['Water_Quality_Score'] = (
            df_domain['pH'] * 0.3 +
            (1 / (df_domain['Total Dissolved Solids'] + 1)) * 0.3 +
            (1 / (df_domain['Total_Metals'] + 1)) * 0.4
        )
        print("Created Water_Quality_Score")

        numerical_features.extend(
            ['pH_TDS_Interaction', 'Water_Quality_Score'])

    # 4. Mineral balance features
    if 'Calcium' in df.columns and 'Magnesium' in df.columns:
        # Calcium-Magnesium ratio
        df_domain['Ca_Mg_Ratio'] = df_domain['Calcium'] / \
            (df_domain['Magnesium'] + 1)
        print("Created Ca_Mg_Ratio")

        # Total hardness
        df_domain['Total_Hardness'] = 2.5 * \
            df_domain['Calcium'] + 4.1 * df_domain['Magnesium']
        print("Created Total_Hardness")

        numerical_features.extend(['Ca_Mg_Ratio', 'Total_Hardness'])

    # Print summary of new features
    new_features = [f for f in numerical_features if f not in df.columns]
    print(f"\nCreated {len(new_features)} new domain-specific features:")
    for feature in new_features:
        print(f"- {feature}")

    # Print basic statistics of new features
    print("\nBasic Statistics of New Features:")
    print(df_domain[new_features].describe().T)

    print("=" * 50)
    return df_domain, numerical_features


def split_data(X, y, test_size=0.2, random_state=42, stratify=None):
    """
    Split the dataset into training and testing sets.

    Args:
        X (pandas.DataFrame): Features
        y (pandas.Series): Target variable
        test_size (float): Proportion of the dataset to include in the test split
        random_state (int): Controls the shuffling applied to the data before splitting
        stratify (array-like): If not None, data is split in a stratified fashion

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("\n=== Splitting Data into Training and Testing Sets ===")
    print("=" * 50)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    # Print split information
    print(f"\nTraining Set Size: {X_train.shape[0]} samples")
    print(f"Testing Set Size: {X_test.shape[0]} samples")
    print(f"Total Features: {X_train.shape[1]}")

    # Print target distribution in training and testing sets
    if stratify is not None:
        print("\nTarget Distribution in Training Set:")
        print(y_train.value_counts(normalize=True).round(4))
        print("\nTarget Distribution in Testing Set:")
        print(y_test.value_counts(normalize=True).round(4))

    print("=" * 50)
    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train, random_state=42):
    """
    Train multiple machine learning models.

    Args:
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training target
        random_state (int): Random state for reproducibility

    Returns:
        dict: Dictionary of trained models
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB

    print("\n=== Training Machine Learning Models ===")
    print("=" * 50)

    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=random_state),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
        'Support Vector Machine': SVC(random_state=random_state),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB()
    }

    # Train each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        print(f"Training completed for {name}")

    print("=" * 50)
    return models


def evaluate_models(models, X_test, y_test):
    """
    Evaluate trained models using various metrics.

    Args:
        models (dict): Dictionary of trained models
        X_test (pandas.DataFrame): Testing features
        y_test (pandas.Series): Testing target

    Returns:
        pandas.DataFrame: Evaluation metrics for each model
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import confusion_matrix, classification_report

    print("\n=== Evaluating Model Performance ===")
    print("=" * 50)

    # Initialize results dictionary
    results = {}

    # Evaluate each model
    for name, model in models.items():
        print(f"\nEvaluating {name}...")

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Store results
        results[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }

        # Print detailed metrics
        print(f"\n{name} Performance:")
        print("-" * 30)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Print confusion matrix
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    # Convert results to DataFrame
    results_df = pd.DataFrame(results).T

    # Print summary
    print("\nModel Performance Summary:")
    print("-" * 50)
    print(results_df.round(4))

    print("=" * 50)
    return results_df


def evaluate_single_model(model, X_test, y_test):
    """
    Evaluate a single model using accuracy, precision, recall, and F1-score.

    Args:
        model: Trained machine learning model
        X_test (pandas.DataFrame): Testing features
        y_test (pandas.Series): Testing target

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    print("\n=== Evaluating Model Performance ===")
    print("=" * 50)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Create results dictionary
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    # Print results
    print("\nModel Performance:")
    print("-" * 30)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("=" * 50)
    return results


def get_results(y_test, y_pred):
    """
    Calculate and return evaluation metrics for model predictions.

    Args:
        y_test (array-like): True labels
        y_pred (array-like): Predicted labels

    Returns:
        dict: Dictionary containing accuracy, precision, recall, and F1-score
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Create results dictionary
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    return results


def explore_data(df):
    """
    Perform initial data exploration and print key statistics.
    """
    # Display initial dataframe
    display_initial_dataframe(df)

    # Identify features
    numerical_features, categorical_features = identify_features(df)

    # Perform comprehensive data analysis
    comprehensive_data_analysis(df, numerical_features, categorical_features)

    # Display comprehensive statistics
    display_dataframe_statistics(df, numerical_features, categorical_features)

    # Visualize distributions and relationships
    visualize_distributions(df, numerical_features, categorical_features)

    # Visualize trends
    visualize_trends(df, numerical_features)

    # Plot histograms of numerical features before transformation
    plot_numerical_histograms(
        df, numerical_features, title='Histograms of Numerical Features Before Transformation')

    print("\n=== Data Quality Summary ===")
    summary = get_data_summary(df)
    print(summary)

    # Display missing counts for all features
    display_missing_counts(df)

    # Visualize missing values
    visualize_missing_values(df)

    # Analyze missing values
    analyze_missing_values(df)

    # Display missing counts for categorical features
    display_categorical_missing_counts(df)

    # Handle missing values
    df_processed = handle_missing_values(df)

    # Create domain-specific features
    df_domain, numerical_features = create_domain_features(
        df_processed, numerical_features)

    # Create interaction features
    df_interactions = create_interaction_features(
        df_domain, numerical_features)

    # Create aggregate features
    df_aggregates = create_aggregate_features(
        df_interactions, numerical_features, categorical_features)

    # Preprocess all features using ColumnTransformer
    df_encoded, preprocessor, feature_names = preprocess_features(
        df_aggregates, numerical_features, categorical_features)

    # Visualize transformed distributions
    visualize_transformed_distributions(df_encoded)

    # Analyze feature statistics
    analyze_feature_statistics(
        df_encoded, numerical_features, categorical_features)

    # Analyze memory usage
    analyze_memory_usage(df_encoded)

    # Analyze target distribution with default class labels for water potability
    class_labels = {0: 'Not Potable',
                    1: 'Potable'} if 'Target' in df_encoded.columns else None
    analyze_target_distribution(
        df_encoded, class_labels=class_labels, plot_type='bar')
    analyze_target_distribution(
        df_encoded, class_labels=class_labels, plot_type='pie')

    # Analyze numerical features
    analyze_numerical_features(df_encoded, plot=True)

    # Analyze categorical features
    analyze_categorical_features(df_encoded)

    # Save preprocessed data
    save_preprocessed_data(df_encoded)

    # Split the data into training and testing sets
    X = df_encoded.drop('Target', axis=1)
    y = df_encoded['Target']
    X_train, X_test, y_train, y_test = split_data(X, y, stratify=y)

    # Train and evaluate models
    models = train_models(X_train, y_train)
    results_df = evaluate_models(models, X_test, y_test)

    # Evaluate single model (Random Forest)
    model_rf = models['Random Forest']
    single_model_results = evaluate_single_model(model_rf, X_test, y_test)

    return X_train, X_test, y_train, y_test, preprocessor, numerical_features, categorical_features, feature_names, models, results_df, single_model_results


def main():
    """
    Main function to demonstrate data loading and exploration.
    """
    # Set up environment
    setup_environment()

    # Load data
    df = load_data()

    # Explore data
    explore_data(df)

    return df


if __name__ == "__main__":
    main()
