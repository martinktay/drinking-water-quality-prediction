import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import shap
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


def create_visualization_directory():
    """Create directory for saving visualizations."""
    os.makedirs('visualizations', exist_ok=True)


def plot_feature_distributions(df):
    """Plot distributions of numerical features."""
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

    for column in numerical_columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=column, kde=True)
        plt.title(f'Distribution of {column}')
        plt.savefig(f'visualizations/{column}_distribution.png')
        plt.close()


def plot_correlation_matrix(df):
    """Plot correlation matrix of numerical features."""
    plt.figure(figsize=(15, 12))
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = df[numerical_columns].corr()

    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.savefig('visualizations/correlation_matrix.png')
    plt.close()


def plot_target_distribution(df):
    """Plot distribution of target variable."""
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Target')
    plt.title('Distribution of Water Quality Classes')
    plt.savefig('visualizations/target_distribution.png')
    plt.close()


def plot_feature_importance(model, feature_names, model_name):
    """Plot feature importance from the model."""
    plt.figure(figsize=(12, 8))

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.title(f'Feature Importance - {model_name}')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [
                   feature_names[i] for i in indices], rotation=90)
    elif hasattr(model, 'coef_'):
        # For MLP and other linear models
        importances = np.abs(model.coef_[0])
        indices = np.argsort(importances)[::-1]

        plt.title(f'Feature Importance (Coefficients) - {model_name}')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [
                   feature_names[i] for i in indices], rotation=90)
    else:
        print(f"Cannot plot feature importance for {model_name}")
        return

    plt.tight_layout()
    plt.savefig(f'visualizations/feature_importance_{model_name}.png')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix for model predictions."""
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'visualizations/confusion_matrix_{model_name}.png')
    plt.close()


def plot_shap_values(model, X_test, feature_names, model_name):
    """Plot SHAP values for model interpretation."""
    try:
        if hasattr(model, 'feature_importances_'):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)

            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_test,
                              feature_names=feature_names, show=False)
            plt.title(f'SHAP Values Summary - {model_name}')
            plt.tight_layout()
            plt.savefig(f'visualizations/shap_summary_{model_name}.png')
            plt.close()
    except Exception as e:
        print(f"Could not generate SHAP values for {model_name}: {str(e)}")


def main():
    # Create visualization directory
    create_visualization_directory()

    # Load data
    df = pd.read_csv('water_quality_dataset_100k_new.csv')

    # Plot basic visualizations
    plot_feature_distributions(df)
    plot_correlation_matrix(df)
    plot_target_distribution(df)

    # Load trained models and test data
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv').squeeze()
    feature_names = X_test.columns.tolist()

    # Load and plot for each model
    models = ['random_forest', 'xgboost', 'lightgbm', 'catboost', 'mlp']
    for model_name in models:
        try:
            model = joblib.load(f'models/{model_name}.joblib')
            y_pred = model.predict(X_test)

            plot_confusion_matrix(y_test, y_pred, model_name)
            plot_feature_importance(model, feature_names, model_name)
            plot_shap_values(model, X_test, feature_names, model_name)
        except Exception as e:
            print(f"Error processing {model_name}: {str(e)}")

    print("Visualizations created successfully!")


if __name__ == "__main__":
    main()
