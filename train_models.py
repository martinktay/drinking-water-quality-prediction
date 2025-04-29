import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Create necessary directories
Path('models').mkdir(exist_ok=True)
Path('reports/model_performance').mkdir(parents=True, exist_ok=True)


def load_data(sample_size=100000):
    """Load the processed dataset with optional sampling."""
    df = pd.read_csv('data/processed/processed_data.csv', index_col=False)

    # Remove any unnamed columns (typically index columns)
    unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)

    print("Available columns:", df.columns.tolist())
    print(f"Using {len(df)} samples for training")
    return df


def preprocess_data(df):
    """Prepare data for training."""
    # Define numeric columns
    numeric_columns = [
        'pH', 'Iron', 'Nitrate', 'Chloride', 'Lead', 'Zinc',
        'Turbidity', 'Fluoride', 'Copper', 'Sulfate', 'Conductivity',
        'Chlorine', 'Total Dissolved Solids', 'Water Temperature',
        'Air Temperature'
    ]

    # Split features and target
    X = df[numeric_columns]
    y = df['Target']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def create_preprocessor():
    """Create the feature preprocessor."""
    numeric_features = [
        'pH', 'Iron', 'Nitrate', 'Chloride', 'Lead', 'Zinc',
        'Turbidity', 'Fluoride', 'Copper', 'Sulfate', 'Conductivity',
        'Chlorine', 'Total Dissolved Solids', 'Water Temperature',
        'Air Temperature'
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features)
        ])

    return preprocessor


def evaluate_model(model, X_test, y_test, name):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }

    # Save detailed classification report
    report = classification_report(y_test, y_pred)
    with open(f'reports/model_performance/{name}_report.txt', 'w') as f:
        f.write(report)

    return metrics


def plot_model_comparison(results, stage="before_tuning"):
    """Plot model comparison."""
    metrics_df = pd.DataFrame(results).T

    plt.figure(figsize=(12, 6))
    metrics_df.plot(kind='bar', figsize=(12, 6))
    plt.title(f'Model Performance Comparison ({stage})')
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.legend(title='Metrics')
    plt.tight_layout()
    plt.savefig(f'reports/model_performance/comparison_{stage}.png')
    plt.close()


def train_and_save_models():
    """Train and evaluate models."""
    print("Loading data...")
    df = load_data(sample_size=100000)  # Use 100k samples for faster training

    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("Creating preprocessor...")
    preprocessor = create_preprocessor()

    # Define models with initial parameters (reduced complexity)
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=50, max_depth=6, random_state=42),
        'LightGBM': lgb.LGBMClassifier(n_estimators=50, max_depth=6, random_state=42),
        'DecisionTree': DecisionTreeClassifier(max_depth=6, random_state=42),
        'LinearSVC': LinearSVC(random_state=42, max_iter=1000)
    }

    # Train and evaluate models
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('clf', model)
        ])

        # Train model
        pipeline.fit(X_train, y_train)
        print(f"{name} training completed")

        # Evaluate model
        print(f"Evaluating {name}...")
        metrics = evaluate_model(pipeline, X_test, y_test, name)
        results[name] = metrics
        print(f"{name} metrics:", metrics)

        # Save model
        print(f"Saving {name}...")
        joblib.dump(pipeline, f'models/{name.lower()}.pkl')

    # Save preprocessor
    print("Saving preprocessor...")
    joblib.dump(preprocessor, 'models/preprocessor.pkl')

    # Save and plot results
    with open('reports/model_performance/model_metrics.json', 'w') as f:
        json.dump(results, f, indent=4)

    plot_model_comparison(results, "initial_training")

    print("Models and preprocessor saved successfully!")
    return results


def train_and_tune_models():
    """Train and tune models with hyperparameter optimization."""
    print("Loading data...")
    df = load_data(sample_size=100000)

    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)
    preprocessor = create_preprocessor()

    # Define parameter grids for each model
    param_grids = {
        'RandomForest': {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [10, 20],
            'clf__min_samples_split': [2, 5],
            'clf__min_samples_leaf': [1, 2]
        },
        'XGBoost': {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [6, 8],
            'clf__learning_rate': [0.01, 0.1],
            'clf__min_child_weight': [1, 3]
        },
        'LightGBM': {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [6, 8],
            'clf__learning_rate': [0.01, 0.1],
            'clf__num_leaves': [31, 50]
        },
        'DecisionTree': {
            'clf__max_depth': [10, 20],
            'clf__min_samples_split': [2, 5],
            'clf__min_samples_leaf': [1, 2]
        },
        'LinearSVC': {
            'clf__C': [0.1, 1.0],
            'clf__max_iter': [2000]
        }
    }

    # Initialize models
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42),
        'LightGBM': lgb.LGBMClassifier(random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'LinearSVC': LinearSVC(random_state=42)
    }

    # Train and tune models
    results = {}
    for name, model in models.items():
        print(f"\nTuning {name}...")
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('clf', model)
        ])

        # Perform grid search
        grid_search = GridSearchCV(
            pipeline,
            param_grids[name],
            cv=5,
            scoring='f1',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        # Save best model
        print(f"Best parameters for {name}: {grid_search.best_params_}")
        joblib.dump(grid_search.best_estimator_,
                    f'models/{name.lower()}_tuned.pkl')

        # Evaluate best model
        metrics = evaluate_model(
            grid_search.best_estimator_, X_test, y_test, f"{name}_tuned")
        results[name] = metrics
        print(f"{name} tuned metrics:", metrics)

    # Save tuned results
    with open('reports/model_performance/model_metrics_tuned.json', 'w') as f:
        json.dump(results, f, indent=4)

    plot_model_comparison(results, "after_tuning")
    return results


if __name__ == "__main__":
    # Clean up old files
    for f in Path('models').glob('*.pkl'):
        f.unlink()
    for f in Path('reports/model_performance').glob('*'):
        f.unlink()

    # Train and evaluate models
    results = train_and_save_models()
