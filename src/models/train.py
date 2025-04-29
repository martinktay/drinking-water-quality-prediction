"""Model training and evaluation module."""

import logging
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
import json
from pathlib import Path
import yaml
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)


def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml."""
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise


def create_preprocessor(features: list) -> ColumnTransformer:
    """Create the feature preprocessor."""
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), features)
        ])


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, name: str) -> Dict[str, float]:
    """Evaluate model performance."""
    try:
        y_pred = model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }

        # Save detailed classification report
        report = classification_report(y_test, y_pred)
        report_path = Path('reports/model_performance')
        report_path.mkdir(parents=True, exist_ok=True)
        with open(report_path / f'{name}_report.txt', 'w') as f:
            f.write(report)

        return metrics
    except Exception as e:
        logger.error(f"Error evaluating model {name}: {str(e)}")
        raise


def save_model(model: Any, name: str, config: Dict[str, Any]) -> None:
    """Save model to disk."""
    try:
        model_path = Path(config['models'][name]['path'])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        logger.info(f"Saved model {name} to {model_path}")
    except Exception as e:
        logger.error(f"Error saving model {name}: {str(e)}")
        raise


def train_model(
    name: str,
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    config: Dict[str, Any]
) -> Dict[str, float]:
    """Train and evaluate a single model."""
    try:
        logger.info(f"Training {name}...")
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('clf', model)
        ])

        pipeline.fit(X_train, y_train)
        logger.info(f"{name} training completed")

        # Evaluate model
        logger.info(f"Evaluating {name}...")
        metrics = evaluate_model(pipeline, X_test, y_test, name)
        logger.info(f"{name} metrics: {metrics}")

        # Save model
        save_model(pipeline, name, config)

        return metrics
    except Exception as e:
        logger.error(f"Error in model training pipeline for {name}: {str(e)}")
        raise


def train_and_save_models(sample_size: Optional[int] = None) -> Dict[str, Dict[str, float]]:
    """Train and evaluate all models."""
    try:
        config = load_config()

        # Load and preprocess data
        logger.info("Loading data...")
        df = pd.read_csv('data/processed/processed_data.csv')
        if sample_size:
            df = df.sample(n=sample_size, random_state=42)

        features = config['preprocessor']['features']
        X = df[features]
        y = df['Target']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Create preprocessor
        preprocessor = create_preprocessor(features)

        # Initialize models
        models = {
            'RandomForest': RandomForestClassifier(random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42),
            'LightGBM': lgb.LGBMClassifier(random_state=42),
            'DecisionTree': DecisionTreeClassifier(random_state=42),
            'LinearSVC': LinearSVC(random_state=42, max_iter=2000)
        }

        # Train all models
        results = {}
        for name, model in models.items():
            results[name] = train_model(
                name, model, X_train, y_train, X_test, y_test,
                preprocessor, config
            )

        # Save preprocessor
        preprocessor_path = Path(config['preprocessor']['path'])
        preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, preprocessor_path)

        # Save results
        results_path = Path('reports/model_performance/model_metrics.json')
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)

        return results
    except Exception as e:
        logger.error(f"Error in model training pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    from src.utils.logging import setup_logging
    setup_logging(log_file='logs/training.log')
    train_and_save_models(sample_size=100000)
