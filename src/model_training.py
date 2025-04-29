import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import optuna
import joblib
import os
import warnings
import matplotlib.pyplot as plt
from IPython.display import display
warnings.filterwarnings('ignore')


def load_data():
    """Load preprocessed data."""
    X_train = pd.read_csv('../data/X_train.csv')
    X_test = pd.read_csv('../data/X_test.csv')
    y_train = pd.read_csv('../data/y_train.csv').squeeze()
    y_test = pd.read_csv('../data/y_test.csv').squeeze()
    return X_train, X_test, y_train, y_test


def evaluate_model(y_true, y_pred, model_name):
    """Comprehensive model evaluation."""
    print(f"\n{model_name} Results:")
    print(classification_report(y_true, y_pred))

    # Calculate additional metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def cross_validate_model(model, X, y, model_name, cv=5):
    """Perform cross-validation and return mean scores."""
    cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

    print(f"\n{model_name} Cross-Validation Results:")
    print(f"Mean Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

    return {
        'cv_mean_accuracy': scores.mean(),
        'cv_std_accuracy': scores.std()
    }


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train and evaluate Random Forest model."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = evaluate_model(y_test, y_pred, "Random Forest")
    cv_metrics = cross_validate_model(model, X_train, y_train, "Random Forest")
    metrics.update(cv_metrics)

    return model, metrics


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train and evaluate XGBoost model."""
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = evaluate_model(y_test, y_pred, "XGBoost")
    cv_metrics = cross_validate_model(model, X_train, y_train, "XGBoost")
    metrics.update(cv_metrics)

    return model, metrics


def train_lightgbm(X_train, y_train, X_test, y_test):
    """Train and evaluate LightGBM model."""
    model = LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = evaluate_model(y_test, y_pred, "LightGBM")
    cv_metrics = cross_validate_model(model, X_train, y_train, "LightGBM")
    metrics.update(cv_metrics)

    return model, metrics


def train_catboost(X_train, y_train, X_test, y_test):
    """Train and evaluate CatBoost model."""
    model = CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        random_seed=42,
        verbose=False,
        thread_count=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = evaluate_model(y_test, y_pred, "CatBoost")
    cv_metrics = cross_validate_model(model, X_train, y_train, "CatBoost")
    metrics.update(cv_metrics)

    return model, metrics


def train_mlp(X_train, y_train, X_test, y_test):
    """Train and evaluate MLP model with enhanced configuration."""
    model = MLPClassifier(
        hidden_layer_sizes=(200, 100, 50),  # Deeper architecture
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42
    )

    # Train the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = evaluate_model(y_test, y_pred, "MLP")
    cv_metrics = cross_validate_model(model, X_train, y_train, "MLP")
    metrics.update(cv_metrics)

    return model, metrics


def optimize_hyperparameters(X_train, y_train, X_test, y_test):
    """Optimize hyperparameters using Optuna."""
    def objective(trial):
        model = XGBClassifier(
            n_estimators=trial.suggest_int('n_estimators', 50, 200),
            max_depth=trial.suggest_int('max_depth', 3, 10),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
            random_state=42
        )
        model.fit(X_train, y_train)
        return accuracy_score(y_test, model.predict(X_test))

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    print("\nBest hyperparameters:")
    print(study.best_params)

    return study.best_params


def save_models(models):
    """Save trained models."""
    os.makedirs('models', exist_ok=True)
    for name, (model, metrics) in models.items():
        joblib.dump(model, f'models/{name}.joblib')
        # Save metrics
        pd.DataFrame([metrics]).to_csv(
            f'models/{name}_metrics.csv', index=False)


def compare_models(models_metrics):
    """Compare performance of all models."""
    metrics_df = pd.DataFrame(models_metrics).T
    print("\nModel Comparison:")
    display(metrics_df)

    # Plot comparison
    metrics_df.plot(kind='bar', figsize=(12, 6))
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('models/model_comparison.png')
    plt.close()


def main():
    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Train models
    models = {
        'random_forest': train_random_forest(X_train, y_train, X_test, y_test),
        'xgboost': train_xgboost(X_train, y_train, X_test, y_test),
        'lightgbm': train_lightgbm(X_train, y_train, X_test, y_test),
        'catboost': train_catboost(X_train, y_train, X_test, y_test),
        'mlp': train_mlp(X_train, y_train, X_test, y_test)
    }

    # Extract metrics for comparison
    models_metrics = {name: metrics for name, (_, metrics) in models.items()}

    # Compare models
    compare_models(models_metrics)

    # Optimize hyperparameters
    best_params = optimize_hyperparameters(X_train, y_train, X_test, y_test)

    # Save models and metrics
    save_models(models)

    print("\nModel training completed successfully!")


if __name__ == "__main__":
    main()
