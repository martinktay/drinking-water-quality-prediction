"""
Utility functions and imports for the Water Quality Classification project.
This module contains all necessary imports and utility functions for data analysis,
preprocessing, and model training.
"""

# Data manipulation and analysis
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# Progress tracking
from tqdm import tqdm

# Data preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Model selection and evaluation
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)
from sklearn import metrics
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    roc_curve,
    auc,
)

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Set global visualization parameters
plt.style.use('seaborn')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def setup_environment():
    """
    Set up the environment with necessary configurations.
    This includes setting random seeds, display options, and other configurations.
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Set pandas display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:.4f}'.format)

    # Suppress warnings
    import warnings
    warnings.filterwarnings('ignore')
