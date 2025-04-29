import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

def load_data(file_path):
    """Load the water quality dataset."""
    return pd.read_csv(file_path)

def clean_data(df):
    """Clean and preprocess the data."""
    # Drop unnecessary columns
    df = df.drop('Index', axis=1, errors='ignore')
    
    # Handle missing values
    for column in df.columns:
        if df[column].dtype in ['float64', 'int64']:
            df[column].fillna(df[column].median(), inplace=True)
        else:
            df[column].fillna(df[column].mode()[0], inplace=True)
    
    # Convert categorical variables
    categorical_columns = ['Color', 'Source', 'Month', 'Time of Day']
    label_encoders = {}
    
    for column in categorical_columns:
        if column in df.columns:
            label_encoders[column] = LabelEncoder()
            df[column] = label_encoders[column].fit_transform(df[column])
    
    return df, label_encoders

def feature_engineering(df):
    """Create new features from existing ones."""
    # Create interaction features
    df['Temp_Diff'] = df['Air Temperature'] - df['Water Temperature']
    
    # Create ratio features
    df['TDS_Conductivity_Ratio'] = df['Total Dissolved Solids'] / df['Conductivity']
    
    # Create time-based features
    df['Is_Daytime'] = (df['Time of Day'] >= 6) & (df['Time of Day'] <= 18)
    
    return df

def prepare_data(df):
    """Prepare data for modeling."""
    # Separate features and target
    X = df.drop('Target', axis=1)
    y = df['Target']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_columns = X.select_dtypes(include=['float64', 'int64']).columns
    
    X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
    X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])
    
    return X_train, X_test, y_train, y_test, scaler

def main():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Load and preprocess data
    df = load_data('water_quality_dataset_100k_new.csv')
    df, label_encoders = clean_data(df)
    df = feature_engineering(df)
    
    # Prepare data for modeling
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    
    # Save processed data
    X_train.to_csv('data/X_train.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)
    
    print("Data preprocessing completed successfully!")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

if __name__ == "__main__":
    main() 