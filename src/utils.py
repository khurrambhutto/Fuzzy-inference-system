"""
Utility Functions Module

Data loading, preprocessing, and evaluation functions.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def load_data(filepath: str = None) -> pd.DataFrame:
    """
    Load the heart disease dataset.
    
    Args:
        filepath: Path to CSV file (default: data/heart.csv)
        
    Returns:
        DataFrame with heart disease data
    """
    if filepath is None:
        # Try common locations
        possible_paths = [
            Path(__file__).parent.parent / 'data' / 'heart.csv',
            Path('data/heart.csv'),
            Path('heart.csv')
        ]
        for path in possible_paths:
            if path.exists():
                filepath = path
                break
    
    df = pd.read_csv(filepath)
    return df


def preprocess_data(df: pd.DataFrame) -> tuple:
    """
    Preprocess data for fuzzy system.
    
    Selects the 5 input features used by the FIS.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Tuple of (X, y) where X has columns [age, trestbps, chol, thalach, oldpeak]
    """
    # Select features used by fuzzy system
    feature_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    target_col = 'target'
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, 
               random_state: int = 42) -> tuple:
    """
    Split data into train and test sets.
    
    Args:
        X: Features
        y: Target
        test_size: Proportion for test set
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Evaluate model performance.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dict with accuracy, precision, recall, f1, confusion_matrix
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }


def evaluate_with_threshold(y_true: np.ndarray, risk_scores: np.ndarray, 
                           threshold: float = 0.5) -> dict:
    """
    Evaluate with a specific threshold.
    
    Args:
        y_true: True labels
        risk_scores: Predicted risk scores (0-1)
        threshold: Classification threshold
        
    Returns:
        Dict with metrics
        
    Note: Inverted logic - low risk score = disease (based on dataset patterns)
    """
    y_pred = (np.array(risk_scores) < threshold).astype(int)  # Inverted
    return evaluate_model(y_true, y_pred)


def find_best_threshold(y_true: np.ndarray, risk_scores: np.ndarray) -> tuple:
    """
    Find optimal classification threshold.
    
    Args:
        y_true: True labels
        risk_scores: Predicted risk scores
        
    Returns:
        Tuple of (best_threshold, best_f1)
        
    Note: Inverted logic - low risk score = disease
    """
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in np.arange(0.1, 0.9, 0.05):
        y_pred = (np.array(risk_scores) < threshold).astype(int)  # Inverted
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1


def crisp_baseline(X: pd.DataFrame) -> np.ndarray:
    """
    Simple crisp rule baseline for comparison.
    
    Rules:
        - If age > 55 AND (chol > 240 OR trestbps > 140) -> disease
        - If oldpeak > 2 -> disease
        - Otherwise -> no disease
    
    Args:
        X: Feature DataFrame
        
    Returns:
        Array of predictions (0 or 1)
    """
    predictions = []
    
    for _, row in X.iterrows():
        if row['oldpeak'] > 2:
            predictions.append(1)
        elif row['age'] > 55 and (row['chol'] > 240 or row['trestbps'] > 140):
            predictions.append(1)
        else:
            predictions.append(0)
    
    return np.array(predictions)


def calculate_mae(y_true: np.ndarray, risk_scores: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(np.array(y_true) - np.array(risk_scores)))

