"""
Module for creating data processing pipelines.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Add root directory to path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts features from date/time columns."""
    
    def __init__(self, datetime_col='Transaction DateTime'):
        self.datetime_col = datetime_col
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        if self.datetime_col in X_copy.columns:
            # Make sure the column is in datetime format
            X_copy[self.datetime_col] = pd.to_datetime(X_copy[self.datetime_col], errors='coerce')
            
            # Extract date/time features
            X_copy['hour'] = X_copy[self.datetime_col].dt.hour
            X_copy['day'] = X_copy[self.datetime_col].dt.day
            X_copy['month'] = X_copy[self.datetime_col].dt.month
            X_copy['day_of_week'] = X_copy[self.datetime_col].dt.dayofweek
            X_copy['is_weekend'] = X_copy['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
            
            # Define periods of the day
            X_copy['period_of_day'] = X_copy['hour'].apply(
                lambda h: 'early_morning' if 0 <= h < 6 else 
                          'morning' if 6 <= h < 12 else 
                          'afternoon' if 12 <= h < 18 else 'night'
            )
        
        return X_copy

class CustomFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts custom features for fraud detection."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        # Create amount features
        if 'Amount' in X_copy.columns:
            # Amount categories
            X_copy['amount_category'] = pd.cut(
                X_copy['Amount'], 
                bins=[0, 50, 100, 500, 1000, float('inf')], 
                labels=['very_low', 'low', 'medium', 'high', 'very_high']
            )
            
            # Extreme values/outliers (Z-score)
            mean_amount = X_copy['Amount'].mean()
            std_amount = X_copy['Amount'].std()
            X_copy['amount_zscore'] = (X_copy['Amount'] - mean_amount) / std_amount
            X_copy['is_amount_outlier'] = X_copy['amount_zscore'].apply(lambda z: 1 if abs(z) > 3 else 0)
        
        # Create risk indicators
        if all(col in X_copy.columns for col in ['is_weekend', 'period_of_day']):
            # Transactions in early morning on weekends may be more suspicious
            X_copy['high_risk_time'] = ((X_copy['is_weekend'] == 1) & 
                                       (X_copy['period_of_day'] == 'early_morning')).astype(int)
        
        return X_copy

def create_preprocessing_pipeline(categorical_features, numerical_features, datetime_col=None):
    """
    Creates a complete preprocessing pipeline.
    
    Args:
        categorical_features (list): List of categorical columns.
        numerical_features (list): List of numerical columns.
        datetime_col (str, optional): Name of the date/time column, if it exists.
        
    Returns:
        sklearn.pipeline.Pipeline: Preprocessing pipeline.
    """
    transformers = []
    
    # Processing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    transformers.append(('cat', categorical_transformer, categorical_features))
    
    # Processing for numerical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    transformers.append(('num', numerical_transformer, numerical_features))
    
    # Initial data preparation
    preprocessor = ColumnTransformer(transformers, remainder='passthrough')
    
    # Complete pipeline
    pipeline_steps = []
    
    # Add date/time feature extraction, if applicable
    if datetime_col:
        pipeline_steps.append(('datetime_features', DateTimeFeatureExtractor(datetime_col=datetime_col)))
    
    pipeline_steps.extend([
        ('preprocessor', preprocessor),
        ('custom_features', CustomFeatureExtractor())
    ])
    
    return Pipeline(pipeline_steps)

def create_model_pipeline(model, preprocessing_pipeline, use_smote=True, random_state=42):
    """
    Creates a complete pipeline including preprocessing and model.
    
    Args:
        model: Machine learning model.
        preprocessing_pipeline: Preprocessing pipeline.
        use_smote (bool): Whether to use SMOTE to balance classes.
        random_state (int): Seed for reproducibility.
        
    Returns:
        imblearn.pipeline.Pipeline: Complete processing and model pipeline.
    """
    pipeline_steps = [('preprocessing', preprocessing_pipeline)]
    
    if use_smote:
        pipeline_steps.append(('smote', SMOTE(random_state=random_state)))
    
    pipeline_steps.append(('model', model))
    
    return ImbPipeline(pipeline_steps) 
