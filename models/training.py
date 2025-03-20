"""
Module for training credit card fraud detection models.

This module contains functions to train, tune and save different models.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import time
import json
from datetime import datetime
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)

# Add root directory to path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing import load_config

def create_model(model_type, params=None):
    """
    Creates a classification model based on the specified type.
    
    Args:
        model_type (str): Model type ('logistic', 'random_forest', 'xgboost', 'lightgbm', 'gradient_boosting').
        params (dict, optional): Specific parameters for the model.
        
    Returns:
        object: Instantiated classification model.
    """
    # If params is not provided, use an empty dictionary
    if params is None:
        params = {}
    
    # Create the model based on the type
    if model_type == 'logistic':
        model = LogisticRegression(random_state=42, **params)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42, **params)
    elif model_type == 'xgboost':
        model = XGBClassifier(random_state=42, **params)
    elif model_type == 'lightgbm':
        model = LGBMClassifier(random_state=42, **params)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(random_state=42, **params)
    else:
        raise ValueError(f"Unrecognized model type: {model_type}")
    
    return model

def train_model(X_train, y_train, model_type, params=None):
    """
    Trains a classification model.
    
    Args:
        X_train (pandas.DataFrame): Training features.
        y_train (pandas.Series): Training target.
        model_type (str): Model type.
        params (dict, optional): Model parameters.
        
    Returns:
        object: Trained model.
        float: Training time (in seconds).
    """
    # Create the model
    model = create_model(model_type, params)
    
    # Measure training time
    start_time = time.time()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Calculate training time
    train_time = time.time() - start_time
    
    return model, train_time

def tune_hyperparameters(X_train, y_train, model_type, param_grid, cv=5, scoring='roc_auc', n_iter=10):
    """
    Performs hyperparameter optimization using cross-validation.
    
    Args:
        X_train (pandas.DataFrame): Training features.
        y_train (pandas.Series): Training target.
        model_type (str): Model type.
        param_grid (dict): Parameter grid for search.
        cv (int): Number of folds for cross-validation.
        scoring (str): Metric for optimization.
        n_iter (int): Number of combinations to test (for RandomizedSearchCV).
        
    Returns:
        object: Best model found.
        dict: Best parameters.
        float: Best score.
    """
    # Create the base model
    model = create_model(model_type)
    
    # Decide between exhaustive or randomized search
    if len(param_grid) <= 5:  # If there are few parameters, use GridSearchCV
        print(f"Performing GridSearchCV for {model_type}...")
        search = GridSearchCV(
            model, param_grid, cv=cv, scoring=scoring,
            n_jobs=-1, verbose=1, return_train_score=True
        )
    else:  # Otherwise, use RandomizedSearchCV
        print(f"Performing RandomizedSearchCV for {model_type}...")
        search = RandomizedSearchCV(
            model, param_grid, n_iter=n_iter, cv=cv, scoring=scoring,
            n_jobs=-1, verbose=1, random_state=42, return_train_score=True
        )
    
    # Run the search
    search.fit(X_train, y_train)
    
    # Get the best model, parameters and score
    best_model = search.best_estimator_
    best_params = search.best_params_
    best_score = search.best_score_
    
    print(f"Best parameters for {model_type}:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"Best score ({scoring}): {best_score:.4f}")
    
    return best_model, best_params, best_score

def create_ensemble_model(models_dict, ensemble_type='voting', voting='soft', weights=None):
    """
    Creates an ensemble model from multiple base models.
    
    Args:
        models_dict (dict): Dictionary of {name: model} for base models.
        ensemble_type (str): Ensemble type ('voting' or 'stacking').
        voting (str): Voting type for VotingClassifier ('hard' or 'soft').
        weights (list, optional): Weights for each model in the voting ensemble.
        
    Returns:
        object: Ensemble model.
    """
    # Create a list of tuples (name, model) for the ensemble
    estimators = [(name, model) for name, model in models_dict.items()]
    
    if ensemble_type == 'voting':
        # Create a voting ensemble model
        ensemble_model = VotingClassifier(
            estimators=estimators,
            voting=voting,
            weights=weights
        )
    elif ensemble_type == 'stacking':
        # Create a stacking ensemble model
        ensemble_model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(),
            cv=5
        )
    else:
        raise ValueError(f"Unrecognized ensemble type: {ensemble_type}")
    
    return ensemble_model

def save_model(model, model_name, config=None):
    """
    Saves the trained model to disk.
    
    Args:
        model (object): Trained model.
        model_name (str): Model name.
        config (dict, optional): Project configurations.
        
    Returns:
        str: File path where the model was saved.
    """
    if config is None:
        config = load_config()
    
    # Create the models directory, if it doesn't exist
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), config['paths']['models_dir'])
    os.makedirs(models_dir, exist_ok=True)
    
    # Add timestamp to filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(models_dir, f"{model_name}_{timestamp}.pkl")
    
    # Save the model
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model '{model_name}' saved at: {file_path}")
    
    return file_path

def train_all_models(X_train, y_train, config=None):
    """
    Trains all models defined in the configuration.
    
    Args:
        X_train (pandas.DataFrame): Training features.
        y_train (pandas.Series): Training target.
        config (dict, optional): Project configurations.
        
    Returns:
        dict: Dictionary with trained models.
        dict: Dictionary with training times.
    """
    if config is None:
        config = load_config()
    
    models_config = config['models']
    
    # Dictionaries to store models and training times
    trained_models = {}
    training_times = {}
    
    # Train each model defined in the configuration
    for model_type, params in models_config.items():
        print(f"\nTraining model: {model_type}")
        model, train_time = train_model(X_train, y_train, model_type, params)
        
        trained_models[model_type] = model
        training_times[model_type] = train_time
        
        print(f"Training completed in {train_time:.2f} seconds")
    
    # Create and train an ensemble model
    print("\nCreating ensemble model...")
    ensemble_model = create_ensemble_model(trained_models, ensemble_type='voting', voting='soft')
    
    start_time = time.time()
    ensemble_model.fit(X_train, y_train)
    ensemble_time = time.time() - start_time
    
    trained_models['ensemble'] = ensemble_model
    training_times['ensemble'] = ensemble_time
    
    print(f"Ensemble training completed in {ensemble_time:.2f} seconds")
    
    return trained_models, training_times

def default_param_grids():
    """
    Returns default parameter grids for hyperparameter optimization.
    
    Returns:
        dict: Dictionary with parameter grids for each model type.
    """
    param_grids = {
        'logistic': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'class_weight': [None, 'balanced'],
            'solver': ['liblinear', 'saga'],
            'penalty': ['l1', 'l2']
        },
        'random_forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': [None, 'balanced']
        },
        'xgboost': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'scale_pos_weight': [1, 5, 10]
        },
        'lightgbm': {
            'n_estimators': [50, 100, 200],
            'num_leaves': [31, 50, 100],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'class_weight': [None, 'balanced']
        },
        'gradient_boosting': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'min_samples_split': [2, 5, 10]
        }
    }
    
    return param_grids

def tune_all_models(X_train, y_train, param_grids=None, cv=5, scoring='roc_auc'):
    """
    Performs hyperparameter optimization for all models.
    
    Args:
        X_train (pandas.DataFrame): Training features.
        y_train (pandas.Series): Training target.
        param_grids (dict, optional): Parameter grids. If None, uses default ones.
        cv (int): Number of folds for cross-validation.
        scoring (str): Metric for optimization.
        
    Returns:
        dict: Dictionary with best models.
        dict: Dictionary with best parameters.
        dict: Dictionary with best scores.
    """
    if param_grids is None:
        param_grids = default_param_grids()
    
    # Dictionaries to store results
    best_models = {}
    best_params_dict = {}
    best_scores = {}
    
    # Optimize each model type
    for model_type, param_grid in param_grids.items():
        print(f"\nOptimizing model: {model_type}")
        
        best_model, best_params, best_score = tune_hyperparameters(
            X_train, y_train, model_type, param_grid, cv=cv, scoring=scoring
        )
        
        best_models[model_type] = best_model
        best_params_dict[model_type] = best_params
        best_scores[model_type] = best_score
    
    # Save best parameters to a JSON file
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    params_file = os.path.join(models_dir, f"best_params_{timestamp}.json")
    
    with open(params_file, 'w') as f:
        json.dump(best_params_dict, f, indent=4)
    
    print(f"\nBest parameters saved at: {params_file}")
    
    # Create and train an ensemble model with the best models
    print("\nCreating ensemble model with best models...")
    best_ensemble = create_ensemble_model(best_models, ensemble_type='voting', voting='soft')
    best_ensemble.fit(X_train, y_train)
    
    best_models['ensemble'] = best_ensemble
    
    return best_models, best_params_dict, best_scores

if __name__ == "__main__":
    # Example usage
    from data.preprocessing import prepare_data
    from features.creation import create_all_features
    from features.selection import select_best_features
    
    # Load configuration
    config = load_config()
    
    # Prepare data
    X_train, X_test, y_train, y_test, preprocessors = prepare_data()
    
    print("\n1. Training models with default parameters...")
    # Train models with default parameters
    trained_models, training_times = train_all_models(X_train, y_train, config)
    
    print("\n2. Optimizing hyperparameters...")
    # Optimize models (commented out because it can take a long time)
    # best_models, best_params, best_scores = tune_all_models(X_train, y_train)
    
    # Save models
    print("\n3. Saving models...")
    for model_name, model in trained_models.items():
        save_model(model, model_name, config) 
