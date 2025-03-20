"""
Module for feature selection for credit card fraud detection.

This module contains functions to select the best features for models,
using various methods such as correlation analysis, importance, and others.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import (
    SelectKBest, chi2, f_classif, mutual_info_classif, 
    RFECV, SelectFromModel, VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Add root directory to path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing import load_config

def remove_highly_correlated_features(df, threshold=0.95, keep_target=True, target_col=None):
    """
    Remove highly correlated features, keeping only one from each pair.
    
    Args:
        df (pandas.DataFrame): DataFrame with features.
        threshold (float): Correlation threshold to consider features as correlated.
        keep_target (bool): If True, never removes the target column.
        target_col (str): Name of the target column.
        
    Returns:
        pandas.DataFrame: DataFrame with the selected features.
        list: List of removed features.
    """
    df_corr = df.copy()
    
    # Calculate the correlation matrix
    correlation_matrix = df_corr.corr().abs()
    
    # Create a mask for the upper triangle of the correlation matrix
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    
    # Find features to remove
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    
    # Never remove the target column
    if keep_target and target_col in to_drop:
        to_drop.remove(target_col)
    
    print(f"Removing {len(to_drop)} highly correlated features (> {threshold}):")
    for col in to_drop:
        print(f"  - {col}")
    
    # Remove correlated features
    df_reduced = df_corr.drop(columns=to_drop)
    
    return df_reduced, to_drop

def select_features_univariate(X, y, method='f_classif', k=20):
    """
    Selects the best features using univariate methods.
    
    Args:
        X (pandas.DataFrame): Features.
        y (pandas.Series): Target.
        method (str): Selection method ('f_classif', 'chi2', 'mutual_info').
        k (int): Number of features to select.
        
    Returns:
        pandas.DataFrame: DataFrame with the selected features.
        list: Names of the selected features.
    """
    # Choose the selection method
    if method == 'f_classif':
        selector = SelectKBest(score_func=f_classif, k=k)
    elif method == 'chi2':
        # Chi2 requires non-negative features
        selector = SelectKBest(score_func=chi2, k=k)
    elif method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
    else:
        raise ValueError(f"Unrecognized method: {method}")
    
    # Apply the selection
    X_selected = selector.fit_transform(X, y)
    
    # Get the indices of the selected features
    selected_indices = selector.get_support(indices=True)
    
    # Get the names of the selected features
    selected_features = X.columns[selected_indices].tolist()
    
    # Create a new DataFrame with the selected features
    X_new = X.iloc[:, selected_indices]
    
    # Display the scores of the selected features
    if method != 'mutual_info':  # mutual_info doesn't have defined p-values
        scores = pd.DataFrame({
            'Feature': selected_features,
            'Score': selector.scores_[selected_indices],
            'P-value': selector.pvalues_[selected_indices] if hasattr(selector, 'pvalues_') else [None] * len(selected_indices)
        })
    else:
        scores = pd.DataFrame({
            'Feature': selected_features,
            'Score': selector.scores_[selected_indices]
        })
    
    # Sort by descending score
    scores = scores.sort_values('Score', ascending=False)
    
    print(f"Top {k} features selected by the {method} method:")
    for i, (feature, score) in enumerate(zip(scores['Feature'], scores['Score'])):
        print(f"  {i+1}. {feature}: {score:.4f}")
    
    return X_new, selected_features, scores

def select_features_model_based(X, y, model_type='random_forest', threshold='mean'):
    """
    Selects features based on importance assigned by a model.
    
    Args:
        X (pandas.DataFrame): Features.
        y (pandas.Series): Target.
        model_type (str): Model type ('random_forest', 'xgboost', 'logistic').
        threshold (str or float): Threshold for selection ('mean', 'median', or specific value).
        
    Returns:
        pandas.DataFrame: DataFrame with the selected features.
        list: Names of the selected features.
        DataFrame: Feature importance.
    """
    # Create the model
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'xgboost':
        model = XGBClassifier(n_estimators=100, random_state=42)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif model_type == 'logistic':
        model = LogisticRegression(max_iter=1000, random_state=42)
    else:
        raise ValueError(f"Unrecognized model type: {model_type}")
    
    # Create the model-based selector
    selector = SelectFromModel(model, threshold=threshold)
    
    # Fit the selector
    selector.fit(X, y)
    
    # Transform the data
    X_selected = selector.transform(X)
    
    # Get the selected features
    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask].tolist()
    
    # Create a new DataFrame with the selected features
    X_new = X.loc[:, selected_features]
    
    # Get the feature importance
    model.fit(X, y)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:  # For models like LogisticRegression
        importances = np.abs(model.coef_[0])
    
    # Create DataFrame with importances
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Mark the selected features
    feature_importance['Selected'] = feature_importance['Feature'].isin(selected_features)
    
    print(f"Features selected by the {model_type} model (threshold={threshold}):")
    for i, feature in enumerate(feature_importance[feature_importance['Selected']]['Feature']):
        importance = feature_importance[feature_importance['Feature'] == feature]['Importance'].iloc[0]
        print(f"  {i+1}. {feature}: {importance:.4f}")
    
    return X_new, selected_features, feature_importance

def select_features_rfe(X, y, estimator=None, step=1, cv=5, min_features=5):
    """
    Selects features using Recursive Feature Elimination with cross-validation.
    
    Args:
        X (pandas.DataFrame): Features.
        y (pandas.Series): Target.
        estimator: Sklearn estimator. If None, uses RandomForestClassifier.
        step (int): Number of features to eliminate in each iteration.
        cv (int): Number of folds for cross-validation.
        min_features (int): Minimum number of features to consider.
        
    Returns:
        pandas.DataFrame: DataFrame with the selected features.
        list: Names of the selected features.
    """
    if estimator is None:
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Create the RFE selector with cross-validation
    selector = RFECV(
        estimator=estimator,
        step=step,
        cv=cv,
        min_features_to_select=min_features,
        scoring='roc_auc',
        n_jobs=-1
    )
    
    # Fit the selector
    selector.fit(X, y)
    
    # Get the selected features
    selected_features = X.columns[selector.support_].tolist()
    
    # Create a new DataFrame with the selected features
    X_new = X.loc[:, selected_features]
    
    print(f"Features selected by RFE (optimal number: {selector.n_features_}):")
    for i, feature in enumerate(selected_features):
        print(f"  {i+1}. {feature}")
    
    # Visualize the number of features vs. cross-validation score
    plt.figure(figsize=(10, 6))
    plt.plot(range(min_features, len(selector.grid_scores_) + min_features), selector.grid_scores_)
    plt.xlabel('Number of features')
    plt.ylabel('Cross-validation score')
    plt.title('Number of features vs. Cross-validation score')
    plt.grid(True)
    
    return X_new, selected_features, selector

def apply_pca(X, n_components=None, variance_threshold=0.95):
    """
    Applies PCA for dimensionality reduction.
    
    Args:
        X (pandas.DataFrame): Features.
        n_components (int, optional): Number of components to keep. If None, 
                                     uses the variance threshold.
        variance_threshold (float): Proportion of variance to be retained.
        
    Returns:
        pandas.DataFrame: DataFrame with the principal components.
        object: Fitted PCA object.
    """
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine the number of components
    if n_components is None:
        # Use the maximum possible initially
        pca_all = PCA()
        pca_all.fit(X_scaled)
        
        # Calculate the cumulative explained variance
        cumulative_variance = np.cumsum(pca_all.explained_variance_ratio_)
        
        # Find the number of components that explain the desired variance
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        
        print(f"Number of components selected to explain {variance_threshold*100:.1f}% of the variance: {n_components}")
    
    # Apply PCA with the appropriate number of components
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create a DataFrame with the principal components
    column_names = [f'PC{i+1}' for i in range(n_components)]
    df_pca = pd.DataFrame(X_pca, columns=column_names, index=X.index)
    
    # Show the variance explained by each component
    explained_variance = pd.DataFrame({
        'Component': column_names,
        'Explained Variance': pca.explained_variance_ratio_,
        'Cumulative Variance': np.cumsum(pca.explained_variance_ratio_)
    })
    
    print("Variance explained by each principal component:")
    for i, (component, variance) in enumerate(zip(explained_variance['Component'], explained_variance['Explained Variance'])):
        print(f"  {component}: {variance:.4f} ({explained_variance['Cumulative Variance'].iloc[i]:.4f} cumulative)")
    
    return df_pca, pca, explained_variance

def select_best_features(df, target_col, config=None):
    """
    Complete pipeline for selecting the best features.
    
    Args:
        df (pandas.DataFrame): DataFrame with features and target.
        target_col (str): Name of the target column.
        config (dict, optional): Project configuration.
        
    Returns:
        pandas.DataFrame: DataFrame with the selected features and the target.
        list: List of selected features.
    """
    if config is None:
        config = load_config()
    
    # Separate features and target
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    print(f"\n1. Removing features with low variance...")
    # Remove features with near-zero variance
    var_selector = VarianceThreshold(threshold=0.01)
    X_var = pd.DataFrame(var_selector.fit_transform(X), 
                         columns=X.columns[var_selector.get_support()],
                         index=X.index)
    
    removed_features = [col for col in X.columns if col not in X_var.columns]
    print(f"   Removed {len(removed_features)} features with near-zero variance")
    if removed_features:
        print("   Removed features:")
        for feat in removed_features:
            print(f"     - {feat}")
    
    print(f"\n2. Removing highly correlated features...")
    # Remove highly correlated features
    X_uncorr, dropped_corr = remove_highly_correlated_features(X_var, threshold=0.95)
    
    print(f"\n3. Model-based selection...")
    # Selection based on feature importance
    X_model, model_features, importances = select_features_model_based(
        X_uncorr, y, model_type='random_forest', threshold='mean'
    )
    
    print(f"\n4. Additional verification with univariate selection...")
    # Verify with univariate selection
    k = min(20, X_model.shape[1])
    X_univariate, univariate_features, univariate_scores = select_features_univariate(
        X_model, y, method='f_classif', k=k
    )
    
    # Combine the results to get the final set of features
    final_features = list(set(model_features) & set(univariate_features))
    
    print(f"\nFinal set: {len(final_features)} selected features")
    for feat in sorted(final_features):
        importance = importances[importances['Feature'] == feat]['Importance'].iloc[0]
        score = univariate_scores[univariate_scores['Feature'] == feat]['Score'].iloc[0]
        print(f"  - {feat}: ModelImportance={importance:.4f}, UnivariateScore={score:.4f}")
    
    # Create the final DataFrame with the selected features
    X_final = X[final_features]
    df_final = pd.concat([X_final, y], axis=1)
    
    return df_final, final_features

if __name__ == "__main__":
    # Example of use
    from data.preprocessing import load_data, prepare_data
    from features.creation import create_all_features
    
    # Load and prepare the data
    config = load_config()
    df_raw = load_data(config)
    
    # Create features
    df_features = create_all_features(df_raw)
    
    # Select the best features
    target_col = config['preprocessing']['target_column']
    df_selected, selected_features = select_best_features(df_features, target_col, config)
    
    print(f"\nOriginal dimensions: {df_features.shape}")
    print(f"Dimensions after selection: {df_selected.shape}")
    print(f"Reduction of {df_features.shape[1] - df_selected.shape[1]} features ({((df_features.shape[1] - df_selected.shape[1]) / df_features.shape[1] * 100):.1f}%)") 
