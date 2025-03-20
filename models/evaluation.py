"""
Module for evaluating credit card fraud detection models.

This module contains functions to evaluate model performance using
various metrics and visualizations.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score, roc_curve,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
import shap

# Add the root directory to the path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing import load_config

def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Calculates the main evaluation metrics for a model.
    
    Args:
        y_true (array-like): Actual target values.
        y_pred (array-like): Predicted values (classes).
        y_prob (array-like, optional): Predicted probabilities for the positive class.
        
    Returns:
        dict: Dictionary with calculated metrics.
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['recall'] = recall_score(y_true, y_pred)
    metrics['f1'] = f1_score(y_true, y_pred)
    
    # Probability-based metrics (if available)
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        metrics['pr_auc'] = average_precision_score(y_true, y_prob)
        metrics['brier_score'] = brier_score_loss(y_true, y_prob)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    
    # Additional metrics from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    metrics['true_negatives'] = tn
    metrics['false_positives'] = fp
    metrics['false_negatives'] = fn
    metrics['true_positives'] = tp
    
    # False positive rate and true positive rate
    metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['true_positive_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Fraud cost (a custom business metric, just for example)
    # Assuming the average cost of fraud is $100 and the cost of an investigation is $10
    cost_per_fraud = 100
    cost_per_investigation = 10
    fraud_cost = fn * cost_per_fraud  # Undetected frauds
    investigation_cost = (tp + fp) * cost_per_investigation  # All investigated transactions
    total_cost = fraud_cost + investigation_cost
    metrics['fraud_cost'] = fraud_cost
    metrics['investigation_cost'] = investigation_cost
    metrics['total_cost'] = total_cost
    
    # Additional metrics for class imbalance
    metrics['balanced_accuracy'] = (metrics['true_positive_rate'] + (tn / (tn + fp))) / 2 if (tn + fp) > 0 else 0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, normalize=False, figsize=(8, 6)):
    """
    Plots the confusion matrix.
    
    Args:
        y_true (array-like): Actual target values.
        y_pred (array-like): Predicted values (classes).
        normalize (bool): If True, normalizes the matrix.
        figsize (tuple): Figure size.
        
    Returns:
        matplotlib.figure.Figure: Generated figure.
    """
    # Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize the matrix, if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', cbar=False,
                xticklabels=['Not Fraud', 'Fraud'],
                yticklabels=['Not Fraud', 'Fraud'])
    
    # Figure settings
    plt.ylabel('Actual Value')
    plt.xlabel('Predicted Value')
    plt.title('Confusion Matrix')
    
    return fig

def plot_roc_curve(y_true, y_probs, model_names=None, figsize=(10, 8)):
    """
    Plots the ROC curve for one or more models.
    
    Args:
        y_true (array-like): Actual target values.
        y_probs (list): List of arrays with predicted probabilities.
        model_names (list, optional): List of model names.
        figsize (tuple): Figure size.
        
    Returns:
        matplotlib.figure.Figure: Generated figure.
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # If y_probs is a single array, transform it into a list
    if not isinstance(y_probs, list):
        y_probs = [y_probs]
    
    # If model_names is not provided, create generic names
    if model_names is None:
        model_names = [f'Model {i+1}' for i in range(len(y_probs))]
    
    # Plot the ROC curve for each model
    for i, y_prob in enumerate(y_probs):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f'{model_names[i]} (AUC = {roc_auc:.3f})')
    
    # Add the diagonal (random model)
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    
    # Figure settings
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(True)
    
    return fig

def plot_precision_recall_curve(y_true, y_probs, model_names=None, figsize=(10, 8)):
    """
    Plots the Precision-Recall curve for one or more models.
    
    Args:
        y_true (array-like): Actual target values.
        y_probs (list): List of arrays with predicted probabilities.
        model_names (list, optional): List of model names.
        figsize (tuple): Figure size.
        
    Returns:
        matplotlib.figure.Figure: Generated figure.
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # If y_probs is a single array, transform it into a list
    if not isinstance(y_probs, list):
        y_probs = [y_probs]
    
    # If model_names is not provided, create generic names
    if model_names is None:
        model_names = [f'Model {i+1}' for i in range(len(y_probs))]
    
    # Calculate the baseline (positive class prevalence)
    baseline = np.mean(y_true)
    
    # Plot the Precision-Recall curve for each model
    for i, y_prob in enumerate(y_probs):
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, label=f'{model_names[i]} (AP = {pr_auc:.3f})')
    
    # Add the baseline line
    ax.axhline(y=baseline, color='r', linestyle='--', label=f'Baseline (Prevalence = {baseline:.3f})')
    
    # Figure settings
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='best')
    ax.grid(True)
    
    return fig

def plot_calibration_curve(y_true, y_probs, model_names=None, n_bins=10, figsize=(10, 8)):
    """
    Plots the calibration curve for one or more models.
    
    Args:
        y_true (array-like): Actual target values.
        y_probs (list): List of arrays with predicted probabilities.
        model_names (list, optional): List of model names.
        n_bins (int): Number of bins for the calibration curve.
        figsize (tuple): Figure size.
        
    Returns:
        matplotlib.figure.Figure: Generated figure.
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # If y_probs is a single array, transform it into a list
    if not isinstance(y_probs, list):
        y_probs = [y_probs]
    
    # If model_names is not provided, create generic names
    if model_names is None:
        model_names = [f'Model {i+1}' for i in range(len(y_probs))]
    
    # Add the perfect diagonal
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    
    # Plot the calibration curve for each model
    for i, y_prob in enumerate(y_probs):
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        ax.plot(prob_pred, prob_true, 's-', label=f'{model_names[i]}')
    
    # Figure settings
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('Average Predicted Probability')
    ax.set_ylabel('Positive Fraction')
    ax.set_title('Calibration Curve (Reliability Curve)')
    ax.legend(loc='best')
    ax.grid(True)
    
    return fig

def plot_shap_summary(model, X, max_display=20, figsize=(12, 8)):
    """
    Plots a SHAP summary for the model.
    
    Args:
        model: Trained model.
        X (pandas.DataFrame): DataFrame with data for calculating SHAP.
        max_display (int): Maximum number of features to display.
        figsize (tuple): Figure size.
        
    Returns:
        matplotlib.figure.Figure: Generated figure.
    """
    # Create the figure
    plt.figure(figsize=figsize)
    
    # Check the model type and use the appropriate explainer
    model_name = type(model).__name__
    if model_name in ['RandomForestClassifier', 'GradientBoostingClassifier', 'XGBClassifier', 'LGBMClassifier']:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # For tree models with two classes, we take only the SHAP values of the positive class (index 1)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
    else:
        # For other models, we use KernelExplainer (slower)
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X, 100))
        shap_values = explainer.shap_values(shap.sample(X, 100))
        
        # For models with two classes, we take only the SHAP values of the positive class (index 1)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
    
    # Plot the SHAP summary
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    
    fig = plt.gcf()
    return fig

def plot_threshold_metrics(y_true, y_prob, thresholds=None, figsize=(12, 8)):
    """
    Plots the variation of metrics as a function of the classification threshold.
    
    Args:
        y_true (array-like): Actual target values.
        y_prob (array-like): Predicted probabilities for the positive class.
        thresholds (array-like, optional): Thresholds to be evaluated.
        figsize (tuple): Figure size.
        
    Returns:
        matplotlib.figure.Figure: Generated figure.
    """
    # If thresholds are not provided, create a sequence
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    
    # Initialize lists to store metrics
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    false_positive_rate_list = []
    
    # Calculate metrics for each threshold
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # Calculate false positive rate
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Store metrics
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        false_positive_rate_list.append(false_positive_rate)
    
    # Create the figure
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Plot metrics
    ax1.plot(thresholds, accuracy_list, 'b-', label='Accuracy')
    ax1.plot(thresholds, precision_list, 'g-', label='Precision')
    ax1.plot(thresholds, recall_list, 'r-', label='Recall')
    ax1.plot(thresholds, f1_list, 'c-', label='F1-score')
    
    # Figure settings for axis Y1
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Score')
    ax1.set_ylim([0, 1])
    
    # Create second axis Y for false positive rate
    ax2 = ax1.twinx()
    ax2.plot(thresholds, false_positive_rate_list, 'm--', label='False Positive Rate')
    ax2.set_ylabel('False Positive Rate')
    ax2.set_ylim([0, 1])
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    # Add title
    plt.title('Metrics Variation by Threshold')
    plt.grid(True)
    
    return fig

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the performance of a model and generates visualizations.
    
    Args:
        model: Trained model.
        X_test (pandas.DataFrame): Test features.
        y_test (pandas.Series): Test target.
        
    Returns:
        dict: Dictionary with metrics and visualizations.
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_prob)
    
    # Generate visualizations
    figures = {}
    figures['confusion_matrix'] = plot_confusion_matrix(y_test, y_pred)
    figures['roc_curve'] = plot_roc_curve(y_test, y_prob)
    figures['pr_curve'] = plot_precision_recall_curve(y_test, y_prob)
    figures['calibration_curve'] = plot_calibration_curve(y_test, y_prob)
    figures['threshold_metrics'] = plot_threshold_metrics(y_test, y_prob)
    
    # Try to generate SHAP visualization (may fail for some models)
    try:
        figures['shap_summary'] = plot_shap_summary(model, X_test.sample(min(100, len(X_test))))
    except Exception as e:
        print(f"Unable to generate SHAP summary: {e}")
    
    return {"metrics": metrics, "figures": figures}

def compare_models(models, X_test, y_test, model_names=None):
    """
    Compares the performance of multiple models.
    
    Args:
        models (list): List of trained models.
        X_test (pandas.DataFrame): Test features.
        y_test (pandas.Series): Test target.
        model_names (list, optional): List of model names.
        
    Returns:
        dict: Dictionary with comparison results.
    """
    # If model_names are not provided, create generic names
    if model_names is None:
        model_names = [f'Model {i+1}' for i in range(len(models))]
    
    # Dictionaries to store results
    all_metrics = {}
    y_probs = []
    
    # Evaluate each model
    for i, model in enumerate(models):
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_probs.append(y_prob)
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        all_metrics[model_names[i]] = metrics
    
    # Create DataFrame with main metrics
    metrics_df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': [all_metrics[name]['accuracy'] for name in model_names],
        'Precision': [all_metrics[name]['precision'] for name in model_names],
        'Recall': [all_metrics[name]['recall'] for name in model_names],
        'F1-score': [all_metrics[name]['f1'] for name in model_names],
        'ROC AUC': [all_metrics[name]['roc_auc'] for name in model_names],
        'PR AUC': [all_metrics[name]['pr_auc'] for name in model_names],
        'Brier Score': [all_metrics[name]['brier_score'] for name in model_names]
    })
    
    # Sort by ROC AUC descending
    metrics_df = metrics_df.sort_values('ROC AUC', ascending=False).reset_index(drop=True)
    
    # Generate comparative visualizations
    figures = {}
    figures['roc_curves'] = plot_roc_curve(y_test, y_probs, model_names)
    figures['pr_curves'] = plot_precision_recall_curve(y_test, y_probs, model_names)
    figures['calibration_curves'] = plot_calibration_curve(y_test, y_probs, model_names)
    
    return {
        "metrics_df": metrics_df,
        "all_metrics": all_metrics,
        "figures": figures
    }

def generate_classification_report(model, X_test, y_test, model_name="Model", figsize=(8, 6)):
    """
    Generates a detailed classification report.
    
    Args:
        model: Trained model.
        X_test (pandas.DataFrame): Test features.
        y_test (pandas.Series): Test target.
        model_name (str): Model name.
        figsize (tuple): Figure size.
        
    Returns:
        dict: Dictionary with textual report and figure.
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_prob)
    
    # Generate textual classification report
    report_text = classification_report(y_test, y_pred)
    
    # Create the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot the confusion matrix
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Not Fraud', 'Fraud'],
                yticklabels=['Not Fraud', 'Fraud'], ax=ax1)
    ax1.set_ylabel('Actual Value')
    ax1.set_xlabel('Predicted Value')
    ax1.set_title('Confusion Matrix')
    
    # Plot main metrics in bars
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    metrics_values = [metrics[m] for m in metrics_to_plot]
    ax2.bar(metrics_to_plot, metrics_values, color='steelblue')
    ax2.set_ylim([0, 1])
    ax2.set_ylabel('Score')
    ax2.set_title('Main Metrics')
    plt.xticks(rotation=45)
    
    # Add general title
    plt.suptitle(f'Classification Report - {model_name}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return {
        "report_text": report_text,
        "figure": fig
    }

def evaluate_model_on_custom_segments(model, X_test, y_test, segment_col):
    """
    Evaluates the model's performance on different segments of the data.
    
    Args:
        model: Trained model.
        X_test (pandas.DataFrame): Test features.
        y_test (pandas.Series): Test target.
        segment_col (str): Name of the column to be used for segmentation.
        
    Returns:
        pandas.DataFrame: DataFrame with metrics per segment.
    """
    # Create a copy of X_test for manipulation
    X_with_target = X_test.copy()
    X_with_target['target'] = y_test.values
    
    # Make predictions on the complete test set
    X_with_target['prediction'] = model.predict(X_test)
    X_with_target['probability'] = model.predict_proba(X_test)[:, 1]
    
    # List to store results
    results = []
    
    # Evaluate for each segment
    for segment in X_with_target[segment_col].unique():
        # Filter the segment
        segment_data = X_with_target[X_with_target[segment_col] == segment]
        
        # Extract y_true, y_pred, and y_prob for the segment
        y_segment_true = segment_data['target']
        y_segment_pred = segment_data['prediction']
        y_segment_prob = segment_data['probability']
        
        # Calculate metrics for the segment
        metrics = calculate_metrics(y_segment_true, y_segment_pred, y_segment_prob)
        
        # Add segment information and metrics to results
        results.append({
            'Segment': segment,
            'Count': len(segment_data),
            'Fraud_Rate': y_segment_true.mean(),
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1': metrics['f1'],
            'AUC': metrics['roc_auc']
        })
    
    # Convert results to DataFrame
    df_results = pd.DataFrame(results)
    
    # Sort by descending count
    df_results = df_results.sort_values('Count', ascending=False).reset_index(drop=True)
    
    return df_results

if __name__ == "__main__":
    # Example usage
    from data.preprocessing import prepare_data
    from models.training import train_all_models
    
    # Load configuration
    config = load_config()
    
    # Prepare data
    X_train, X_test, y_train, y_test, preprocessors = prepare_data()
    
    # Train models
    models, _ = train_all_models(X_train, y_train, config)
    
    # Evaluate individual model
    print("\nEvaluating models individually...")
    for model_name, model in models.items():
        print(f"\nEvaluating model: {model_name}")
        result = evaluate_model(model, X_test, y_test)
        
        print("Metrics:")
        for metric, value in result['metrics'].items():
            if metric != 'confusion_matrix':
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}:\n{value}")
    
    # Compare models
    print("\nComparing models...")
    model_list = list(models.values())
    model_names = list(models.keys())
    comparison = compare_models(model_list, X_test, y_test, model_names)
    
    print("\nComparison table of metrics:")
    print(comparison['metrics_df'].to_string(index=False)) 
