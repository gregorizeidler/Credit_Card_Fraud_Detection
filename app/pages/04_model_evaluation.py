"""
Model Evaluation Page for the fraud detection application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import pickle
from sklearn.metrics import confusion_matrix

# Add root directory to path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.preprocessing import load_config, load_data

# Page configuration
st.set_page_config(
    page_title="Model Evaluation - Fraud Detection", 
    page_icon="📊",
    layout="wide"
)

# Page title
st.title("Model Evaluation")
st.markdown("""
    This page allows you to evaluate the performance of trained models for credit card
    fraud detection through various metrics and visualizations.
""")

# Model selection
st.sidebar.header("Settings")
model_options = ["Random Forest", "XGBoost", "Logistic Regression", "LightGBM", "Ensemble"]
selected_model = st.sidebar.selectbox("Select a model to evaluate:", model_options)

# Threshold settings
threshold = st.sidebar.slider(
    "Classification Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.5,
    step=0.05,
    help="Threshold for classifying a transaction as fraud"
)

# Load data (simulated)
# In a real implementation, we would load existing prediction results
def load_evaluation_data():
    # Simulation of evaluation data
    np.random.seed(42)  # For reproducibility
    
    n_samples = 1000
    y_true = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])  # 5% fraud
    
    # Simulated probabilities (closer to the real class but with some errors)
    y_prob = np.zeros(n_samples)
    
    # For non-fraud (y_true = 0)
    mask_non_fraud = (y_true == 0)
    y_prob[mask_non_fraud] = np.random.beta(1, 5, size=mask_non_fraud.sum())
    
    # For fraud (y_true = 1)
    mask_fraud = (y_true == 1)
    y_prob[mask_fraud] = np.random.beta(5, 1, size=mask_fraud.sum())
    
    # Create DataFrame
    eval_df = pd.DataFrame({
        'true_label': y_true,
        'fraud_probability': y_prob,
        'transaction_amount': np.random.lognormal(5, 1, size=n_samples),  # Transaction values
        'transaction_time': pd.date_range(start='2023-01-01', periods=n_samples, freq='10T'),
        'merchant_category': np.random.choice(['retail', 'restaurant', 'travel', 'online', 'other'], size=n_samples),
        'prediction': (y_prob >= threshold).astype(int)
    })
    
    return eval_df

# Load data
eval_df = load_evaluation_data()

# Main interface
st.header(f"Model Evaluation: {selected_model}")

# Tabs for different visualizations
tabs = st.tabs([
    "General Metrics", 
    "Confusion Matrix", 
    "ROC/PR Curves", 
    "Calibration", 
    "Threshold Analysis"
])

with tabs[0]:
    st.subheader("Performance Metrics")
    
    # Calculate metrics with the selected threshold
    eval_df['prediction'] = (eval_df['fraud_probability'] >= threshold).astype(int)
    
    # Calculate metrics
    tn, fp, fn, tp = confusion_matrix(eval_df['true_label'], eval_df['prediction']).ravel()
    total = tn + fp + fn + tp
    
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # ROC AUC (simulated)
    roc_auc = 0.92  # Simulated value
    
    # Display metrics in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric("Accuracy", f"{accuracy:.2%}")
    col2.metric("Precision", f"{precision:.2%}")
    col3.metric("Recall", f"{recall:.2%}")
    col4.metric("F1-Score", f"{f1:.2%}")
    col5.metric("ROC AUC", f"{roc_auc:.4f}")
    
    # Additional metrics
    st.subheader("Detailed Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Detailed metrics table
        detailed_metrics = pd.DataFrame({
            "Metric": [
                "True Positives (TP)", 
                "False Positives (FP)", 
                "True Negatives (TN)", 
                "False Negatives (FN)",
                "Specificity",
                "False Positive Rate",
                "False Negative Rate"
            ],
            "Value": [
                tp,
                fp,
                tn,
                fn,
                tn / (tn + fp) if (tn + fp) > 0 else 0,
                fp / (fp + tn) if (fp + tn) > 0 else 0,
                fn / (fn + tp) if (fn + tp) > 0 else 0
            ],
            "Description": [
                "Correctly detected fraud",
                "Normal transactions classified as fraud",
                "Correctly classified normal transactions",
                "Undetected fraud",
                "Proportion of normal transactions correctly identified",
                "Proportion of false alarms",
                "Proportion of missed fraud"
            ]
        })
        
        st.dataframe(detailed_metrics)
    
    with col2:
        # Bar chart for correct vs. incorrect
        results_data = pd.DataFrame({
            'Category': ['Correct', 'Incorrect'],
            'Count': [tp + tn, fp + fn]
        })
        
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(
            results_data['Category'], 
            results_data['Count'],
            color=['#4CAF50', '#F44336']
        )
        
        ax.set_title('Correct vs. Incorrect')
        ax.set_ylabel('Number of Transactions')
        
        # Add labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        ax.set_ylim(0, total * 1.1)  # Give space for labels
        
        st.pyplot(fig)

with tabs[1]:
    st.subheader("Confusion Matrix")
    
    # Calculate confusion matrix
    cm = confusion_matrix(eval_df['true_label'], eval_df['prediction'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Absolute confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Not Fraud', 'Fraud'],
                    yticklabels=['Not Fraud', 'Fraud'])
        ax.set_ylabel('True Value')
        ax.set_xlabel('Predicted Value')
        ax.set_title('Confusion Matrix (absolute values)')
        st.pyplot(fig)
    
    with col2:
        # Normalized confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', cbar=False,
                    xticklabels=['Not Fraud', 'Fraud'],
                    yticklabels=['Not Fraud', 'Fraud'])
        ax.set_ylabel('True Value')
        ax.set_xlabel('Predicted Value')
        ax.set_title('Confusion Matrix (normalized values)')
        st.pyplot(fig)
    
    # Error analysis
    st.subheader("Error Analysis")
    
    # 1. False Positives (normal transactions classified as fraud)
    fp_data = eval_df[(eval_df['true_label'] == 0) & (eval_df['prediction'] == 1)]
    
    # 2. False Negatives (undetected fraud)
    fn_data = eval_df[(eval_df['true_label'] == 1) & (eval_df['prediction'] == 0)]
    
    # Display samples of each error type
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### False Positives (Sample)")
        st.markdown("Normal transactions incorrectly classified as fraud:")
        
        if not fp_data.empty:
            st.dataframe(fp_data.sort_values('fraud_probability', ascending=False).head(5))
        else:
            st.info("There are no false positives with the current threshold.")
    
    with col2:
        st.markdown("#### False Negatives (Sample)")
        st.markdown("Frauds not detected by the model:")
        
        if not fn_data.empty:
            st.dataframe(fn_data.sort_values('fraud_probability', ascending=False).head(5))
        else:
            st.info("There are no false negatives with the current threshold.")

with tabs[2]:
    st.subheader("ROC and Precision-Recall Curves")
    
    # Generate points for curves (simulated)
    # In a real implementation, these would be calculated from actual predictions
    
    # Points for ROC curve (simulated)
    fpr = np.linspace(0, 1, 100)
    tpr = np.power(fpr, 0.3)  # Simulated ROC curve with good performance
    
    # Points for Precision-Recall curve (simulated)
    recall_points = np.linspace(0, 1, 100)
    precision_points = np.exp(-2.5 * recall_points)  # Simulated PR curve
    
    col1, col2 = st.columns(2)
    
    with col1:
        # ROC curve
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, 'b-', label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], 'r--', label='Random')
        
        # Mark point corresponding to the current threshold
        idx = min(int(threshold * 100), 99)  # Index corresponding to the threshold
        ax.plot(fpr[idx], tpr[idx], 'ko', markersize=8, label=f'Threshold = {threshold:.2f}')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate (Recall)')
        ax.set_title('ROC Curve')
        ax.legend(loc="lower right")
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)
    
    with col2:
        # Precision-Recall curve
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall_points, precision_points, 'b-', label='Precision-Recall curve')
        
        # Baseline line (proportion of positives in the dataset)
        baseline = len(eval_df[eval_df['true_label'] == 1]) / len(eval_df)
        ax.axhline(y=baseline, color='r', linestyle='--', 
                   label=f'Baseline ({baseline:.2%})')
        
        # Mark point corresponding to the current threshold
        idx = min(int(threshold * 100), 99)  # Index corresponding to the threshold
        recall_threshold = recall_points[idx]
        precision_threshold = precision_points[idx]
        ax.plot(recall_threshold, precision_threshold, 'ko', markersize=8, 
                label=f'Threshold = {threshold:.2f}')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="upper right")
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)
    
    # Curve explanation
    st.markdown("""
        ### Curve Interpretation
        
        **Receiver Operating Characteristic (ROC):**
        - Shows the trade-off between true positive rate (sensitivity) and false positive rate (1-specificity).
        - A curve closer to the upper left corner indicates better performance.
        - The AUC (area under the curve) quantifies the model's performance independent of the threshold.
        
        **Precision-Recall Curve:**
        - Shows the trade-off between precision and recall (sensitivity).
        - It's particularly useful when classes are imbalanced, like in fraud detection.
        - A curve closer to the upper right corner indicates better performance.
    """)

with tabs[3]:
    st.subheader("Model Calibration")
    
    # Generate simulated calibration data
    # In a real implementation, this would be calculated from actual predictions
    
    # Create probability bins
    bins = np.linspace(0, 1, 11)  # 10 bins
    bin_midpoints = (bins[:-1] + bins[1:]) / 2
    
    # Create axis of observed positive fraction
    # Simulate a curve slightly below the diagonal (slightly optimistic model)
    fraction_positives = bin_midpoints * 0.9 + 0.05
    
    # Plot calibration curve
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calibration curve
    ax.plot(bin_midpoints, fraction_positives, 's-', label=f'Calibration of {selected_model}')
    
    # Reference diagonal line (perfect calibration)
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Observed Positive Fraction')
    ax.set_title('Calibration Curve')
    ax.legend(loc="lower right")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    st.pyplot(fig)
    
    # Histogram of probability distribution by true class
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for label, color, title in zip([0, 1], ['#4CAF50', '#F44336'], ['Not Fraud', 'Fraud']):
        # Filter by true class
        mask = eval_df['true_label'] == label
        
        # Add histogram
        ax.hist(eval_df.loc[mask, 'fraud_probability'], bins=20, alpha=0.6, color=color, label=title)
    
    # Add vertical line for threshold
    ax.axvline(x=threshold, color='k', linestyle='--', label=f'Threshold = {threshold:.2f}')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_xlabel('Predicted Fraud Probability')
    ax.set_ylabel('Number of Transactions')
    ax.set_title('Probability Distribution by True Class')
    ax.legend()
    
    st.pyplot(fig)
    
    # Calibration explanation
    st.markdown("""
        ### What is Model Calibration?
        
        Calibration refers to how well the probabilities predicted by the model correspond to the observed frequencies 
        of positive events. A well-calibrated model assigns:
        
        - Probability of 0.8 to samples that have ~80% chance of being positive
        - Probability of 0.5 to samples that have ~50% chance of being positive
        
        **Interpretation:**
        - If the curve is below the diagonal, the model is optimistic (overestimates probabilities)
        - If the curve is above the diagonal, the model is pessimistic (underestimates probabilities)
        - The closer to the diagonal, the better calibrated the model
        
        Good calibration is essential for making decisions based on probabilities, especially 
        when there are different costs associated with different types of error.
    """)

with tabs[4]:
    st.subheader("Threshold Analysis")
    
    # Calculate metrics for different thresholds
    thresholds = np.linspace(0.01, 0.99, 99)
    metrics_by_threshold = []
    
    for t in thresholds:
        # Apply threshold
        predicted = (eval_df['fraud_probability'] >= t).astype(int)
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(eval_df['true_label'], predicted).ravel()
        
        # Calculate derived metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Add to DataFrame
        metrics_by_threshold.append({
            'threshold': t,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'specificity': specificity,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        })
    
    metrics_df = pd.DataFrame(metrics_by_threshold)
    
    # Plot metrics by threshold
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(metrics_df['threshold'], metrics_df['precision'], 'b-', label='Precision')
    ax.plot(metrics_df['threshold'], metrics_df['recall'], 'g-', label='Recall')
    ax.plot(metrics_df['threshold'], metrics_df['f1'], 'r-', label='F1-Score')
    ax.plot(metrics_df['threshold'], metrics_df['accuracy'], 'c-', label='Accuracy')
    ax.plot(metrics_df['threshold'], metrics_df['specificity'], 'm-', label='Specificity')
    
    # Add vertical line for the current threshold
    ax.axvline(x=threshold, color='k', linestyle='--', label=f'Current Threshold = {threshold:.2f}')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Metric Value')
    ax.set_title('Performance Metrics by Threshold')
    ax.legend(loc="center right")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    st.pyplot(fig)
    
    # Optimal thresholds for different criteria
    optimal_thresholds = {
        'F1 Score': metrics_df.loc[metrics_df['f1'].idxmax(), 'threshold'],
        'Precision': metrics_df.loc[metrics_df['precision'].idxmax(), 'threshold'],
        'Recall': metrics_df.loc[metrics_df['recall'].idxmax(), 'threshold'],
        'Precision-Recall Balance': metrics_df.loc[(metrics_df['precision'] - metrics_df['recall']).abs().idxmin(), 'threshold']
    }
    
    # Display optimal thresholds
    st.subheader("Optimal Thresholds by Criteria")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Optimal thresholds table
        optimal_df = pd.DataFrame({
            'Criteria': list(optimal_thresholds.keys()),
            'Optimal Threshold': list(optimal_thresholds.values())
        })
        st.dataframe(optimal_df)
    
    with col2:
        # Recommendation based on use case
        st.markdown("""
            ### Threshold Recommendation
            
            The choice of threshold depends on business objectives:
            
            - **Maximize Recall**: Prioritize detecting the maximum number of frauds, accepting more false alarms.
            - **Maximize Precision**: Prioritize minimizing false alarms, accepting losing some frauds.
            - **Maximize F1-Score**: Seek a balance between precision and recall.
            
            For fraud detection systems in credit card transactions, the cost of 
            losing a fraud (false negative) is usually higher than the cost of investigating a false alarm. 
            We recommend using a lower threshold to prioritize recall, unless the volume 
            of transactions makes the number of false positives operationally infeasible.
        """)
    
    # Fraud cost vs. false alarms
    st.subheader("Cost Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Inputs for costs
        cost_fn = st.number_input("Cost of an Undetected Fraud (R$)", value=1000, step=100)
        cost_fp = st.number_input("Cost of Investigating a False Alarm (R$)", value=50, step=10)
    
    with col2:
        # Calculate total costs for each threshold
        metrics_df['cost'] = (metrics_df['false_negatives'] * cost_fn) + (metrics_df['false_positives'] * cost_fp)
        
        # Find threshold with minimum cost
        min_cost_idx = metrics_df['cost'].idxmin()
        optimal_cost_threshold = metrics_df.loc[min_cost_idx, 'threshold']
        min_cost = metrics_df.loc[min_cost_idx, 'cost']
        
        st.success(f"""
            **Optimal Threshold for Minimizing Costs**: {optimal_cost_threshold:.2f}
            
            With this threshold, the estimated total cost would be R$ {min_cost:.2f}
        """)
    
    # Cost chart by threshold
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(metrics_df['threshold'], metrics_df['cost'], 'b-', label='Total Cost')
    ax.plot(metrics_df['threshold'], metrics_df['false_negatives'] * cost_fn, 'r--', 
           label='Cost of Undetected Frauds')
    ax.plot(metrics_df['threshold'], metrics_df['false_positives'] * cost_fp, 'g--', 
           label='Cost of False Alarms')
    
    # Mark threshold with minimum cost
    ax.axvline(x=optimal_cost_threshold, color='k', linestyle='--', 
              label=f'Optimal Threshold = {optimal_cost_threshold:.2f}')
    
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Cost (R$)')
    ax.set_title('Cost Analysis by Threshold')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    st.pyplot(fig) 
