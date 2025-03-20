"""
Modeling Page for the fraud detection application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import pickle
import time

# Add root directory to path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.preprocessing import load_config, load_data, prepare_data
from features.creation import create_all_features
from features.selection import select_best_features

# Page configuration
st.set_page_config(
    page_title="Modeling - Fraud Detection", 
    page_icon="🤖",
    layout="wide"
)

# Page title
st.title("Modeling and Training")
st.markdown("""
    This page allows you to train and compare different machine learning models
    to detect fraudulent credit card transactions.
""")

# Load data (with cache)
@st.cache_data
def load_cached_data():
    """Loads data with cache."""
    config = load_config()
    df = load_data(config)
    return df

# Load data
with st.spinner("Loading data..."):
    df = load_cached_data()

# Model settings
st.sidebar.header("Training Settings")

# Model selection
st.sidebar.subheader("Select Models")
models_to_train = {
    "Logistic Regression": st.sidebar.checkbox("Logistic Regression", value=True),
    "Random Forest": st.sidebar.checkbox("Random Forest", value=True),
    "XGBoost": st.sidebar.checkbox("XGBoost", value=True),
    "LightGBM": st.sidebar.checkbox("LightGBM", value=False),
    "Ensemble": st.sidebar.checkbox("Ensemble (Voting)", value=False)
}

# General training settings
st.sidebar.subheader("General Settings")
test_size = st.sidebar.slider("Test Size (%)", min_value=10, max_value=40, value=20) / 100
cv_folds = st.sidebar.slider("Cross Validation (k)", min_value=3, max_value=10, value=5)

# Class balancing options
st.sidebar.subheader("Class Balancing")
balance_method = st.sidebar.radio(
    "Balancing Method",
    options=["None", "SMOTE", "RandomUnderSampler", "Class Weights"],
    index=1
)

# Hyperparameters by model
st.sidebar.subheader("Hyperparameters")
show_hyperparams = st.sidebar.checkbox("Configure Hyperparameters", value=False)

if show_hyperparams:
    if models_to_train["Logistic Regression"]:
        st.sidebar.markdown("**Logistic Regression**")
        lr_c = st.sidebar.select_slider(
            "C (Regularization)",
            options=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            value=1.0
        )
        lr_solver = st.sidebar.selectbox(
            "Solver",
            options=["liblinear", "lbfgs", "newton-cg", "sag", "saga"],
            index=1
        )
    
    if models_to_train["Random Forest"]:
        st.sidebar.markdown("**Random Forest**")
        rf_n_estimators = st.sidebar.slider(
            "Number of Trees",
            min_value=50,
            max_value=300,
            value=100,
            step=10
        )
        rf_max_depth = st.sidebar.slider(
            "Maximum Depth",
            min_value=5,
            max_value=30,
            value=10
        )
    
    if models_to_train["XGBoost"]:
        st.sidebar.markdown("**XGBoost**")
        xgb_learning_rate = st.sidebar.slider(
            "Learning Rate",
            min_value=0.01,
            max_value=0.3,
            value=0.1,
            step=0.01
        )
        xgb_n_estimators = st.sidebar.slider(
            "Number of Estimators (XGB)",
            min_value=50,
            max_value=300,
            value=100,
            step=10
        )

# Main content
st.header("Train Fraud Detection Models")

# Load data for processing
with st.expander("View loaded data"):
    st.dataframe(df.head())
    st.write(f"Dimensions: {df.shape[0]} rows and {df.shape[1]} columns")

# Prepare data and feature engineering
st.subheader("Data Preparation")

col1, col2 = st.columns(2)
with col1:
    target_col = "Fraud Indicator"
    if target_col in df.columns:
        st.success(f"✅ Target column '{target_col}' found in data")
        
        # Show class distribution
        class_counts = df[target_col].value_counts()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(
            class_counts, 
            labels=["Not Fraud", "Fraud"],
            autopct='%1.1f%%',
            colors=['#4CAF50', '#F44336'],
            explode=[0, 0.1]
        )
        ax.set_title('Class Distribution')
        st.pyplot(fig)
    else:
        st.error(f"❌ Target column '{target_col}' not found in data")

with col2:
    # Feature summary
    st.write("Feature Engineering:")
    
    # Display number of original features and after processing
    feature_metrics = {
        "Metrics": ["Original Features", "After Engineering", "After Selection"],
        "Quantity": [df.shape[1], df.shape[1] + 15, 20]  # simulated values
    }
    st.table(pd.DataFrame(feature_metrics))
    
    # Button to see feature details
    if st.button("View Feature Details"):
        st.info("In a complete implementation, selected feature details would be shown here.")

# Training options
st.subheader("Start Training")

if st.button("Train Selected Models"):
    # Here would be the real training code, but we'll simulate to demonstrate the UI
    
    # Create a progress container
    progress_container = st.container()
    
    # Start training
    with progress_container:
        st.markdown("### Training Progress")
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        # Simulate training for each selected model
        models_count = sum(models_to_train.values())
        trained_models = []
        training_results = []
        
        current_model_idx = 0
        
        for model_name, selected in models_to_train.items():
            if selected:
                current_model_idx += 1
                progress_text.text(f"Training model: {model_name} ({current_model_idx}/{models_count})...")
                
                # Simulate training time
                for pct in range(1, 101):
                    progress_bar.progress(pct / 100)
                    time.sleep(0.02)  # Simulate processing
                
                # Simulate performance metrics
                accuracy = np.random.uniform(0.86, 0.95)
                precision = np.random.uniform(0.83, 0.92)
                recall = np.random.uniform(0.8, 0.9)
                f1 = np.random.uniform(0.81, 0.91)
                roc_auc = np.random.uniform(0.88, 0.97)
                train_time = np.random.uniform(0.8, 10.0)
                
                # Add to results
                training_results.append({
                    "Model": model_name,
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1-Score": f1,
                    "ROC AUC": roc_auc,
                    "Time (s)": train_time
                })
                
                # Update total progress bar
                progress_bar.progress(current_model_idx / models_count)
        
        progress_text.text("Training completed!")
        
        # Display training results
        st.subheader("Training Results")
        results_df = pd.DataFrame(training_results)
        results_df_display = results_df.style.background_gradient(
            subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC'], 
            cmap='YlGn'
        )
        st.dataframe(results_df_display)
        
        # Comparative metrics chart
        st.subheader("Model Comparison")
        
        metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC AUC"]
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bar_width = 0.15
        x = np.arange(len(metrics_to_plot))
        
        for i, (_, row) in enumerate(results_df.iterrows()):
            offset = (i - len(results_df) / 2) * bar_width + bar_width / 2
            ax.bar(x + offset, row[metrics_to_plot], width=bar_width, label=row["Model"])
        
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_to_plot)
        ax.set_ylim(0.7, 1.0)  # Focus on relevant range
        ax.set_xlabel('Metric')
        ax.set_ylabel('Value')
        ax.set_title('Metric Comparison by Model')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        st.pyplot(fig)
        
        # Display information about the best model
        best_model_idx = results_df["ROC AUC"].idxmax()
        best_model = results_df.iloc[best_model_idx]
        
        st.subheader("Best Model")
        st.info(f"""
            **{best_model['Model']}** achieved the best performance:
            - ROC AUC: {best_model['ROC AUC']:.4f}
            - Accuracy: {best_model['Accuracy']:.4f}
            - Precision: {best_model['Precision']:.4f}
            - Recall: {best_model['Recall']:.4f}
            - F1-Score: {best_model['F1-Score']:.4f}
            - Training Time: {best_model['Time (s)']:.2f} seconds
        """)
        
        # Export options (simulated)
        st.subheader("Export Model")
        export_format = st.selectbox("Export Format", ["Pickle (.pkl)", "ONNX (.onnx)", "JSON (.json)"])
        
        if st.button("Save Model"):
            st.success("Model saved successfully! (simulation)")

# Additional information
with st.expander("How to select the best model?"):
    st.markdown("""
        ### Model Selection Criteria
        
        For credit card fraud detection, we recommend prioritizing the following metrics:
        
        1. **ROC AUC**: It's a robust metric for imbalanced data, common in fraud cases.
           - Values above 0.95 are considered excellent.
        
        2. **Recall (sensitivity)**: It's essential to maximize the detection of actual fraud.
           - The cost of missing a fraud (false negative) is generally higher than a false alert.
        
        3. **Precision**: Important to reduce false alerts.
           - Should be balanced with Recall, depending on operational costs.
        
        4. **F1-Score**: Harmonic mean between precision and recall, useful when seeking balance.
        
        5. **Inference time**: For real-time applications, also consider model speed.
    """) 
