"""
Main application file for the Credit Card Fraud Detection system.
This file contains the Streamlit web application.
"""

import os
import sys
import logging
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from contextlib import nullcontext
import io

# Add project root to path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from utils.config_utils import load_config
from data.preprocessing import load_data, prepare_data, generate_synthetic_data
from features.creation import create_all_features
from features.selection import select_best_features
from features.advanced_features import create_all_advanced_features
from models.training import create_model, train_model, create_ensemble_model
from models.evaluation import evaluate_model, compare_models
from models.model_registry import ModelRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info("Imports completed successfully")

# Configure Streamlit page
def configure_page():
    """
    Configure the Streamlit page settings.
    Sets the page title, icon, layout, and theme.
    """
    st.set_page_config(
        page_title="Credit Card Fraud Detection",
        page_icon="🔒",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/yourusername/fraud-credit',
            'Report a bug': 'https://github.com/yourusername/fraud-credit/issues',
            'About': 'Credit Card Fraud Detection System - Developed with ML and Streamlit'
        }
    )
    
    # Set dark theme
    st.markdown("""
        <style>
        :root {
            --background-color: #0e1117;
            --text-color: #fafafa;
            --font: 'Source Sans Pro', sans-serif;
        }
        
        .stApp {
            background-color: var(--background-color);
            color: var(--text-color);
        }
        
        .stSidebar {
            background-color: #262730;
        }
        
        /* Custom styling for headers */
        h1, h2, h3, h4, h5 {
            color: #4da6ff;
            font-family: var(--font);
        }
        
        /* Custom styling for links */
        a {
            color: #29b5e8 !important;
            text-decoration: none;
        }
        
        /* Button styling */
        .stButton>button {
            background-color: #4da6ff;
            color: white;
            border-radius: 5px;
            border: none;
            padding: 8px 16px;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            background-color: #3485cc;
            color: white;
        }
        
        /* Create a professional-looking box for metrics */
        div[data-testid="stMetricValue"] {
            background-color: #262730;
            padding: 10px;
            border-radius: 5px;
            border-left: 5px solid #4da6ff;
        }
        </style>
    """, unsafe_allow_html=True)

# Main application function
def main():
    """
    Main application function.
    Sets up the application configuration, loads data, and handles navigation.
    """
    # Setup the page configuration
    configure_page()
    
    # Set application title
    st.title("Credit Card Fraud Detection System")
    st.markdown("### Interactive platform for fraud analysis and detection")
    
    # Initialize session state for data if not already present
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Setup sidebar
    st.sidebar.title("Navigation")
    
    # Page selector
    pages = {
        "Home": "Home",
        "Data Upload": "Data Upload",
        "Data Exploration": "Exploratory Analysis",
        "Feature Engineering": "Feature Engineering",
        "Modeling": "Modeling and Training",
        "Evaluation": "Model Evaluation",
        "Monitoring Dashboard": "Real-time Monitoring",
        "Inference": "Real-time Inference"
    }
    
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    
    # Display selected page
    if selection == "Home":
        display_home_page()
    elif selection == "Data Upload":
        upload_data()
    elif selection == "Data Exploration":
        if not st.session_state.data_loaded:
            st.warning("Please upload your data first in the 'Data Upload' section.")
            upload_data()
        else:
            display_eda_page(st.session_state.data)
    elif selection == "Feature Engineering":
        if not st.session_state.data_loaded:
            st.warning("Please upload your data first in the 'Data Upload' section.")
            upload_data()
        else:
            display_feature_engineering_page(st.session_state.data)
    elif selection == "Modeling":
        if not st.session_state.data_loaded:
            st.warning("Please upload your data first in the 'Data Upload' section.")
            upload_data()
        else:
            display_modeling_page(st.session_state.data)
    elif selection == "Evaluation":
        if not st.session_state.data_loaded:
            st.warning("Please upload your data first in the 'Data Upload' section.")
            upload_data()
        else:
            display_evaluation_page(st.session_state.data)
    elif selection == "Monitoring Dashboard":
        if not st.session_state.data_loaded:
            st.warning("Please upload your data first in the 'Data Upload' section.")
            upload_data()
        else:
            display_monitoring_dashboard(st.session_state.data)
    elif selection == "Inference":
        if not st.session_state.data_loaded:
            st.warning("Please upload your data first in the 'Data Upload' section.")
            upload_data()
        else:
            display_inference_page(st.session_state.data)
    
    # Show data source info in sidebar if data is loaded
    if st.session_state.data_loaded:
        st.sidebar.success(f"Using {'uploaded' if st.session_state.data_source == 'uploaded' else 'synthetic'} data")
        if st.sidebar.button("Upload Different Data"):
            st.session_state.data_loaded = False
            st.session_state.page_selection = "Data Upload"
            st.rerun()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Credit Card Fraud Detection System\n\n"
        "© 2025 | Version 1.0.0\n\n"
        "Built with Streamlit and ML"
    )

def upload_data():
    """
    Allows users to upload data or use a synthetic dataset.
    Stores the uploaded data in the session state for use across the application.
    """
    st.header("Data Upload")
    
    st.markdown("""
    ## Upload your credit card transaction data
    
    Upload your CSV file containing credit card transaction data or use our synthetic dataset.
    The data should contain transaction information and a binary fraud indicator column.
    
    **All other application features will use this uploaded data.**
    """)
    
    data_option = st.radio(
        "Select data source:",
        ["Upload your own data", "Use synthetic data"]
    )
    
    if data_option == "Upload your own data":
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.success(f"Data loaded successfully: {data.shape[0]} rows and {data.shape[1]} columns")
                
                # Store data in session
                st.session_state.data = data
                st.session_state.data_loaded = True
                st.session_state.data_source = "uploaded"
                
                # Preview the data
                st.subheader("Data Preview")
                st.dataframe(data.head())
                
                # Data info
                st.subheader("Data Info")
                buffer = io.StringIO()
                data.info(buf=buffer)
                st.text(buffer.getvalue())
                
                # Check for fraud indicator column
                if 'Fraud Indicator' not in data.columns:
                    st.warning("""
                    Warning: 'Fraud Indicator' column not found. 
                    For best results, your data should include a binary fraud indicator column (0 = normal, 1 = fraud).
                    """)
                
                # Button to continue to data exploration
                if st.button("Continue to Data Exploration"):
                    st.session_state.page_selection = "Data Exploration"
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error loading data: {e}")
    else:
        # Generate synthetic data
        if st.button("Generate Synthetic Data"):
            # Use the function from preprocessing instead of the local one
            data = generate_synthetic_data()
            st.success(f"Synthetic data generated: {data.shape[0]} rows and {data.shape[1]} columns")
            
            # Store data in session
            st.session_state.data = data
            st.session_state.data_loaded = True
            st.session_state.data_source = "synthetic"
            
            # Preview the data
            st.subheader("Data Preview")
            st.dataframe(data.head())
            
            # Button to continue to data exploration
            if st.button("Continue to Data Exploration"):
                st.session_state.page_selection = "Data Exploration"
                st.rerun()
                
    # Display message if no data is loaded
    if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
        st.info("Please upload your data or generate synthetic data to proceed.")

# Application pages
def display_home_page():
    """
    Display the home page of the application.
    Includes project overview, key statistics, and workflow instructions.
    """
    # Title and introduction
    st.markdown("## Welcome to the Credit Card Fraud Detection System")
    
    # Overview cards using columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div style="padding: 15px; background-color: #262730; border-radius: 10px; height: 150px;">
                <h3 style="color: #4da6ff;">🔍 Analyze</h3>
                <p>Explore transaction data with advanced visualizations and analytics</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style="padding: 15px; background-color: #262730; border-radius: 10px; height: 150px;">
                <h3 style="color: #4da6ff;">🧠 Learn</h3>
                <p>Train machine learning models to identify fraudulent patterns</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div style="padding: 15px; background-color: #262730; border-radius: 10px; height: 150px;">
                <h3 style="color: #4da6ff;">🛡️ Detect</h3>
                <p>Identify suspicious transactions in real-time with high accuracy</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Project description
    st.markdown("### About the Project")
    st.markdown("""
        This system uses advanced machine learning techniques to detect fraudulent credit card transactions.
        With increasing online payments, protecting customers from fraud has become more important than ever.
        
        Key project features:
        - **Data Exploration**: Interactive visualizations to understand transaction patterns
        - **Feature Engineering**: Creation of relevant fraud indicators  
        - **Model Training**: Multiple algorithms including Random Forest, XGBoost, and Deep Learning
        - **Real-time Detection**: Score new transactions with pre-trained models
    """)
    
    # Application workflow
    st.markdown("### How to Use")
    
    st.markdown("""
        **1. Exploratory Analysis**
        - Analyze transaction distribution
        - Visualize correlations and patterns
        - Identify key risk indicators
        
        **2. Feature Engineering**
        - Create behavioral features
        - Extract temporal patterns
        - Calculate risk scores
        
        **3. Modeling**
        - Train multiple ML models
        - Optimize hyperparameters
        - Create ensemble models
        
        **4. Evaluation**
        - Measure model performance
        - Compare different approaches
        - Analyze confusion matrices
        
        **5. Inference**
        - Score new transactions
        - Get fraud probability
        - Understand model decisions
    """)
    
    # Add quick navigation buttons
    st.markdown("### Quick Navigation")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Explore Data"):
            # Use session state to navigate
            st.session_state.page_selection = "Data Exploration"
            st.rerun()
    
    with col2:
        if st.button("⚙️ Engineer Features"):
            st.session_state.page_selection = "Feature Engineering"
            st.rerun()
    
    with col3:
        if st.button("🤖 Train Models"):
            st.session_state.page_selection = "Modeling"
            st.rerun()

def display_eda_page(df):
    """
    Display the Exploratory Data Analysis page.
    Visualizes transaction data with charts and statistics.
    """
    st.header("Exploratory Data Analysis")
    
    # Data Summary
    st.subheader("Data Summary")
    st.write(f"DataFrame dimensions: {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Show first rows
    with st.expander("View first rows of the dataset"):
        st.dataframe(df.head())
    
    # Descriptive statistics
    with st.expander("Descriptive Statistics"):
        st.dataframe(df.describe())
    
    # Fraud Distribution
    st.subheader("Fraud Distribution")
    if 'Fraud Indicator' in df.columns:
        fraud_count = df['Fraud Indicator'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Count table
            st.write("Count by Class:")
            st.dataframe(pd.DataFrame({
                'Class': ['Non-Fraud', 'Fraud'],
                'Count': [fraud_count.get(0, 0), fraud_count.get(1, 0)]
            }))
        
        with col2:
            # Pie chart
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(
                fraud_count, 
                labels=['Non-Fraud', 'Fraud'], 
                autopct='%1.1f%%',
                colors=['#4CAF50', '#F44336'],
                explode=[0, 0.1]
            )
            ax.set_title('Fraud Distribution')
            st.pyplot(fig)
    
    # Transaction Amount Analysis
    st.subheader("Transaction Amount Analysis")
    if 'Amount' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram of amounts
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=df, x='Amount', hue='Fraud Indicator' if 'Fraud Indicator' in df.columns else None, bins=30, ax=ax)
            ax.set_title('Transaction Amount Distribution')
            ax.set_xlabel('Transaction Amount')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
        
        with col2:
            # Boxplot of amounts by fraud
            if 'Fraud Indicator' in df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(data=df, x='Fraud Indicator', y='Amount', ax=ax)
                ax.set_title('Transaction Amount by Fraud Indicator')
                ax.set_xlabel('Fraud (0=No, 1=Yes)')
                ax.set_ylabel('Transaction Amount')
                st.pyplot(fig)

def display_feature_engineering_page(df):
    """
    Display the Feature Engineering page.
    Shows created features and their visualizations.
    """
    st.header("Feature Engineering")
    
    # Create features
    with st.spinner("Creating features..."):
        df_features = create_all_features(df)
        
        # Identify new features
        original_cols = set(df.columns)
        feature_cols = set(df_features.columns)
        new_features = feature_cols - original_cols
    
    # Show created features
    st.subheader("Created Features")
    st.write(f"{len(new_features)} new features were created from the original data.")
    
    # Feature category filter
    feature_categories = {
        "All": list(new_features),
        "Temporal": [f for f in new_features if any(kw in f for kw in ["time", "hour", "day", "week", "month", "period"])],
        "Amount": [f for f in new_features if any(kw in f for kw in ["amount", "value"])],
        "Location": [f for f in new_features if any(kw in f for kw in ["city", "state", "location"])],
        "Behavior": [f for f in new_features if any(kw in f for kw in ["customer", "behavior", "transaction"])],
        "Fraud Patterns": [f for f in new_features if any(kw in f for kw in ["risk", "fraud", "unusual"])]
    }
    
    selected_category = st.selectbox("Filter by category:", list(feature_categories.keys()))
    selected_features = feature_categories[selected_category]
    
    # Feature table
    if selected_features:
        feature_df = pd.DataFrame({
            "Feature": selected_features,
            "Type": [str(df_features[f].dtype) for f in selected_features]
        })
        st.dataframe(feature_df)
        
        # Select feature to visualize
        selected_feature = st.selectbox("Select a feature to visualize:", selected_features)
        
        if selected_feature:
            st.subheader(f"Feature Visualization: {selected_feature}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Feature statistics
                st.write("Statistics:")
                st.dataframe(df_features[selected_feature].describe())
            
            with col2:
                # Appropriate visualization based on data type
                fig, ax = plt.subplots(figsize=(10, 6))
                
                if df_features[selected_feature].dtype in ['int64', 'float64']:
                    # Numeric: histogram
                    if 'Fraud Indicator' in df_features.columns:
                        sns.histplot(data=df_features, x=selected_feature, hue='Fraud Indicator', bins=30, ax=ax)
                    else:
                        sns.histplot(data=df_features, x=selected_feature, bins=30, ax=ax)
                else:
                    # Categorical: count
                    if 'Fraud Indicator' in df_features.columns:
                        df_plot = df_features.groupby([selected_feature, 'Fraud Indicator']).size().reset_index(name='count')
                        sns.barplot(data=df_plot, x=selected_feature, y='count', hue='Fraud Indicator', ax=ax)
                    else:
                        sns.countplot(data=df_features, x=selected_feature, ax=ax)
                
                ax.set_title(f'Distribution of {selected_feature}')
                plt.xticks(rotation=45)
                st.pyplot(fig)

def display_modeling_page(df):
    """
    Display the Modeling page.
    Allows training and comparing different fraud detection models.
    """
    st.header("Modeling")
    
    st.info("This page allows you to train and compare different fraud detection models.")
    
    # Model selection
    st.subheader("Select models to train")
    
    col1, col2 = st.columns(2)
    
    with col1:
        models_to_train = {
            "Logistic Regression": st.checkbox("Logistic Regression", value=True),
            "Random Forest": st.checkbox("Random Forest", value=True),
            "XGBoost": st.checkbox("XGBoost", value=True),
            "LightGBM": st.checkbox("LightGBM", value=False)
        }
    
    with col2:
        # Training parameters
        test_size = st.slider("Test set size (%)", 10, 40, 20) / 100
        use_smote = st.checkbox("Use SMOTE to balance classes", value=True)
        random_state = st.number_input("Random State", min_value=1, max_value=100, value=42)
    
    # Button to train models
    if st.button("Train Models"):
        st.info("Model training would be executed here, using the selected data and parameters.")
        st.success("This is a demonstration. In a complete implementation, training would be performed and results would be displayed.")
        
        # Here the actual training function would be called
        # Example:
        # from models.training import train_all_models
        # models, times = train_all_models(X_train, y_train, config)
        
        # Show simulated results
        st.subheader("Training Results (Simulated)")
        
        # Simulated data
        model_results = {
            "Model": ["Logistic Regression", "Random Forest", "XGBoost", "Ensemble"],
            "Accuracy": [0.923, 0.942, 0.951, 0.956],
            "Precision": [0.876, 0.901, 0.924, 0.932],
            "Recall": [0.863, 0.885, 0.903, 0.915],
            "F1-Score": [0.869, 0.893, 0.913, 0.923],
            "ROC AUC": [0.934, 0.962, 0.974, 0.978],
            "Time (s)": [1.2, 8.5, 5.7, 12.1]
        }
        
        # Filter only selected models
        selected_models = ["Logistic Regression", "Random Forest", "XGBoost", "Ensemble"]
        
        # Display results table
        results_df = pd.DataFrame(model_results)
        st.dataframe(results_df)
        
        # Metrics chart
        fig, ax = plt.subplots(figsize=(12, 6))
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC AUC"]
        
        for i, model in enumerate(selected_models):
            values = [model_results[metric][i] for metric in metrics]
            ax.plot(metrics, values, marker='o', label=model)
        
        ax.set_ylim(0.8, 1.0)
        ax.legend()
        ax.set_title("Metrics Comparison by Model")
        ax.grid(True)
        st.pyplot(fig)

def display_evaluation_page(df):
    """
    Display the Model Evaluation page.
    Allows evaluation of trained models using various metrics and visualizations.
    """
    st.header("Model Evaluation")
    
    st.info("This page allows you to evaluate trained models through various metrics and visualizations.")
    
    # Model selection
    model_options = ["Random Forest", "XGBoost", "Logistic Regression", "LightGBM", "Ensemble"]
    selected_model = st.selectbox("Select a model to evaluate:", model_options)
    
    # Simulated metrics
    if selected_model:
        # Add threshold adjustment section
        st.subheader("Risk Threshold Adjustment")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            threshold = st.slider(
                "Detection Threshold", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.5, 
                step=0.05,
                help="Adjust the probability threshold for classifying a transaction as fraudulent."
            )
            
            st.markdown("""
            **Threshold impact:**
            - **Lower threshold** (closer to 0): More transactions flagged as fraud (higher recall, lower precision).
            - **Higher threshold** (closer to 1): Fewer transactions flagged as fraud (lower recall, higher precision).
            """)
            
            # Risk appetite selection
            risk_appetite = st.radio(
                "Select your risk appetite:",
                ["Conservative (prioritize catching all fraud)", 
                 "Balanced (equal focus on precision and recall)",
                 "Permissive (minimize false positives)"]
            )
            
            if risk_appetite == "Conservative (prioritize catching all fraud)":
                st.info("Recommended threshold: 0.3 - 0.4")
                if threshold > 0.4:
                    st.warning("Your current threshold is higher than recommended for a conservative approach.")
            elif risk_appetite == "Permissive (minimize false positives)":
                st.info("Recommended threshold: 0.6 - 0.8")
                if threshold < 0.6:
                    st.warning("Your current threshold is lower than recommended for a permissive approach.")
        
        with col2:
            st.metric("Current Threshold", f"{threshold:.2f}")
            
            # Visualize threshold as a gauge
            fig, ax = plt.subplots(figsize=(3, 3))
            
            # Create a simple gauge
            ax.add_patch(plt.matplotlib.patches.Wedge((0.5, 0), 0.5, 0, 180, width=0.1, color='lightgray'))
            ax.add_patch(plt.matplotlib.patches.Wedge((0.5, 0), 0.5, 0, 180 * threshold, width=0.1, color='#4da6ff'))
            
            # Add threshold indicator
            ax.plot([0.5, 0.5 + 0.4 * np.cos(np.pi * threshold)], 
                    [0, 0.4 * np.sin(np.pi * threshold)], 'k-', lw=2)
            
            ax.text(0.5, -0.15, f"Threshold: {threshold:.2f}", ha='center')
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.2, 0.6)
            ax.axis('off')
            
            st.pyplot(fig)
        
        # Display calculated metrics based on the threshold
        st.subheader("Impact on Metrics")
        
        # Simulate different metrics based on threshold
        precision = 0.75 + (threshold * 0.35)  # Higher threshold → higher precision
        if precision > 1.0:
            precision = 1.0
            
        recall = 0.95 - (threshold * 0.55)  # Higher threshold → lower recall
        if recall < 0.0:
            recall = 0.0
            
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{min(0.8 + threshold/10, 0.99):.1%}")
        col2.metric("Precision", f"{precision:.1%}")
        col3.metric("Recall", f"{recall:.1%}")
        col4.metric("F1-Score", f"{f1:.1%}")
        
        # Show confusion matrix that updates with threshold
        st.subheader("Confusion Matrix at Current Threshold")
        
        # Simulate confusion matrix based on threshold
        # Lower threshold: more positive predictions (more FP, fewer FN)
        # Higher threshold: fewer positive predictions (fewer FP, more FN)
        
        # Base numbers (total 1000 examples with 20% fraud)
        total = 1000
        actual_neg = 800
        actual_pos = 200
        
        # Calculate predictions based on threshold
        # As threshold increases, we predict fewer positives
        pred_pos_rate = 0.3 - 0.1 * threshold + (1 - threshold) * 0.2  # Decreases with threshold
        
        # Calculate confusion matrix values
        # True Positives: accurate fraud detections (decreases with threshold)
        tp = int(actual_pos * (1.0 - 0.7 * threshold))
        
        # False Negatives: missed frauds (increases with threshold)
        fn = actual_pos - tp
        
        # False Positives: false alarms (decreases with threshold)
        fp = int(actual_neg * (0.2 * (1.0 - threshold)))
        
        # True Negatives: correctly identified legitimate transactions
        tn = actual_neg - fp
        
        cm = np.array([[tn, fp], [fn, tp]])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Non-Fraud', 'Fraud'],
                    yticklabels=['Non-Fraud', 'Fraud'])
        ax.set_ylabel('True Value')
        ax.set_xlabel('Predicted Value')
        ax.set_title(f'Confusion Matrix (Threshold = {threshold:.2f})')
        st.pyplot(fig)
        
        # Cost-benefit analysis
        st.subheader("Cost-Benefit Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Adjustable cost parameters
            fraud_cost = st.number_input("Average cost per undetected fraud ($)", min_value=0, value=1000, step=100)
            investigation_cost = st.number_input("Cost per investigation ($)", min_value=0, value=50, step=10)
            
            # Calculate costs
            cost_of_missed_fraud = fn * fraud_cost
            cost_of_investigations = (tp + fp) * investigation_cost
            total_cost = cost_of_missed_fraud + cost_of_investigations
            
            # Display costs
            st.metric("Total Estimated Cost", f"${total_cost:,}")
            
        with col2:
            # Pie chart for cost breakdown
            fig, ax = plt.subplots(figsize=(6, 6))
            costs = [cost_of_missed_fraud, cost_of_investigations]
            labels = [f'Undetected Fraud (${cost_of_missed_fraud:,})', 
                     f'Investigations (${cost_of_investigations:,})']
            ax.pie(costs, labels=labels, autopct='%1.1f%%', 
                   colors=['#ff9999', '#66b3ff'],
                   explode=[0.1, 0])
            ax.set_title('Cost Breakdown')
            st.pyplot(fig)
            
        # Note about the simulated nature
        st.info("Note: These metrics are simulated for demonstration purposes. In a production system, they would be calculated from actual model predictions on validation data.")

        # Original visualizations
        st.subheader("Additional Visualizations")
        
        tab1, tab2, tab3, tab4 = st.tabs(["ROC Curve", "Precision-Recall", "SHAP", "LIME"])
        
        with tab1:
            # Simulated ROC Curve
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Simulated data
            fpr = np.linspace(0, 1, 100)
            tpr = np.power(fpr, 0.2)  # Simulated ROC curve
            
            # Add current threshold point
            threshold_idx = int(threshold * 99)  # Convert threshold to index
            current_fpr = fpr[threshold_idx]
            current_tpr = tpr[threshold_idx]
            
            ax.plot(fpr, tpr, label=f'ROC curve (area = 0.962)')
            ax.plot([0, 1], [0, 1], 'k--')
            ax.scatter([current_fpr], [current_tpr], color='red', s=100, 
                      label=f'Current threshold ({threshold:.2f})')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve with Current Threshold')
            ax.legend(loc="lower right")
            st.pyplot(fig)
        
        with tab2:
            # Simulated Precision-Recall Curve
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Simulated data
            recall_values = np.linspace(0, 1, 100)
            precision_values = np.exp(-2.5 * recall_values)  # Simulated PR curve
            
            # Add current threshold point
            threshold_idx = int((1-threshold) * 99)  # Convert threshold to index (inverse for PR curve)
            current_recall = recall_values[threshold_idx]
            current_precision = precision_values[threshold_idx]
            
            ax.plot(recall_values, precision_values, label=f'PR curve (AP = 0.924)')
            ax.axhline(y=0.2, color='r', linestyle='--', alpha=0.5, label='Baseline (0.2)')
            ax.scatter([current_recall], [current_precision], color='red', s=100, 
                      label=f'Current threshold ({threshold:.2f})')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curve with Current Threshold')
            ax.legend(loc="upper right")
            st.pyplot(fig)
        
        with tab3:
            # SHAP plot simulation
            st.image("https://shap.readthedocs.io/en/latest/_images/shap_summary_plot.png", 
                     caption="Example SHAP plot (illustrative image)")
            st.write("This visualization shows the importance and impact of each feature in the model's decision.")
            
            # Additional description for SHAP
            st.markdown("""
            **SHAP (SHapley Additive exPlanations)** values interpret the impact of features by showing:
            - Which features contribute most to the prediction
            - Whether they increase (red) or decrease (blue) the probability of fraud
            - The magnitude of each feature's contribution
            """)
        
        with tab4:
            # LIME explanation (new)
            st.markdown("### LIME Explanation for a Sample Transaction")
            st.markdown("""
            **LIME (Local Interpretable Model-agnostic Explanations)** creates a locally 
            faithful explanation by approximating the model around a specific prediction.
            """)
            
            # Simulated LIME visualization
            sample_features = [
                ("Transaction Amount", 0.32, "Higher amount increases fraud probability"),
                ("Night Time Transaction", 0.27, "Night time increases fraud probability"),
                ("Unusual Location", 0.23, "Unusual merchant location increases fraud probability"),
                ("Customer History", -0.15, "Good customer history decreases fraud probability"),
                ("Merchant Rating", -0.12, "Trusted merchant decreases fraud probability")
            ]
            
            # Create a horizontal bar chart for LIME
            fig, ax = plt.subplots(figsize=(10, 5))
            features = [x[0] for x in sample_features]
            importance = [x[1] for x in sample_features]
            colors = ['#ff6666' if imp > 0 else '#66b3ff' for imp in importance]
            
            bars = ax.barh(features, importance, color=colors)
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax.set_xlabel('Feature Contribution')
            ax.set_title('LIME Explanation for Sample High-Risk Transaction')
            
            # Add a legend
            ax.bar(0, 0, color='#ff6666', label='Increases Fraud Risk')
            ax.bar(0, 0, color='#66b3ff', label='Decreases Fraud Risk')
            ax.legend(loc='lower right')
            
            st.pyplot(fig)
            
            # Add text explanations for each feature
            st.markdown("#### Feature Explanations:")
            for feature, importance, explanation in sample_features:
                st.markdown(f"**{feature}**: {explanation} ({importance:.2f})")
            
            st.markdown("""
            LIME helps explain individual predictions by showing which features in that specific 
            transaction contributed to the model's decision. This is particularly useful for understanding 
            why a particular transaction was flagged as fraudulent.
            """)

def display_monitoring_dashboard(df):
    """
    Display the Monitoring Dashboard page.
    Shows real-time statistics and trends about fraud detection predictions.
    """
    st.header("Fraud Monitoring Dashboard")
    
    st.markdown("""
    This dashboard provides real-time monitoring of fraud detection metrics and trends. 
    Track key performance indicators, alert patterns, and system health all in one place.
    """)
    
    # Dashboard settings
    with st.expander("Dashboard Settings"):
        col1, col2, col3 = st.columns(3)
        with col1:
            time_period = st.selectbox(
                "Time Period", 
                ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Last Quarter", "Last Year"]
            )
        with col2:
            refresh_rate = st.selectbox(
                "Auto-refresh Rate", 
                ["Off", "30 seconds", "1 minute", "5 minutes", "15 minutes", "1 hour"]
            )
        with col3:
            threshold = st.slider(
                "Fraud Alert Threshold", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.5,
                step=0.05
            )
    
    # Initialize or update session state with simulated data if needed
    if 'monitoring_data' not in st.session_state:
        # Generate some simulated monitoring data
        st.session_state.monitoring_data = generate_monitoring_data()
    
    # KPI metrics in cards
    st.subheader("Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate KPIs from the monitoring data
    transactions = st.session_state.monitoring_data['transaction_counts'].sum()
    fraud_count = st.session_state.monitoring_data['fraud_counts'].sum()
    fraud_rate = (fraud_count / transactions) * 100
    avg_response = st.session_state.monitoring_data['response_time'].mean()
    
    # Format and display KPIs
    col1.metric(
        "Total Transactions", 
        f"{transactions:,}",
        delta="+12.4%",
        delta_color="normal"
    )
    
    col2.metric(
        "Fraud Rate", 
        f"{fraud_rate:.2f}%",
        delta="-0.8%",
        delta_color="inverse"  # Lower fraud rate is better
    )
    
    col3.metric(
        "False Positive Rate", 
        "2.3%",
        delta="-0.5%",
        delta_color="inverse"  # Lower false positive rate is better
    )
    
    col4.metric(
        "Avg. Response Time", 
        f"{avg_response:.1f}ms",
        delta="-10.2ms",
        delta_color="inverse"  # Lower response time is better
    )
    
    # Main Dashboard Tabs
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs([
        "Transaction Activity", 
        "Fraud Trends", 
        "Alert Management",
        "System Health"
    ])
    
    # Tab 1: Transaction Activity
    with tab1:
        st.subheader("Transaction Volume")
        
        # Transaction volume over time
        fig = plt.figure(figsize=(10, 5))
        plt.plot(
            st.session_state.monitoring_data['date'], 
            st.session_state.monitoring_data['transaction_counts'], 
            'b-', 
            linewidth=2
        )
        plt.fill_between(
            st.session_state.monitoring_data['date'], 
            st.session_state.monitoring_data['transaction_counts'], 
            alpha=0.2
        )
        plt.grid(True, alpha=0.3)
        plt.title('Transaction Volume Over Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        
        # Transaction by channel
        with col1:
            st.subheader("Transactions by Channel")
            channels = ['Online', 'In-store', 'Mobile App', 'ATM', 'Phone']
            values = [42, 28, 18, 8, 4]  # Percentages
            
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(values, labels=channels, autopct='%1.1f%%', startangle=90, 
                   colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
            ax.axis('equal')
            st.pyplot(fig)
            
        # Transaction by time of day
        with col2:
            st.subheader("Transactions by Time of Day")
            hours = np.arange(24)
            tx_by_hour = [
                12, 8, 5, 3, 2, 4, 10, 25, 45, 55, 
                63, 70, 72, 68, 65, 60, 58, 62, 55, 45, 
                38, 30, 22, 15
            ]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(hours, tx_by_hour, color='#5499C7')
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel('Number of Transactions')
            ax.set_xticks(np.arange(0, 24, 2))
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
    
    # Tab 2: Fraud Trends
    with tab2:
        st.subheader("Fraud Detection Overview")
        
        # Fraud rate over time
        fig = plt.figure(figsize=(10, 5))
        fraud_rate = (st.session_state.monitoring_data['fraud_counts'] / 
                    st.session_state.monitoring_data['transaction_counts']) * 100
        
        plt.plot(
            st.session_state.monitoring_data['date'], 
            fraud_rate, 
            'r-', 
            linewidth=2
        )
        plt.fill_between(
            st.session_state.monitoring_data['date'], 
            fraud_rate, 
            alpha=0.2, 
            color='red'
        )
        plt.grid(True, alpha=0.3)
        plt.title('Fraud Rate Over Time (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        
        # Fraud by category
        with col1:
            st.subheader("Fraud by Category")
            
            categories = ['Card Not Present', 'Stolen Card', 'Account Takeover', 
                         'Application Fraud', 'Other']
            values = [45, 25, 15, 10, 5]  # Percentages
            
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(values, labels=categories, autopct='%1.1f%%', startangle=90,
                  colors=['#ff9999', '#ff6666', '#ff3333', '#cc0000', '#990000'])
            ax.axis('equal')
            st.pyplot(fig)
        
        # Geographical fraud distribution
        with col2:
            st.subheader("Geographical Fraud Distribution")
            
            # Simplified world map with fraud hotspots
            from matplotlib.patches import Rectangle
            
            fig, ax = plt.subplots(figsize=(10, 6))
            img = plt.imread('https://raw.githubusercontent.com/matplotlib/basemap/main/examples/st-helens.jpg')  # Placeholder image
            ax.imshow(img, extent=[-180, 180, -90, 90], alpha=0.8)
            
            # Simulated hotspots (longitude, latitude, intensity)
            hotspots = [
                (-74, 40.7, 85),   # New York
                (-0.1, 51.5, 75),  # London
                (116.4, 39.9, 65),  # Beijing
                (103.8, 1.4, 55),   # Singapore
                (55.3, 25.3, 60),   # Dubai
                (28.0, -26.2, 50),  # Johannesburg
            ]
            
            for lon, lat, size in hotspots:
                ax.scatter(lon, lat, s=size*5, c='red', alpha=0.6, edgecolors='k')
            
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('Fraud Hotspots Worldwide')
            
            st.pyplot(fig)
        
        # Fraud patterns
        st.subheader("Emerging Fraud Patterns")
        
        pattern_data = {
            "Pattern": ["Unusual Transaction Timing", "Cross-border Transactions", 
                       "Multiple Failed Attempts", "Velocity Changes", "Amount Anomalies"],
            "Detection Rate": [92, 85, 80, 78, 72],
            "False Positive Rate": [3.2, 5.1, 2.8, 4.2, 6.5],
            "Trend": ["↑", "→", "↑", "↓", "→"],
        }
        
        st.dataframe(pd.DataFrame(pattern_data), use_container_width=True)
        
    # Tab 3: Alert Management
    with tab3:
        st.subheader("Fraud Alerts")
        
        # Alert filters
        col1, col2, col3 = st.columns(3)
        with col1:
            alert_status = st.selectbox(
                "Alert Status",
                ["All", "New", "In Review", "Resolved", "False Positive"]
            )
        with col2:
            alert_priority = st.selectbox(
                "Priority",
                ["All", "Critical", "High", "Medium", "Low"]
            )
        with col3:
            alert_date = st.date_input(
                "Date Range",
                value=(datetime.now() - timedelta(days=7), datetime.now())
            )
        
        # Sample alert data
        alert_data = {
            "Alert ID": ["ALT-" + str(i).zfill(6) for i in range(1, 8)],
            "Timestamp": [
                "2025-03-20 00:42:13", "2025-03-19 22:15:47", "2025-03-19 18:30:22",
                "2025-03-19 14:12:05", "2025-03-19 10:45:33", "2025-03-19 08:17:19",
                "2025-03-18 23:50:41"
            ],
            "Transaction ID": ["TX-" + str(i).zfill(8) for i in range(10001, 10008)],
            "Amount": ["$1,247.99", "$85.23", "$2,500.00", "$432.10", "$950.75", "$123.45", "$5,000.00"],
            "Risk Score": [0.92, 0.85, 0.78, 0.75, 0.72, 0.68, 0.67],
            "Status": ["New", "In Review", "New", "Resolved", "False Positive", "Resolved", "In Review"],
            "Priority": ["Critical", "High", "High", "Medium", "Low", "Medium", "Critical"]
        }
        
        alert_df = pd.DataFrame(alert_data)
        
        # Apply filters
        if alert_status != "All":
            alert_df = alert_df[alert_df["Status"] == alert_status]
        if alert_priority != "All":
            alert_df = alert_df[alert_df["Priority"] == alert_priority]
        
        # Color-code the risk scores
        def color_risk_score(val):
            if val >= 0.8:
                return 'background-color: #FFCCCC'
            elif val >= 0.7:
                return 'background-color: #FFEECC'
            else:
                return 'background-color: #EEFFCC'
        
        styled_df = alert_df.style.applymap(
            lambda x: color_risk_score(x) if isinstance(x, float) else '',
            subset=['Risk Score']
        )
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Alert summary
        st.subheader("Alert Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Alert resolution time
            st.markdown("**Average Alert Resolution Time**")
            
            resolution_data = {
                "Priority": ["Critical", "High", "Medium", "Low"],
                "Time (minutes)": [12, 45, 120, 240]
            }
            
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(
                resolution_data["Priority"], 
                resolution_data["Time (minutes)"],
                color=['#D98880', '#F5B041', '#F9E79F', '#ABEBC6']
            )
            
            ax.set_ylabel('Resolution Time (minutes)')
            ax.set_title('Average Alert Resolution Time by Priority')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                        f'{height:.0f}m',
                        ha='center', va='bottom')
            
            st.pyplot(fig)
            
        with col2:
            # Alert disposition
            st.markdown("**Alert Disposition**")
            
            disposition_data = {
                "Category": ["True Positive", "False Positive", "Under Investigation"],
                "Percentage": [68, 24, 8]
            }
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.pie(
                disposition_data["Percentage"],
                labels=disposition_data["Category"],
                autopct='%1.1f%%',
                startangle=90,
                colors=['#5DADE2', '#F4D03F', '#CACFD2']
            )
            ax.axis('equal')
            
            st.pyplot(fig)
    
    # Tab 4: System Health
    with tab4:
        st.subheader("System Performance")
        
        # Model performance metrics
        col1, col2 = st.columns(2)
        
        with col1:
            # Model response time
            st.markdown("**Model Response Time (ms)**")
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(
                st.session_state.monitoring_data['date'],
                st.session_state.monitoring_data['response_time'],
                'g-', linewidth=2
            )
            ax.fill_between(
                st.session_state.monitoring_data['date'],
                st.session_state.monitoring_data['response_time'],
                alpha=0.2, color='green'
            )
            ax.grid(True, alpha=0.3)
            ax.set_title('Model Response Time Trend')
            ax.set_ylabel('Time (ms)')
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            plt.tight_layout()
            
            st.pyplot(fig)
            
        with col2:
            # System uptime
            st.markdown("**System Uptime and Reliability**")
            
            uptime_data = {
                "Component": ["ML Model Service", "API Gateway", "Database", "Frontend UI"],
                "Uptime": [99.98, 99.99, 99.95, 99.97]
            }
            
            fig, ax = plt.subplots(figsize=(10, 4))
            bars = ax.barh(
                uptime_data["Component"],
                uptime_data["Uptime"],
                color='#5DADE2'
            )
            ax.set_xlim(99.5, 100)  # Focus on the range that matters for uptime
            ax.set_xlabel('Uptime (%)')
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels at the end of bars
            for bar in bars:
                width = bar.get_width()
                ax.text(width - 0.1, bar.get_y() + bar.get_height()/2.,
                        f'{width:.2f}%',
                        ha='right', va='center', color='white', fontweight='bold')
            
            st.pyplot(fig)
        
        # Model drift
        st.subheader("Model Drift Monitoring")
        
        # Generate some drift data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        accuracy = [0.956 - i*0.002 + np.random.uniform(-0.002, 0.002) for i in range(30)]
        precision = [0.923 - i*0.0015 + np.random.uniform(-0.003, 0.003) for i in range(30)]
        recall = [0.912 - i*0.0025 + np.random.uniform(-0.004, 0.002) for i in range(30)]
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(dates, accuracy, 'b-', label='Accuracy')
        ax.plot(dates, precision, 'g-', label='Precision')
        ax.plot(dates, recall, 'r-', label='Recall')
        
        # Add threshold lines
        ax.axhline(y=0.92, color='gray', linestyle='--', alpha=0.7, label='Retraining Threshold')
        
        ax.set_title('Model Performance Metrics Over Time')
        ax.set_ylabel('Score')
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Threshold breached alert
        if min(recall) < 0.88:
            st.warning("⚠️ Recall has dropped below threshold (0.88). Consider retraining the model.")
        
        # Upcoming retraining
        st.info("📅 Next scheduled model retraining: 2025-04-15")

def generate_monitoring_data():
    """
    Generate sample monitoring data for the dashboard.
    
    Returns:
        pandas.DataFrame: DataFrame with simulated monitoring data.
    """
    # Create a date range for the last 30 days
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    
    # Generate simulated transaction counts (with weekly pattern)
    base_count = 15000
    day_of_week_factor = [0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 0.6]  # Mon-Sun
    
    transaction_counts = []
    for date in dates:
        day_factor = day_of_week_factor[date.dayofweek]
        count = int(base_count * day_factor * (1 + 0.1 * np.sin(date.day / 15)))
        count += np.random.randint(-500, 500)  # Add some noise
        transaction_counts.append(max(0, count))
    
    # Generate fraud counts (roughly 0.2% to 0.5% of transactions)
    fraud_counts = []
    for count in transaction_counts:
        fraud_rate = np.random.uniform(0.002, 0.005)
        fraud_count = int(count * fraud_rate)
        fraud_counts.append(fraud_count)
    
    # Generate response times (50-70ms with some spikes)
    response_time = []
    for _ in range(len(dates)):
        base_time = np.random.uniform(50, 70)
        # Add occasional spikes
        if np.random.random() < 0.1:
            base_time *= np.random.uniform(1.2, 1.5)
        response_time.append(base_time)
    
    # Create the dataframe
    monitoring_data = pd.DataFrame({
        'date': dates,
        'transaction_counts': transaction_counts,
        'fraud_counts': fraud_counts,
        'response_time': response_time
    })
    
    return monitoring_data

def display_inference_page(df):
    """
    Display the Real-time Inference page.
    Allows making predictions on new transactions to detect fraud.
    """
    st.header("Real-time Inference")
    
    st.info("This page allows you to make predictions on new transactions to detect fraud.")
    
    # Input form
    st.subheader("Enter Transaction Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=10.0)
        capture_method = st.selectbox("Capture Method", ["Online", "POS", "ATM", "Manual"])
        payment_method = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "Wire Transfer"])
        card_brand = st.selectbox("Card Brand", ["Visa", "Mastercard", "American Express", "Discover"])
    
    with col2:
        merchant_category = st.number_input("Merchant Category Code (MCC)", min_value=1000, max_value=9999, value=5411)
        country = st.selectbox("Transaction Country", ["USA", "UK", "Canada", "Australia", "Other"])
        is_international = st.checkbox("International Transaction", value=False)
        transaction_time = st.time_input("Transaction Time", value=None)
    
    # Add advanced options expander for more inputs
    with st.expander("Advanced Transaction Details"):
        col1, col2 = st.columns(2)
        
        with col1:
            customer_id = st.number_input("Customer ID", min_value=1, value=12345)
            account_age = st.slider("Account Age (months)", min_value=1, max_value=120, value=24)
            previous_transactions = st.number_input("Previous Transaction Count", min_value=0, value=15)
            avg_transaction_amount = st.number_input("Avg Transaction Amount ($)", min_value=0.0, value=75.0)
        
        with col2:
            device_type = st.selectbox("Device Type", ["Mobile", "Desktop", "Tablet", "Other"])
            browser = st.selectbox("Browser", ["Chrome", "Safari", "Firefox", "Edge", "Other"])
            ip_risk = st.slider("IP Risk Score", min_value=0.0, max_value=1.0, value=0.2)
            shipping_billing_match = st.checkbox("Shipping/Billing Address Match", value=True)
    
    # Model selection
    st.subheader("Model Configuration")
    model_name = st.selectbox(
        "Select model for prediction", 
        ["Ensemble (Recommended)", "XGBoost", "Random Forest", "Logistic Regression"]
    )
    
    explain_results = st.checkbox("Generate explanation using LIME", value=True)
    
    # Prediction button
    if st.button("Check for Fraud"):
        with st.spinner("Analyzing transaction..."):
            # In a real implementation, we would create a feature vector and call the model
            # Here we'll simulate the prediction
            
            # Feature importance would come from the model
            simulated_features = {
                "Transaction Amount": 0.32,
                f"Capture Method: {capture_method}": 0.15,
                f"Payment Method: {payment_method}": 0.08,
                "International Transaction": 0.27 if is_international else -0.05,
                f"Merchant Category: {merchant_category}": 0.12,
                "IP Risk": ip_risk * 0.4,
                "Previous Transactions": -0.18 if previous_transactions > 10 else 0.22,
                "Account Age": -0.14 if account_age > 12 else 0.19,
                "Shipping/Billing Match": -0.17 if shipping_billing_match else 0.25
            }
            
            # Sort feature importance
            sorted_features = dict(sorted(simulated_features.items(), key=lambda x: abs(x[1]), reverse=True))
            
            # Simulate a prediction
            if model_name == "Ensemble (Recommended)":
                fraud_probability = 0.82 if amount > 500 or is_international else 0.15
            elif model_name == "XGBoost":
                fraud_probability = 0.78 if amount > 500 or is_international else 0.18
            elif model_name == "Random Forest":
                fraud_probability = 0.75 if amount > 500 or is_international else 0.22
            else:  # Logistic Regression
                fraud_probability = 0.68 if amount > 500 or is_international else 0.25
            
            # Add some randomness to make it interesting
            import random
            fraud_probability = min(1.0, max(0.0, fraud_probability + random.uniform(-0.1, 0.1)))
            
            is_fraud = fraud_probability > 0.5
        
        # Display result
        st.subheader("Analysis Result")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if is_fraud:
                st.error("⚠️ FRAUD ALERT DETECTED!")
                st.markdown(f"Fraud Probability: **{fraud_probability:.2%}**")
                st.markdown("Risk: **HIGH**")
                st.markdown("Recommendation: Block transaction and investigate")
            else:
                st.success("✅ Safe Transaction")
                st.markdown(f"Fraud Probability: **{fraud_probability:.2%}**")
                st.markdown("Risk: **LOW**")
                st.markdown("Recommendation: Approve transaction")
        
        # Fraud score gauge
        fig, ax = plt.subplots(figsize=(10, 3))
        
        # Create gauge visualization
        gauge_colors = [(0.3, 0.7, 0.3), (0.7, 0.7, 0.3), (0.7, 0.3, 0.3)]  # Green, Yellow, Red
        
        # Background gauge (gray)
        ax.add_patch(plt.matplotlib.patches.Rectangle((0, 0), 1, 0.3, color='lightgray'))
        
        # Colored sections
        ax.add_patch(plt.matplotlib.patches.Rectangle((0, 0), 0.33, 0.3, color=gauge_colors[0]))
        ax.add_patch(plt.matplotlib.patches.Rectangle((0.33, 0), 0.33, 0.3, color=gauge_colors[1]))
        ax.add_patch(plt.matplotlib.patches.Rectangle((0.66, 0), 0.34, 0.3, color=gauge_colors[2]))
        
        # Fraud probability marker
        marker_x = fraud_probability
        ax.add_patch(plt.matplotlib.patches.Circle((marker_x, 0.15), 0.1, color='white', zorder=10))
        ax.add_patch(plt.matplotlib.patches.Circle((marker_x, 0.15), 0.08, color='black', zorder=11))
        
        # Add labels
        ax.text(0.15, 0.4, "Low Risk", ha='center')
        ax.text(0.5, 0.4, "Medium Risk", ha='center')
        ax.text(0.85, 0.4, "High Risk", ha='center')
        
        # Set axis limits and remove ticks
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.5)
        ax.axis('off')
        
        st.pyplot(fig)
        
        if explain_results:
            st.subheader("LIME Explanation")
            
            st.markdown("""
            LIME (Local Interpretable Model-agnostic Explanations) helps understand
            why this specific transaction received its fraud score by approximating the 
            complex model with a simpler, interpretable one around this prediction.
            """)
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Create LIME visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Feature impacts
                features = list(sorted_features.keys())[:7]  # Top 7 features
                importance = list(sorted_features.values())[:7]
                
                # Determine colors based on whether feature increases or decreases fraud probability
                colors = ['#ff6666' if imp > 0 else '#66b3ff' for imp in importance]
                
                y_pos = range(len(features))
                ax.barh(y_pos, importance, color=colors)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(features)
                ax.set_xlabel('Impact on Fraud Probability')
                ax.set_title('Why this prediction was made')
                
                # Add a reference line
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                
                # Add a legend
                ax.barh(-1, 0, color='#ff6666', label='Increases Fraud Risk')
                ax.barh(-1, 0, color='#66b3ff', label='Decreases Fraud Risk')
                ax.legend(loc='upper right')
                
                st.pyplot(fig)
            
            with col2:
                # Decision summary
                st.markdown("### Decision Summary")
                
                top_risk_factors = [f for f, i in sorted_features.items() if i > 0][:3]
                top_safe_factors = [f for f, i in sorted_features.items() if i < 0][:3]
                
                if top_risk_factors:
                    st.markdown("**Top risk factors:**")
                    for f in top_risk_factors:
                        st.markdown(f"- {f}")
                
                if top_safe_factors:
                    st.markdown("**Top safety factors:**")
                    for f in top_safe_factors:
                        st.markdown(f"- {f}")
            
            # Feature importance explanation
            st.subheader("Feature Importance Explanations")
            
            explanations = {
                "Transaction Amount": "Higher transaction amounts are associated with increased fraud risk.",
                "Capture Method: Online": "Online transactions have higher fraud rates than in-person.",
                "Payment Method: Credit Card": "Credit cards have different fraud patterns than debit cards.",
                "International Transaction": "Cross-border transactions show elevated fraud risk.",
                "Merchant Category": "Certain merchant categories are more susceptible to fraud.",
                "IP Risk": "IP addresses associated with suspicious activity increase risk.",
                "Previous Transactions": "More transaction history typically decreases fraud risk.",
                "Account Age": "Newer accounts have higher fraud likelihood.",
                "Shipping/Billing Match": "Address mismatches are associated with fraud."
            }
            
            for feature, importance in list(sorted_features.items())[:5]:
                base_feature = feature.split(':')[0].strip()
                if base_feature in explanations:
                    color = "red" if importance > 0 else "blue"
                    st.markdown(f"**{feature}** ({importance:.2f}): <span style='color:{color}'>{explanations[base_feature]}</span>", unsafe_allow_html=True)
                
            st.info("""
            Note: This LIME explanation is a simulation for demonstration purposes. 
            In a production system, LIME would analyze the model's behavior by perturbing 
            the input and observing the effect on the output.
            """)
            
        # Risk factors (simulated)
        st.subheader("Risk Factors")
        
        risk_factors = [
            ("Transaction amount", "Medium", 0.3),
            ("Transaction time", "Low", 0.1),
            ("Location", "High" if is_international else "Low", 0.7 if is_international else 0.1),
            ("Customer history", "Medium", 0.4),
            ("Purchase pattern", "High" if is_fraud else "Low", 0.8 if is_fraud else 0.2)
        ]
        
        risk_df = pd.DataFrame(risk_factors, columns=["Factor", "Level", "Score"])
        
        # Bar chart for risk factors
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(risk_df["Factor"], risk_df["Score"], color=['#ff9999' if x > 0.5 else '#99ff99' for x in risk_df["Score"]])
        ax.set_xlim(0, 1)
        ax.set_xlabel('Risk Level')
        ax.set_title('Contribution to Fraud Risk')
        
        # Add values to bars
        for i, bar in enumerate(bars):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{risk_df["Score"][i]:.2f}', 
                    va='center', fontweight='bold')
        
        st.pyplot(fig)

# Run the application
if __name__ == "__main__":
    main() 
