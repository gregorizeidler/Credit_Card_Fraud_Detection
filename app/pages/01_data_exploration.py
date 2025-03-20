"""
Data Exploration Page for the fraud detection application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add root directory to path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.preprocessing import load_config, load_data

# Page configuration
st.set_page_config(
    page_title="Data Exploration - Fraud Detection", 
    page_icon="📊",
    layout="wide"
)

# Page title
st.title("Advanced Data Exploration")
st.markdown("""
    This page allows a detailed exploration of credit card transaction data.
    Analyze distributions, correlations, and trends to better understand fraud patterns.
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

# Tabs for different analyses
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Temporal Analysis", "Merchant Analysis", "Correlations"])

with tab1:
    st.header("Data Overview")
    
    # Statistical summary
    st.subheader("Statistical Summary")
    
    # Column selection for analysis
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    selected_cols = st.multiselect("Select columns for analysis:", numeric_cols, default=numeric_cols[:5])
    
    if selected_cols:
        st.dataframe(df[selected_cols].describe())
        
        # Histograms
        st.subheader("Distributions")
        cols_per_row = 2
        total_rows = (len(selected_cols) + cols_per_row - 1) // cols_per_row
        
        for row in range(total_rows):
            cols = st.columns(cols_per_row)
            for i in range(cols_per_row):
                idx = row * cols_per_row + i
                if idx < len(selected_cols):
                    col = selected_cols[idx]
                    with cols[i]:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        if 'Fraud Indicator' in df.columns:
                            sns.histplot(data=df, x=col, hue='Fraud Indicator', multiple='stack', bins=20, ax=ax)
                        else:
                            sns.histplot(data=df, x=col, bins=20, ax=ax)
                        plt.title(f'Distribution of {col}')
                        plt.xticks(rotation=45)
                        st.pyplot(fig)

with tab2:
    st.header("Temporal Analysis")
    
    # Check date/time columns
    datetime_cols = []
    if 'Transaction DateTime' in df.columns:
        datetime_cols.append('Transaction DateTime')
    elif 'Transaction Date' in df.columns:
        datetime_cols.append('Transaction Date')
    elif 'Exact Date' in df.columns:
        datetime_cols.append('Exact Date')
    
    if datetime_cols:
        # Select date/time column
        datetime_col = st.selectbox("Select date/time column:", datetime_cols)
        
        # Convert to datetime if not already
        if pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
            df_time = df.copy()
        else:
            df_time = df.copy()
            df_time[datetime_col] = pd.to_datetime(df_time[datetime_col], errors='coerce')
        
        # Extract temporal components
        df_time['hour'] = df_time[datetime_col].dt.hour
        df_time['day_of_week'] = df_time[datetime_col].dt.dayofweek
        df_time['month'] = df_time[datetime_col].dt.month
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Transactions by hour of day
            st.subheader("Transactions by Hour of Day")
            
            hour_counts = df_time.groupby(['hour', 'Fraud Indicator']).size().unstack(fill_value=0)
            fig, ax = plt.subplots(figsize=(10, 6))
            hour_counts.plot(kind='bar', stacked=True, ax=ax, 
                            color=['#4CAF50', '#F44336'] if 'Fraud Indicator' in df.columns else '#1f77b4')
            plt.title('Transactions by Hour of Day')
            plt.xlabel('Hour')
            plt.ylabel('Number of Transactions')
            plt.xticks(rotation=45)
            plt.legend(['Not Fraud', 'Fraud'] if 'Fraud Indicator' in df.columns else [])
            st.pyplot(fig)
        
        with col2:
            # Transactions by day of week
            st.subheader("Transactions by Day of Week")
            
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            df_time['day_name'] = df_time['day_of_week'].apply(lambda x: day_names[x])
            
            day_counts = df_time.groupby(['day_name', 'Fraud Indicator']).size().unstack(fill_value=0)
            # Reorder days of week
            if not day_counts.empty:
                day_counts = day_counts.reindex(day_names)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                day_counts.plot(kind='bar', stacked=True, ax=ax, 
                                color=['#4CAF50', '#F44336'] if 'Fraud Indicator' in df.columns else '#1f77b4')
                plt.title('Transactions by Day of Week')
                plt.xlabel('Day of Week')
                plt.ylabel('Number of Transactions')
                plt.xticks(rotation=45)
                plt.legend(['Not Fraud', 'Fraud'] if 'Fraud Indicator' in df.columns else [])
                st.pyplot(fig)
        
        # Temporal trend
        st.subheader("Temporal Trend of Transactions")
        timeframe = st.selectbox("Select aggregation period:", 
                                ["Daily", "Weekly", "Monthly"])
        
        # Define aggregation frequency
        if timeframe == "Daily":
            freq = 'D'
            title = "Daily Trend"
        elif timeframe == "Weekly":
            freq = 'W'
            title = "Weekly Trend"
        else:
            freq = 'M'
            title = "Monthly Trend"
        
        # Group by period
        df_time.set_index(datetime_col, inplace=True)
        
        # Calculate transaction count and fraud rate by period
        if 'Fraud Indicator' in df.columns:
            trend_data = df_time.groupby(pd.Grouper(freq=freq)).agg(
                total_transactions=('Fraud Indicator', 'count'),
                fraud_transactions=('Fraud Indicator', 'sum')
            )
            trend_data['fraud_rate'] = trend_data['fraud_transactions'] / trend_data['total_transactions']
            
            # Plot trend
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Plot number of transactions
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Number of Transactions', color='tab:blue')
            ax1.plot(trend_data.index, trend_data['total_transactions'], color='tab:blue')
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            
            # Create second y-axis for fraud rate
            ax2 = ax1.twinx()
            ax2.set_ylabel('Fraud Rate', color='tab:red')
            ax2.plot(trend_data.index, trend_data['fraud_rate'], color='tab:red')
            ax2.tick_params(axis='y', labelcolor='tab:red')
            
            plt.title(f"{title} of Transactions and Fraud Rate")
            fig.tight_layout()
            st.pyplot(fig)
        else:
            # If there's no fraud indicator, just plot number of transactions
            trend_data = df_time.groupby(pd.Grouper(freq=freq)).size()
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(trend_data.index, trend_data.values)
            ax.set_xlabel('Date')
            ax.set_ylabel('Number of Transactions')
            plt.title(f"{title} of Transactions")
            st.pyplot(fig)
    else:
        st.warning("No date/time column found in the data.")

with tab3:
    st.header("Merchant Analysis")
    
    merchant_cols = []
    if 'Merchant ID' in df.columns:
        merchant_cols.append('Merchant ID')
    if 'Merchant Category Code (MCC)' in df.columns:
        merchant_cols.append('Merchant Category Code (MCC)')
    if 'Merchant City' in df.columns:
        merchant_cols.append('Merchant City')
    if 'Merchant State' in df.columns:
        merchant_cols.append('Merchant State')
    
    if merchant_cols:
        # Select merchant column
        merchant_col = st.selectbox("Select merchant column:", merchant_cols)
        
        # Show top merchants by transaction volume
        st.subheader(f"Top 10 {merchant_col} by Transaction Volume")
        
        # Group by merchant
        merchant_counts = df.groupby(merchant_col).size().sort_values(ascending=False).head(10)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        merchant_counts.plot(kind='bar', ax=ax)
        plt.title(f'Top 10 {merchant_col} by Transaction Volume')
        plt.xlabel(merchant_col)
        plt.ylabel('Number of Transactions')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Show fraud rate by merchant, if available
        if 'Fraud Indicator' in df.columns:
            st.subheader(f"Top 10 {merchant_col} by Fraud Rate")
            
            # Filter for merchants with at least 5 transactions
            merchant_fraud = df.groupby(merchant_col).agg(
                total=('Fraud Indicator', 'count'),
                fraud=('Fraud Indicator', 'sum')
            )
            merchant_fraud['fraud_rate'] = merchant_fraud['fraud'] / merchant_fraud['total']
            
            # Filter merchants with at least 5 transactions
            merchant_fraud = merchant_fraud[merchant_fraud['total'] >= 5]
            
            # Top 10 merchants with highest fraud rate
            top_fraud = merchant_fraud.sort_values('fraud_rate', ascending=False).head(10)
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(top_fraud.index, top_fraud['fraud_rate'])
            plt.title(f'Top 10 {merchant_col} by Fraud Rate')
            plt.xlabel(merchant_col)
            plt.ylabel('Fraud Rate')
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            # Table with results
            st.dataframe(top_fraud)
            
            # Heatmap by location, if available
            if 'Merchant City' in df.columns and 'Merchant State' in df.columns:
                st.subheader("Fraud Heatmap by Location")
                
                # Group by city and state
                location_fraud = df.groupby(['Merchant State', 'Merchant City']).agg(
                    total=('Fraud Indicator', 'count'),
                    fraud=('Fraud Indicator', 'sum')
                )
                location_fraud['fraud_rate'] = location_fraud['fraud'] / location_fraud['total']
                
                # Filter locations with at least 3 transactions
                location_fraud = location_fraud[location_fraud['total'] >= 3]
                
                # Convert to matrix for heatmap
                pivot = location_fraud.pivot_table(
                    values='fraud_rate', 
                    index='Merchant State',
                    columns='Merchant City',
                    fill_value=0
                )
                
                # Plot heatmap
                if not pivot.empty:
                    fig, ax = plt.subplots(figsize=(12, 8))
                    sns.heatmap(pivot, annot=True, cmap='YlOrRd', ax=ax)
                    plt.title('Fraud Rate by Location')
                    st.pyplot(fig)
                else:
                    st.warning("Insufficient data to create heatmap.")
    else:
        st.warning("No merchant column found in the data.")

with tab4:
    st.header("Correlation Analysis")
    
    # Select only numeric columns
    num_df = df.select_dtypes(include=['float64', 'int64'])
    
    if not num_df.empty:
        # Calculate correlation matrix
        corr_matrix = num_df.corr()
        
        # Plot heatmap
        st.subheader("Correlation Matrix")
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    annot=True, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
        plt.title('Correlation Matrix')
        st.pyplot(fig)
        
        # Correlation with fraud indicator (if available)
        if 'Fraud Indicator' in num_df.columns:
            st.subheader("Correlation with Fraud Indicator")
            
            # Get correlations with fraud indicator
            fraud_corr = corr_matrix['Fraud Indicator'].drop('Fraud Indicator').sort_values(ascending=False)
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 8))
            fraud_corr.plot(kind='bar', ax=ax)
            plt.title('Feature Correlation with Fraud Indicator')
            plt.xlabel('Feature')
            plt.ylabel('Correlation')
            plt.xticks(rotation=45)
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            st.pyplot(fig)
            
            # Plot distributions for top 3 correlated features
            st.subheader("Distribution of Features Most Correlated with Fraud")
            
            top_features = fraud_corr.abs().sort_values(ascending=False).head(3).index.tolist()
            
            for feature in top_features:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(data=df, x=feature, hue='Fraud Indicator', bins=30, ax=ax)
                plt.title(f'Distribution of {feature} by Fraud Indicator')
                plt.xticks(rotation=45)
                st.pyplot(fig)
    else:
        st.warning("No numeric column found in the data.") 
