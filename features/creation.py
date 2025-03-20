"""
Module for creating features for credit card fraud detection.

This module contains functions to create derived features that can be useful
for identifying fraudulent transactions.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Add root directory to path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing import load_config

logger = logging.getLogger(__name__)

def create_amount_features(df):
    """
    Creates features derived from transaction amount.
    
    Args:
        df (pandas.DataFrame): DataFrame with transaction data.
        
    Returns:
        pandas.DataFrame: DataFrame with new features added.
    """
    df_new = df.copy()
    
    # Log of transaction amount (to address skewness)
    if 'Amount' in df_new.columns:
        df_new['amount_log'] = np.log1p(df_new['Amount'])
        
        # Categorize transaction amount
        df_new['amount_category'] = pd.cut(
            df_new['Amount'], 
            bins=[0, 10, 50, 100, 500, 1000, float('inf')],
            labels=['very_small', 'small', 'medium', 'large', 'very_large', 'huge']
        )
    
    return df_new

def create_temporal_features(df):
    """
    Creates temporal features from transaction date and time.
    
    Args:
        df (pandas.DataFrame): DataFrame with transaction data.
        
    Returns:
        pandas.DataFrame: DataFrame with new features added.
    """
    df_new = df.copy()
    
    # Check if datetime column exists or can be created
    datetime_col = None
    
    # Try to use existing datetime column
    if 'Transaction DateTime' in df_new.columns:
        try:
            # Convert to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(df_new['Transaction DateTime']):
                df_new['Transaction DateTime'] = pd.to_datetime(df_new['Transaction DateTime'], errors='coerce')
            datetime_col = 'Transaction DateTime'
        except Exception as e:
            print(f"Warning: Could not convert Transaction DateTime to datetime: {e}")
    
    # Try to create from separate date and time columns
    elif 'Transaction Date' in df_new.columns and 'Transaction Time' in df_new.columns:
        try:
            df_new['Transaction DateTime'] = pd.to_datetime(
                df_new['Transaction Date'] + ' ' + df_new['Transaction Time'], 
                errors='coerce'
            )
            datetime_col = 'Transaction DateTime'
        except Exception as e:
            print(f"Warning: Could not create datetime from date and time columns: {e}")
    
    # Alternative column names
    elif 'Date' in df_new.columns:
        try:
            df_new['Transaction DateTime'] = pd.to_datetime(df_new['Date'], errors='coerce')
            datetime_col = 'Transaction DateTime'
        except Exception as e:
            print(f"Warning: Could not convert Date to datetime: {e}")
    
    elif 'Exact Date' in df_new.columns:
        try:
            df_new['Transaction DateTime'] = pd.to_datetime(df_new['Exact Date'], errors='coerce')
            datetime_col = 'Transaction DateTime'
        except Exception as e:
            print(f"Warning: Could not convert Exact Date to datetime: {e}")
    
    # If we have datetime information and it's valid, create time-based features
    if datetime_col and not df_new[datetime_col].isna().all():
        # Extract temporal components
        try:
            df_new['hour_of_day'] = df_new[datetime_col].dt.hour
            df_new['day_of_week'] = df_new[datetime_col].dt.dayofweek
            df_new['is_weekend'] = df_new['day_of_week'].isin([5, 6]).astype(int)
            df_new['month'] = df_new[datetime_col].dt.month
            df_new['day_of_month'] = df_new[datetime_col].dt.day
            
            # Time period categories
            bins = [0, 6, 12, 18, 24]
            labels = ['early_morning', 'morning', 'afternoon', 'evening']
            df_new['period_of_day'] = pd.cut(df_new['hour_of_day'], bins=bins, labels=labels, include_lowest=True)
            
            # Flag for night time (higher risk)
            df_new['is_night_time'] = ((df_new['hour_of_day'] >= 0) & (df_new['hour_of_day'] < 6) | 
                                      (df_new['hour_of_day'] >= 22)).astype(int)
        except Exception as e:
            print(f"Warning: Error extracting temporal features: {e}")
    else:
        print("Warning: No valid datetime column found. Temporal features will not be created.")
    
    return df_new

def create_location_features(df):
    """
    Create location-based features that capture fraud patterns by merchant location.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing transaction data
    
    Returns:
    --------
    pandas.DataFrame
        Dataframe with original columns plus new location-based features
    """
    df_location = df.copy()
    
    # Skip if location data not available
    if not all(col in df_location.columns for col in ['Merchant City', 'Merchant State']):
        logger.warning("Merchant location columns missing, skipping location feature creation")
        return df_location
    
    # Create fraud rate by city
    if 'Fraud Indicator' in df_location.columns:
        # Calculate fraud rate by city
        city_fraud = df_location.groupby('Merchant City')['Fraud Indicator'].agg(['mean', 'count']).reset_index()
        city_fraud.columns = ['Merchant City', 'City Fraud Rate', 'Transaction Count']
        
        # Only include cities with sufficient data to avoid overfitting
        min_transactions = 5  # Minimum number of transactions per city
        city_fraud = city_fraud[city_fraud['Transaction Count'] >= min_transactions]
        
        # Merge back to original dataframe
        df_location = df_location.merge(city_fraud[['Merchant City', 'City Fraud Rate']], 
                                         on='Merchant City', how='left')
        
        # Fill missing values (cities with few transactions)
        df_location['City Fraud Rate'].fillna(df_location['Fraud Indicator'].mean(), inplace=True)
        
        # Create city risk categories
        # Handle potential duplicate bin edges by using duplicates='drop'
        city_fraud_rate = df_location['City Fraud Rate']
        
        try:
            city_risk = pd.qcut(
                city_fraud_rate,
                q=[0, 0.25, 0.5, 0.75, 1.0],
                labels=['low_risk', 'medium_low_risk', 'medium_high_risk', 'high_risk'],
                duplicates='drop'
            )
            df_location['City Risk Category'] = city_risk
        except ValueError:
            # If we can't create quartiles, use manual thresholds
            logger.warning("Could not create city risk quartiles, using manual thresholds")
            bins = [0, 0.01, 0.05, 0.10, 1.0]
            labels = ['low_risk', 'medium_low_risk', 'medium_high_risk', 'high_risk']
            df_location['City Risk Category'] = pd.cut(
                city_fraud_rate, 
                bins=bins, 
                labels=labels,
                duplicates='drop'
            )
        
        # Calculate fraud rate by state
        state_fraud = df_location.groupby('Merchant State')['Fraud Indicator'].agg(['mean', 'count']).reset_index()
        state_fraud.columns = ['Merchant State', 'State Fraud Rate', 'Transaction Count']
        
        # Only include states with sufficient data
        min_transactions = 10  # Minimum number of transactions per state
        state_fraud = state_fraud[state_fraud['Transaction Count'] >= min_transactions]
        
        # Merge back to original dataframe
        df_location = df_location.merge(state_fraud[['Merchant State', 'State Fraud Rate']], 
                                         on='Merchant State', how='left')
        
        # Fill missing values
        df_location['State Fraud Rate'].fillna(df_location['Fraud Indicator'].mean(), inplace=True)
        
        # Create state risk categories
        state_fraud_rate = df_location['State Fraud Rate']
        
        try:
            state_risk = pd.qcut(
                state_fraud_rate,
                q=[0, 0.25, 0.5, 0.75, 1.0],
                labels=['low_risk', 'medium_low_risk', 'medium_high_risk', 'high_risk'],
                duplicates='drop'
            )
            df_location['State Risk Category'] = state_risk
        except ValueError:
            # If we can't create quartiles, use manual thresholds
            logger.warning("Could not create state risk quartiles, using manual thresholds")
            bins = [0, 0.01, 0.05, 0.10, 1.0]
            labels = ['low_risk', 'medium_low_risk', 'medium_high_risk', 'high_risk']
            df_location['State Risk Category'] = pd.cut(
                state_fraud_rate, 
                bins=bins, 
                labels=labels,
                duplicates='drop'
            )
    
    # Create distance features if we have customer location
    if all(col in df_location.columns for col in ['Customer Zip', 'Merchant Zip']):
        # Here we would calculate the distance between customer and merchant
        # For simplicity, we'll just create a binary feature
        df_location['Different Zip'] = (df_location['Customer Zip'] != df_location['Merchant Zip']).astype(int)
    
    # Create binary flag for high-risk locations (optional)
    if 'City Risk Category' in df_location.columns:
        df_location['High Risk Location'] = df_location['City Risk Category'].apply(
            lambda x: 1 if x == 'high_risk' else 0
        )
    
    logger.info(f"Created {df_location.shape[1] - df.shape[1]} location-based features")
    return df_location

def create_customer_behavior_features(df):
    """
    Creates features based on customer behavior.
    
    Args:
        df (pandas.DataFrame): DataFrame with transaction data.
        
    Returns:
        pandas.DataFrame: DataFrame with new features added.
    """
    df_new = df.copy()
    
    if 'Card Number' in df_new.columns:
        # Sort by customer and datetime
        if 'Transaction DateTime' in df_new.columns:
            df_new = df_new.sort_values(['Card Number', 'Transaction DateTime'])
        
        # Number of transactions per customer
        df_new['customer_transaction_count'] = df_new.groupby('Card Number')['Transaction ID'].transform('count')
        
        # Average transaction amount per customer
        df_new['customer_avg_amount'] = df_new.groupby('Card Number')['Amount'].transform('mean')
        
        # Amount deviation from customer average
        df_new['amount_deviation_from_avg'] = df_new['Amount'] / df_new['customer_avg_amount']
        
        # Difference from last transaction (amount)
        df_new['amount_diff_last_transaction'] = df_new.groupby('Card Number')['Amount'].diff()
        
        # Flag if this is customer's first transaction
        df_new['is_first_transaction'] = (df_new.groupby('Card Number').cumcount() == 0).astype(int)
    
    return df_new

def create_merchant_features(df):
    """
    Creates features based on merchant information.
    
    Args:
        df (pandas.DataFrame): DataFrame with transaction data.
        
    Returns:
        pandas.DataFrame: DataFrame with new features added.
    """
    df_new = df.copy()
    
    if 'Merchant ID' in df_new.columns:
        # Number of transactions per merchant
        df_new['merchant_transaction_count'] = df_new.groupby('Merchant ID')['Transaction ID'].transform('count')
        
        # Average transaction amount per merchant
        if 'Amount' in df_new.columns:
            df_new['merchant_avg_amount'] = df_new.groupby('Merchant ID')['Amount'].transform('mean')
        
        # Merchant fraud rate (using target leakage with cross-validation in a real implementation)
        if 'Fraud Indicator' in df_new.columns:
            try:
                df_new['merchant_fraud_rate'] = df_new.groupby('Merchant ID')['Fraud Indicator'].transform('mean')
                
                # Categorize merchants by fraud rate
                # First, check if there are enough unique values for qcut
                merchant_fraud_rates = df_new['merchant_fraud_rate'].clip(0, 1)
                unique_rates = merchant_fraud_rates.unique()
                
                if len(unique_rates) >= 4:
                    try:
                        df_new['merchant_risk_category'] = pd.qcut(
                            merchant_fraud_rates, 
                            q=4, 
                            labels=['low_risk', 'medium_low_risk', 'medium_high_risk', 'high_risk'],
                            duplicates='drop'
                        )
                    except Exception as e:
                        print(f"Warning: Could not create merchant risk categories with qcut: {e}")
                        # Handle case with too few unique values
                        median = merchant_fraud_rates.median()
                        df_new['merchant_risk_category'] = pd.Series(
                            ['low_risk' if x <= median/2 else 
                             'medium_low_risk' if x <= median else
                             'medium_high_risk' if x <= median*1.5 else
                             'high_risk' for x in merchant_fraud_rates],
                            index=df_new.index
                        )
                else:
                    # For limited unique values, use manual categorization
                    median = merchant_fraud_rates.median()
                    
                    # Simple binary if very few unique values
                    if len(unique_rates) <= 2:
                        df_new['merchant_risk_category'] = pd.Series(
                            ['low_risk' if x <= median else 'high_risk' for x in merchant_fraud_rates],
                            index=df_new.index
                        )
                    else:
                        # Three categories if we have enough variation
                        df_new['merchant_risk_category'] = pd.Series(
                            ['low_risk' if x <= median*0.5 else 
                             'medium_risk' if x <= median*1.5 else
                             'high_risk' for x in merchant_fraud_rates],
                            index=df_new.index
                        )
            except Exception as e:
                print(f"Warning: Could not create merchant fraud rate features: {e}")
    
    return df_new

def create_payment_method_features(df):
    """
    Creates features based on payment method.
    
    Args:
        df (pandas.DataFrame): DataFrame with transaction data.
        
    Returns:
        pandas.DataFrame: DataFrame with new features added.
    """
    df_new = df.copy()
    
    payment_cols = ['Payment Method', 'Capture Method', 'Card Brand', 'Issuer Bank', 'Issuer Country']
    
    if all(col in df_new.columns for col in payment_cols):
        # Combination of payment method and capture method
        df_new['payment_capture_combo'] = df_new['Payment Method'] + '_' + df_new['Capture Method']
        
        # Features for each payment method combination
        for col in payment_cols:
            # Fraud rate by payment method
            if 'Fraud Indicator' in df_new.columns:
                col_fraud_rate = df_new.groupby(col)['Fraud Indicator'].mean()
                df_new[f'{col.lower()}_fraud_rate'] = df_new[col].map(col_fraud_rate)
        
        # Flag for international transactions (issuer country different)
        # In a real scenario, we would need information about the country where the transaction occurred
        if 'Issuer Country' in df_new.columns:
            df_new['is_international'] = (df_new['Issuer Country'] != 'USA').astype(int)
    
    return df_new

def create_velocity_features(df):
    """
    Creates velocity features (transaction frequency over periods).
    
    Args:
        df (pandas.DataFrame): DataFrame with transaction data.
        
    Returns:
        pandas.DataFrame: DataFrame with new features added.
    """
    df_new = df.copy()
    
    # Need time information and customer identifier
    if 'Card Number' in df_new.columns:
        # Ensure Transaction DateTime is datetime type
        datetime_col = None
        if 'Transaction DateTime' in df_new.columns:
            try:
                # Convert to datetime if it's not already
                if not pd.api.types.is_datetime64_any_dtype(df_new['Transaction DateTime']):
                    df_new['Transaction DateTime'] = pd.to_datetime(df_new['Transaction DateTime'], errors='coerce')
                datetime_col = 'Transaction DateTime'
            except Exception as e:
                print(f"Warning: Could not convert Transaction DateTime to datetime: {e}")
        
        # If we have datetime information, create time-based features
        if datetime_col and not df_new[datetime_col].isna().all():
            # Sort by customer and time
            df_new = df_new.sort_values(['Card Number', datetime_col])
            
            # Calculate time since last transaction (in minutes)
            df_new['time_since_last_transaction'] = df_new.groupby('Card Number')[datetime_col].diff().dt.total_seconds() / 60
            
            # Fill NaN (first transaction) with a large value
            df_new['time_since_last_transaction'] = df_new['time_since_last_transaction'].fillna(99999)
            
            # Flag for quick transactions (less than 30 minutes between transactions)
            df_new['is_quick_transaction'] = (df_new['time_since_last_transaction'] < 30).astype(int)
            
            try:
                # Try a simplified approach for transaction count, safer with various data formats
                df_sorted = df_new.sort_values([datetime_col])
                df_new['transaction_count_last_24h'] = 0  # Default value
                
                # For each customer, count transactions in last 24 hours
                for customer in df_new['Card Number'].unique():
                    # Get customer transactions
                    customer_df = df_sorted[df_sorted['Card Number'] == customer].copy()
                    
                    if len(customer_df) > 1:
                        # For each transaction, count how many transactions occurred in 24h before it
                        for i, row in customer_df.iterrows():
                            current_time = row[datetime_col]
                            if pd.notna(current_time):
                                time_24h_ago = current_time - pd.Timedelta(hours=24)
                                count = len(customer_df[(customer_df[datetime_col] >= time_24h_ago) & 
                                                       (customer_df[datetime_col] <= current_time)])
                                df_new.loc[i, 'transaction_count_last_24h'] = count
            except Exception as e:
                print(f"Warning: Could not calculate transaction count last 24h: {e}")
                # Create a simple count feature instead
                df_new['transaction_count_last_24h'] = df_new.groupby('Card Number')['Card Number'].transform('count')
    
    return df_new

def create_fraud_pattern_features(df):
    """
    Creates features specific to known fraud patterns.
    
    Args:
        df (pandas.DataFrame): DataFrame with transaction data.
        
    Returns:
        pandas.DataFrame: DataFrame with new features added.
    """
    df_new = df.copy()
    
    # Pattern 1: Low value transactions followed by high value ones
    if 'Card Number' in df_new.columns and 'Amount' in df_new.columns:
        if 'Transaction DateTime' in df_new.columns:
            df_new = df_new.sort_values(['Card Number', 'Transaction DateTime'])
        
        # Flag for transactions with amount much higher than previous
        df_new['amount_ratio_to_previous'] = df_new.groupby('Card Number')['Amount'].transform(
            lambda x: x / x.shift(1)
        )
        
        # Flag transactions with amount much higher than previous (5x+)
        df_new['high_amount_increase'] = (df_new['amount_ratio_to_previous'] > 5).astype(int)
    
    # Pattern 2: Transactions in multiple countries in short period (we don't have transaction location in this example)
    
    # Pattern 3: Transactions at unusual hours for the customer
    if 'Card Number' in df_new.columns and 'hour_of_day' in df_new.columns:
        # Calculate average transaction hour per customer
        customer_avg_hour = df_new.groupby('Card Number')['hour_of_day'].transform('mean')
        
        # Calculate deviation from average hour (considering cyclic nature of hours)
        hour_diff = np.minimum(
            np.abs(df_new['hour_of_day'] - customer_avg_hour),
            24 - np.abs(df_new['hour_of_day'] - customer_avg_hour)
        )
        
        # Flag transactions at unusual hours (more than 6 hours difference from average)
        df_new['unusual_hour_for_customer'] = (hour_diff > 6).astype(int)
    
    return df_new

def create_all_features(df):
    """
    Applies all feature creation functions to the DataFrame.
    
    Args:
        df (pandas.DataFrame): Original DataFrame.
        
    Returns:
        pandas.DataFrame: DataFrame with all features created.
    """
    df_features = df.copy()
    
    # Apply each feature creation function
    df_features = create_temporal_features(df_features)
    df_features = create_amount_features(df_features)
    df_features = create_location_features(df_features)
    df_features = create_customer_behavior_features(df_features)
    df_features = create_merchant_features(df_features)
    df_features = create_payment_method_features(df_features)
    df_features = create_velocity_features(df_features)
    df_features = create_fraud_pattern_features(df_features)
    
    return df_features

if __name__ == "__main__":
    # Example usage
    config = load_config()
    from data.preprocessing import load_data
    
    # Load data
    df = load_data()
    
    # Create all features
    df_with_features = create_all_features(df)
    
    # Show information about new features
    new_columns = set(df_with_features.columns) - set(df.columns)
    print(f"New features created ({len(new_columns)}):")
    for col in sorted(new_columns):
        print(f"  - {col}") 
