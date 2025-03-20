"""
Module for data preprocessing for credit card fraud detection.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler

# Add root directory to path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration functions from utils module
from utils.config_utils import load_config, get_data_path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_synthetic_data(n_samples=10000, fraud_ratio=0.05, random_state=42):
    """
    Generates synthetic data for credit card fraud detection.
    
    Args:
        n_samples (int): Number of transactions to generate.
        fraud_ratio (float): Ratio of fraudulent transactions (between 0 and 1).
        random_state (int): Random seed for reproducibility.
        
    Returns:
        pandas.DataFrame: DataFrame with synthetic transaction data.
    """
    np.random.seed(random_state)
    
    # Calculate number of fraud and legitimate transactions
    n_fraud = int(n_samples * fraud_ratio)
    n_legitimate = n_samples - n_fraud
    
    # Generate transaction IDs
    transaction_ids = [f"TX{i:010d}" for i in range(1, n_samples + 1)]
    
    # Generate customer data
    card_numbers = [f"CARD{i:08d}" for i in np.random.choice(5000, size=n_samples, replace=True)]
    
    # Generate transaction amounts (different distributions for fraud and legitimate)
    legitimate_amounts = np.random.lognormal(mean=4.0, sigma=1.0, size=n_legitimate)
    # Fraud has more small transactions and occasional large ones
    fraud_amounts = np.concatenate([
        np.random.lognormal(mean=3.0, sigma=1.5, size=int(n_fraud * 0.7)),
        np.random.lognormal(mean=6.0, sigma=2.0, size=n_fraud - int(n_fraud * 0.7))
    ])
    np.random.shuffle(fraud_amounts)
    
    amounts = np.concatenate([legitimate_amounts, fraud_amounts])
    amounts = np.round(amounts, 2)
    
    # Generate transaction dates and times
    base_date = datetime.now() - timedelta(days=30)
    dates = [base_date + timedelta(
        days=np.random.randint(0, 30),
        hours=np.random.randint(0, 24),
        minutes=np.random.randint(0, 60),
        seconds=np.random.randint(0, 60)
    ) for _ in range(n_samples)]
    
    transaction_dates = [d.strftime('%Y-%m-%d') for d in dates]
    transaction_times = [d.strftime('%H:%M:%S') for d in dates]
    transaction_datetimes = dates
    
    # Generate merchant data
    merchant_ids = [f"MERCH{i:06d}" for i in np.random.choice(1000, size=n_samples, replace=True)]
    merchant_names = [f"Merchant {i}" for i in range(1, 501)]
    merchant_categories = ['Retail', 'Food', 'Entertainment', 'Travel', 'Services', 'Other']
    
    merchant_names_selected = np.random.choice(merchant_names, size=n_samples, replace=True)
    merchant_categories_selected = np.random.choice(merchant_categories, size=n_samples, replace=True)
    
    # Generate location data
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 
              'San Antonio', 'San Diego', 'Dallas', 'San Jose', 'Austin', 'Jacksonville',
              'Fort Worth', 'Columbus', 'Charlotte', 'San Francisco', 'Indianapolis', 'Seattle']
    
    states = ['NY', 'CA', 'IL', 'TX', 'AZ', 'PA', 'TX', 'CA', 'TX', 'CA', 'TX', 'FL',
              'TX', 'OH', 'NC', 'CA', 'IN', 'WA']
    
    city_state = list(zip(cities, states))
    
    # For legitimate transactions - mostly consistent locations
    legitimate_locations = np.random.choice(len(city_state), size=n_legitimate, replace=True)
    legitimate_cities = [city_state[i][0] for i in legitimate_locations]
    legitimate_states = [city_state[i][1] for i in legitimate_locations]
    
    # For fraudulent transactions - more varied locations
    fraud_locations = np.random.choice(len(city_state), size=n_fraud, replace=True)
    fraud_cities = [city_state[i][0] for i in fraud_locations]
    fraud_states = [city_state[i][1] for i in fraud_locations]
    
    merchant_cities = legitimate_cities + fraud_cities
    merchant_states = legitimate_states + fraud_states
    
    # Generate payment method data
    payment_methods = ['Credit', 'Debit', 'ACH', 'Wire Transfer']
    capture_methods = ['Chip', 'Swipe', 'Manual Entry', 'Online', 'Contactless']
    card_brands = ['Visa', 'Mastercard', 'Amex', 'Discover']
    
    # Legitimate transactions - more chip and contactless
    legitimate_capture = np.random.choice(
        capture_methods, 
        size=n_legitimate, 
        replace=True, 
        p=[0.4, 0.2, 0.1, 0.2, 0.1]
    )
    
    # Fraudulent transactions - more manual entry and online
    fraud_capture = np.random.choice(
        capture_methods, 
        size=n_fraud, 
        replace=True, 
        p=[0.1, 0.1, 0.4, 0.3, 0.1]
    )
    
    capture_methods_selected = np.concatenate([legitimate_capture, fraud_capture])
    payment_methods_selected = np.random.choice(payment_methods, size=n_samples, replace=True)
    card_brands_selected = np.random.choice(card_brands, size=n_samples, replace=True)
    
    # Generate issuer data
    issuer_banks = ['Bank of America', 'Chase', 'Wells Fargo', 'Citibank', 'Capital One', 
                   'US Bank', 'PNC Bank', 'TD Bank', 'Truist Bank', 'HSBC']
    issuer_countries = ['USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'USA', 'UK']
    
    issuer_data = list(zip(issuer_banks, issuer_countries))
    issuer_indices = np.random.choice(len(issuer_data), size=n_samples, replace=True)
    
    issuer_banks_selected = [issuer_data[i][0] for i in issuer_indices]
    issuer_countries_selected = [issuer_data[i][1] for i in issuer_indices]
    
    # Generate fraud indicator (target variable)
    fraud_indicator = np.zeros(n_samples, dtype=int)
    fraud_indicator[n_legitimate:] = 1
    
    # Shuffle all data to mix fraud and legitimate transactions
    all_data = list(zip(
        transaction_ids, card_numbers, amounts, transaction_dates, transaction_times,
        transaction_datetimes, merchant_ids, merchant_names_selected, merchant_categories_selected,
        merchant_cities, merchant_states, payment_methods_selected, capture_methods_selected,
        card_brands_selected, issuer_banks_selected, issuer_countries_selected, fraud_indicator
    ))
    
    np.random.shuffle(all_data)
    
    # Unpack shuffled data
    (transaction_ids, card_numbers, amounts, transaction_dates, transaction_times,
     transaction_datetimes, merchant_ids, merchant_names_selected, merchant_categories_selected,
     merchant_cities, merchant_states, payment_methods_selected, capture_methods_selected,
     card_brands_selected, issuer_banks_selected, issuer_countries_selected, fraud_indicator) = zip(*all_data)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Transaction ID': transaction_ids,
        'Card Number': card_numbers,
        'Amount': amounts,
        'Transaction Date': transaction_dates,
        'Transaction Time': transaction_times,
        'Transaction DateTime': transaction_datetimes,
        'Merchant ID': merchant_ids,
        'Merchant Name': merchant_names_selected,
        'Merchant Category': merchant_categories_selected,
        'Merchant City': merchant_cities,
        'Merchant State': merchant_states,
        'Payment Method': payment_methods_selected,
        'Capture Method': capture_methods_selected,
        'Card Brand': card_brands_selected,
        'Issuer Bank': issuer_banks_selected,
        'Issuer Country': issuer_countries_selected,
        'Fraud Indicator': fraud_indicator
    })
    
    logger.info(f"Generated synthetic data: {df.shape[0]} rows and {df.shape[1]} columns.")
    logger.info(f"Fraud ratio: {df['Fraud Indicator'].mean():.2%}")
    
    return df

def load_data(config=None, use_synthetic=False):
    """
    Loads raw data from CSV file or generates synthetic data.
    
    Args:
        config (dict, optional): Project configurations. If None, loads from config.yaml.
        use_synthetic (bool): Whether to use synthetic data instead of loading from file.
        
    Returns:
        pandas.DataFrame: DataFrame with loaded data.
    """
    if config is None:
        config = load_config()
    
    if use_synthetic:
        logger.info("Using synthetic data instead of loading from file.")
        return generate_synthetic_data()
    
    # Get data file path
    data_path = get_data_path(config, 'raw_data')
    
    try:
        # Load data
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded: {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except FileNotFoundError:
        logger.warning(f"Data file not found at {data_path}. Generating synthetic data instead.")
        return generate_synthetic_data()

def clean_data(df, config=None):
    """
    Performs basic data cleaning (column removal, null value handling, etc.).
    
    Args:
        df (pandas.DataFrame): DataFrame with raw data.
        config (dict, optional): Project configurations. If None, loads from config.yaml.
        
    Returns:
        pandas.DataFrame: DataFrame with cleaned data.
    """
    if config is None:
        config = load_config()
    
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Remove unnecessary columns
    columns_to_drop = config['preprocessing']['drop_columns']
    df_clean = df_clean.drop(columns=columns_to_drop, errors='ignore')
    
    # Handle null values
    # For numeric values, fill with median
    num_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # For categorical values, fill with mode (most frequent value)
    cat_cols = df_clean.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    
    # Combine date and time into a single datetime field, if they exist
    if 'Transaction Date' in df_clean.columns and 'Transaction Time' in df_clean.columns:
        df_clean['Transaction DateTime'] = pd.to_datetime(
            df_clean['Transaction Date'] + ' ' + df_clean['Transaction Time'], 
            errors='coerce'
        )
    
    return df_clean

def encode_categorical_features(df, encoding_method='label', config=None):
    """
    Encodes categorical variables in the DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame with categorical variables.
        encoding_method (str): Encoding method ('label', 'onehot', 'target').
        config (dict, optional): Project configurations.
        
    Returns:
        pandas.DataFrame: DataFrame with encoded categorical variables.
        dict: Dictionary with used encoders (for future transformation).
    """
    if config is None:
        config = load_config()
    
    # Make a copy to avoid modifying the original
    df_encoded = df.copy()
    
    # Identify categorical columns
    cat_cols = df_encoded.select_dtypes(include=['object']).columns
    
    encoders = {}
    
    if encoding_method == 'label':
        # Label Encoding
        for col in cat_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le
            
    elif encoding_method == 'onehot':
        # One-Hot Encoding
        for col in cat_cols:
            # Create encoder
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            # Apply one-hot encoding
            encoded = ohe.fit_transform(df_encoded[[col]])
            # Create new column names
            col_names = [f"{col}_{val}" for val in ohe.categories_[0]]
            # Create a dataframe with encoded results
            encoded_df = pd.DataFrame(encoded, columns=col_names, index=df_encoded.index)
            # Add new columns to original dataframe
            df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
            # Remove original column
            df_encoded = df_encoded.drop(columns=[col])
            # Store encoder
            encoders[col] = ohe
    
    # Could also implement Target Encoding, but that depends on the target
    # and other problem-specific considerations
    
    return df_encoded, encoders

def scale_numeric_features(df, config=None):
    """
    Scales numeric variables in the DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame with numeric variables.
        config (dict, optional): Project configurations.
        
    Returns:
        pandas.DataFrame: DataFrame with scaled numeric variables.
        dict: Dictionary with used scalers (for future transformation).
    """
    if config is None:
        config = load_config()
    
    # Make a copy to avoid modifying the original
    df_scaled = df.copy()
    
    # Identify numeric columns (excluding target)
    target_col = config['preprocessing']['target_column']
    num_cols = df_scaled.select_dtypes(include=['int64', 'float64']).columns
    num_cols = [col for col in num_cols if col != target_col]
    
    scalers = {}
    
    # Apply scaling to each numeric column
    for col in num_cols:
        scaler = StandardScaler()
        df_scaled[col] = scaler.fit_transform(df_scaled[[col]])
        scalers[col] = scaler
    
    return df_scaled, scalers

def handle_class_imbalance(X, y, method='smote', random_state=42):
    """
    Handles class imbalance in the dataset.
    
    Args:
        X (pandas.DataFrame): Features.
        y (pandas.Series): Target.
        method (str): Method to handle imbalance ('none', 'smote', 'adasyn', 'random').
        random_state (int): Seed for reproducibility.
        
    Returns:
        pandas.DataFrame, pandas.Series: Balanced X and y.
    """
    if method == 'none':
        return X, y
    
    if method == 'smote':
        sampler = SMOTE(random_state=random_state)
    elif method == 'adasyn':
        sampler = ADASYN(random_state=random_state)
    elif method == 'random':
        sampler = RandomOverSampler(random_state=random_state)
    else:
        raise ValueError(f"Unrecognized balancing method: {method}")
    
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    
    return X_resampled, y_resampled

def prepare_data(df=None, config=None):
    """
    Main function that performs the entire data preprocessing pipeline.
    
    Args:
        df (pandas.DataFrame, optional): DataFrame with raw data. If None, loads from file.
        config (dict, optional): Project configurations. If None, loads from config.yaml.
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, preprocessors
    """
    if config is None:
        config = load_config()
    
    # Load data, if not provided
    if df is None:
        df = load_data(config)
    
    # Clean data
    df_clean = clean_data(df, config)
    
    # Separate features and target
    target_col = config['preprocessing']['target_column']
    y = df_clean[target_col]
    X = df_clean.drop(columns=[target_col])
    
    # Encode categorical variables
    encoding_method = config['features']['categorical_encoding']
    X_encoded, encoders = encode_categorical_features(X, encoding_method, config)
    
    # Scale numeric variables, if configured
    if config['features']['perform_scaling']:
        X_processed, scalers = scale_numeric_features(X_encoded, config)
    else:
        X_processed = X_encoded
        scalers = {}
    
    # Split into train and test
    test_size = config['preprocessing']['test_size']
    random_state = config['preprocessing']['random_state']
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Handle class imbalance (only on training data)
    imbalance_method = config['features']['handle_imbalance']
    X_train, y_train = handle_class_imbalance(X_train, y_train, imbalance_method, random_state)
    
    # Gather all used preprocessors
    preprocessors = {
        'encoders': encoders,
        'scalers': scalers
    }
    
    # Save preprocessed data, if configured
    processed_data_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        config['paths']['processed_data']
    )
    
    # Make sure directory exists
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    
    # Create DataFrame to save
    df_processed = X_processed.copy()
    df_processed[target_col] = y
    
    # Save preprocessed data
    df_processed.to_csv(processed_data_path, index=False)
    print(f"Preprocessed data saved at: {processed_data_path}")
    
    return X_train, X_test, y_train, y_test, preprocessors

if __name__ == "__main__":
    # Run preprocessing pipeline
    X_train, X_test, y_train, y_test, preprocessors = prepare_data()
    
    # Print information about processed data
    print(f"\nTraining data: {X_train.shape[0]} rows, {X_train.shape[1]} columns")
    print(f"Test data: {X_test.shape[0]} rows, {X_test.shape[1]} columns")
    
    # Check class distribution
    print(f"\nClass distribution (training):")
    print(y_train.value_counts())
    
    print(f"\nClass distribution (test):")
    print(y_test.value_counts()) 
