"""
Real-time Inference Page for the fraud detection application.
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
import datetime
import random

# Add root directory to path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.preprocessing import load_config

# Page configuration
st.set_page_config(
    page_title="Real-time Inference - Fraud Detection", 
    page_icon="🔍",
    layout="wide"
)

# Page title
st.title("Real-time Inference")
st.markdown("""
    This page allows you to test the fraud detection model in real-time,
    simulating a credit card transaction and checking the fraud probability.
""")

# Model settings in sidebar
st.sidebar.header("Settings")

model_options = ["Random Forest", "XGBoost", "Logistic Regression", "LightGBM", "Ensemble"]
selected_model = st.sidebar.selectbox("Model", model_options, index=4)  # Default is Ensemble

threshold = st.sidebar.slider(
    "Fraud Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.5,
    step=0.05,
    help="Threshold for classifying a transaction as fraud"
)

# Function to generate simulated customer history data
def generate_customer_history(customer_id):
    """Generates simulated history for a customer."""
    np.random.seed(int(customer_id) % 100)  # Seed based on ID for consistency
    
    n_transactions = np.random.randint(5, 50)  # Number of transactions in history
    
    # Transaction dates in the last 90 days
    dates = [datetime.datetime.now() - datetime.timedelta(days=np.random.randint(1, 90)) 
             for _ in range(n_transactions)]
    dates.sort()
    
    # Transaction amounts
    amounts = np.random.lognormal(4, 1, n_transactions)
    
    # Merchant categories
    merchant_categories = np.random.choice(
        ['Retail', 'Restaurant', 'Supermarket', 'Travel', 'Online', 'Services', 'Other'],
        n_transactions
    )
    
    # Create DataFrame
    history_df = pd.DataFrame({
        'Date': dates,
        'Amount': amounts,
        'Category': merchant_categories,
        'Fraud': np.random.choice([0, 1], n_transactions, p=[0.95, 0.05])  # 5% of transactions are fraud
    })
    
    return history_df

# Form for card information
st.header("Transaction Information")

col1, col2 = st.columns(2)

with col1:
    # Customer and card data
    st.subheader("Customer and Card Data")
    
    customer_id = st.text_input("Customer ID", value="12345678")
    card_number = st.text_input("Card Number (last 4 digits)", value="1234")
    card_brand = st.selectbox("Card Brand", ["Visa", "Mastercard", "American Express", "Elo", "Other"])
    card_type = st.selectbox("Card Type", ["Credit", "Debit", "Prepaid"])

with col2:
    # Transaction data
    st.subheader("Transaction Data")
    
    transaction_amount = st.number_input("Transaction Amount ($)", min_value=0.01, value=150.00, step=10.0)
    transaction_date = st.date_input("Transaction Date", value=datetime.datetime.now())
    transaction_time = st.time_input("Transaction Time", value=datetime.datetime.now().time())
    transaction_datetime = datetime.datetime.combine(transaction_date, transaction_time)

# Merchant information
st.subheader("Merchant Information")

col1, col2, col3 = st.columns(3)

with col1:
    merchant_id = st.text_input("Merchant ID", value="MER12345")
    merchant_name = st.text_input("Merchant Name", value="Example Commerce Ltd")

with col2:
    merchant_category = st.selectbox(
        "Merchant Category", 
        ["Retail", "Restaurant", "Supermarket", "Travel", "Online", "Services", "Other"]
    )
    merchant_country = st.selectbox("Country", ["United States", "Brazil", "Other"])

with col3:
    merchant_city = st.text_input("City", value="New York")
    is_new_merchant = st.checkbox("New Merchant (First Transaction)", value=False)

# Transaction details
st.subheader("Transaction Details")

col1, col2 = st.columns(2)

with col1:
    capture_method = st.selectbox(
        "Capture Method", 
        ["Chip", "Contactless", "Manual", "Internet", "Mobile App", "ATM"]
    )
    payment_channel = st.selectbox(
        "Payment Channel",
        ["In-person", "E-commerce", "Mobile App", "Recurring", "Phone"]
    )

with col2:
    device_id = st.text_input("Device ID (if online)", value="", help="For example, Device Token or IP")
    is_international = merchant_country != "United States"
    st.write(f"International Transaction: {'Yes' if is_international else 'No'}")

# Button to check the transaction
if st.button("Check Transaction", help="Click to analyze the transaction for possible fraud"):
    # Create a progress container
    progress_container = st.container()
    
    with progress_container:
        st.markdown("### Processing Transaction...")
        progress_bar = st.progress(0)
        
        # Simulate processing
        for i in range(1, 101):
            # Update progress bar
            progress_bar.progress(i / 100)
            
            # Add delay for simulation
            time.sleep(0.02)
        
        # Simulated prediction result
        st.markdown("### Analysis Result")
        
        # Generate fraud probability based on some factors
        # Let's simulate based on some risk factors
        risk_factors = {
            'high_amount': 1.0 if transaction_amount > 1000 else 0.0,
            'unusual_time': 1.0 if transaction_time.hour < 5 or transaction_time.hour > 22 else 0.0,
            'international': 1.0 if is_international else 0.0,
            'new_merchant': 1.0 if is_new_merchant else 0.0,
            'online_payment': 1.0 if payment_channel == "E-commerce" else 0.0
        }
        
        # Calculate probability based on factors (simplified simulation)
        base_probability = 0.05  # Base fraud probability
        risk_contribution = sum(risk_factors.values()) * 0.15  # Factors contribution
        
        # Add a random factor to vary results
        np.random.seed(int(time.time()))
        random_factor = np.random.beta(2, 5) * 0.2  # Random factor between 0 and 0.2
        
        # Final probability
        fraud_probability = min(base_probability + risk_contribution + random_factor, 0.99)
        
        # Classification based on threshold
        is_fraud = fraud_probability >= threshold
        
        # Display result with style
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if is_fraud:
                st.error("⚠️ FRAUD ALERT")
            else:
                st.success("✅ TRANSACTION APPROVED")
                
        with col2:
            # Risk meter
            st.markdown(f"**Fraud Probability: {fraud_probability:.2%}**")
            
            # Colored progress bar for risk
            risk_html = f"""
                <div style="
                    background: linear-gradient(90deg, green, yellow, red);
                    width: 100%;
                    height: 20px;
                    border-radius: 10px;
                    position: relative;
                ">
                    <div style="
                        position: absolute;
                        left: {fraud_probability * 100}%;
                        top: -10px;
                        transform: translateX(-50%);
                        font-weight: bold;
                    ">▼</div>
                </div>
                <div style="
                    display: flex;
                    justify-content: space-between;
                    width: 100%;
                    margin-top: 5px;
                ">
                    <span>Low Risk</span>
                    <span>High Risk</span>
                </div>
            """
            st.markdown(risk_html, unsafe_allow_html=True)
        
        # Recommendation
        st.subheader("Recommendation")
        
        if is_fraud:
            st.markdown("""
                🚫 **Block transaction and investigate**
                
                This transaction presents a high risk of fraud. We recommend:
                - Block the transaction
                - Contact the customer for verification
                - Request confirmation through a second authentication factor
                - Forward to the fraud analysis team
            """)
        else:
            st.markdown("""
                ✅ **Approve transaction**
                
                This transaction presents a low risk of fraud. We recommend:
                - Approve the transaction
                - Register for future monitoring
            """)
        
        # Risk factors
        st.subheader("Risk Factor Analysis")
        
        # Display risk factors table
        risk_data = []
        
        risk_descriptions = {
            'high_amount': "Transaction amount is unusually high",
            'unusual_time': "Transaction occurred during unusual hours",
            'international': "Transaction is international",
            'new_merchant': "First transaction with this merchant",
            'online_payment': "Online payment (higher risk channel)"
        }
        
        risk_levels = {0.0: "Low", 1.0: "High"}
        
        for factor, value in risk_factors.items():
            risk_data.append({
                "Risk Factor": risk_descriptions[factor],
                "Risk Level": risk_levels[value],
                "Risk Score": value
            })
        
        risk_df = pd.DataFrame(risk_data)
        
        # Style the dataframe
        def highlight_risk(val):
            if val == "High":
                return 'background-color: #ffcccc'
            elif val == "Low":
                return 'background-color: #ccffcc'
            else:
                return ''
        
        styled_risk_df = risk_df.style.applymap(highlight_risk, subset=['Risk Level'])
        
        st.dataframe(styled_risk_df)
        
        # Display customer history
        st.subheader("Customer Transaction History")
        
        # Get customer history
        history_df = generate_customer_history(customer_id)
        
        # Add current transaction at the top
        current_transaction = pd.DataFrame({
            'Date': [transaction_datetime],
            'Amount': [transaction_amount],
            'Category': [merchant_category],
            'Fraud': [1 if is_fraud else 0]
        })
        
        # Combine and display
        all_transactions = pd.concat([current_transaction, history_df]).reset_index(drop=True)
        all_transactions['Date'] = all_transactions['Date'].dt.strftime('%Y-%m-%d %H:%M')
        
        # Style the dataframe
        def highlight_current_and_fraud(df):
            styles = []
            for i in range(len(df)):
                if i == 0:  # Current transaction
                    styles.append('background-color: #e6f3ff')
                elif df.loc[i, 'Fraud'] == 1:  # Fraud
                    styles.append('background-color: #ffcccc')
                else:
                    styles.append('')
            return styles
        
        styled_history = all_transactions.style.apply(lambda _: highlight_current_and_fraud(all_transactions), axis=0)
        
        # Show the history
        st.dataframe(styled_history)
        
        # Visualization of transaction amount over time
        st.subheader("Transaction Amount Pattern")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot historical transactions
        scatter = ax.scatter(
            range(1, len(all_transactions)),
            all_transactions.iloc[1:]['Amount'],
            c=all_transactions.iloc[1:]['Fraud'],
            cmap='coolwarm',
            alpha=0.7,
            s=50
        )
        
        # Add current transaction with a different marker
        ax.scatter(
            0, 
            all_transactions.iloc[0]['Amount'],
            marker='*',
            color='gold' if is_fraud else 'green',
            s=200,
            label='Current Transaction'
        )
        
        # Add legend
        legend1 = ax.legend(*scatter.legend_elements(), title="Fraud")
        ax.add_artist(legend1)
        ax.legend(loc='upper right')
        
        ax.set_xlabel('Transaction Index (0 = current)')
        ax.set_ylabel('Amount ($)')
        ax.set_title('Transaction Amount Pattern')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        st.pyplot(fig)
        
        # Comparison with normal behavior
        st.subheader("Comparison with Normal Behavior")
        
        # Calculate some statistics
        avg_amount = history_df['Amount'].mean()
        avg_frequency = 90 / len(history_df)  # Avg days between transactions
        
        # Compare current transaction with average
        amount_ratio = transaction_amount / avg_amount if avg_amount > 0 else float('inf')
        
        # Display comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Average Transaction Amount", f"${avg_amount:.2f}", 
                     f"{(amount_ratio - 1)*100:.1f}%" if amount_ratio > 1 else f"{(1 - amount_ratio)*100:.1f}%")
        
        with col2:
            st.metric("Average Days Between Transactions", f"{avg_frequency:.1f} days")
        
        # Final decision prompt
        st.subheader("Final Decision")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.button("Approve Transaction", type="primary", disabled=is_fraud)
        
        with col2:
            st.button("Decline Transaction", type="secondary", disabled=not is_fraud)

# Batch Analysis
st.header("Batch Analysis")
st.markdown("""
    Upload a CSV file with multiple transactions for batch analysis.
    The file must contain columns for: amount, date/time, merchant, and other relevant information.
""")

uploaded_file = st.file_uploader("Upload CSV File", type="csv")

if uploaded_file is not None:
    # Load data from file
    try:
        batch_data = pd.read_csv(uploaded_file)
        st.success(f"File loaded successfully: {batch_data.shape[0]} transactions found.")
        
        # Show first few rows
        st.subheader("Data Visualization")
        st.dataframe(batch_data.head())
        
        # Button to process batch
        if st.button("Process Batch"):
            # Create result container
            with st.spinner("Processing transactions in batch..."):
                # Simulated batch processing
                time.sleep(2)  # Simulated processing time
                
                # Add simulated fraud probability column
                batch_data['fraud_probability'] = np.random.beta(2, 15, size=batch_data.shape[0])
                batch_data['is_fraud'] = batch_data['fraud_probability'] >= threshold
                
                # Show results
                st.subheader("Batch Analysis Results")
                
                # Count fraudulent transactions
                fraud_count = batch_data['is_fraud'].sum()
                total_count = batch_data.shape[0]
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Transactions", total_count)
                col2.metric("Suspicious Transactions", fraud_count)
                col3.metric("Fraud Rate", f"{fraud_count/total_count:.2%}")
                
                # Display detailed results
                st.subheader("Details of Suspicious Transactions")
                fraud_transactions = batch_data[batch_data['is_fraud']].sort_values('fraud_probability', ascending=False)
                
                if not fraud_transactions.empty:
                    st.dataframe(fraud_transactions)
                    
                    # Download results
                    csv = fraud_transactions.to_csv(index=False)
                    st.download_button(
                        label="Download Fraud Results",
                        data=csv,
                        file_name="fraud_transactions.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No suspicious transactions found.")
    
    except Exception as e:
        st.error(f"Error processing file: {e}")

# Additional Information
with st.expander("How does real-time detection work?"):
    st.markdown("""
        ### Fraud Detection Process in Real-time
        
        1. **Data Collection**: When a transaction is initiated, we collect various data such as value, 
        time, location, customer history, etc.
        
        2. **Feature Engineering**: We transform these data into relevant features like:
           - Value deviation from customer history pattern
           - Time since last transaction
           - Geographical distance between transactions
           - Merchant behavior patterns
        
        3. **Model Inference**: We send these features to the trained model which returns a 
        fraud probability.
        
        4. **Rule Application**: We combine the model result with business rules to 
        make a decision.
        
        5. **Action**: Based on the risk level, the transaction can be:
           - Approved automatically
           - Sent for additional verification
           - Blocked
        
        This process occurs in milliseconds, allowing near-instantaneous decisions to prevent fraud while minimizing impact on the customer experience.
    """) 
