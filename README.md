# 💳 Credit Card Fraud Detection

## 📋 Description
This project implements a fraud detection system for credit card transactions using advanced data analysis and machine learning techniques. The system includes an interactive web interface built with Streamlit to facilitate data analysis, visualization, and model-driven insights.

## ✨ Features
- 📊 Exploratory analysis of transaction data
- 🛠️ Feature engineering for fraud detection
- 🤖 Training and comparison of multiple ML models
- 📈 Interactive visualizations and dashboards
- 🔍 Model explainability with SHAP and LIME techniques
- ⚖️ Risk threshold adjustments for fraud detection
- 🔄 Real-time transaction monitoring and inference

## 📁 Project Structure
```
├── app/                    # Streamlit application
│   ├── pages/              # Application pages
│   ├── components/         # Reusable UI components
│   └── app.py              # Application entry point
├── data/                   # Data and processing
│   ├── raw/                # Raw data
│   ├── processed/          # Processed data
│   ├── preprocessing.py    # Preprocessing scripts
│   └── pipeline.py         # Processing pipelines with scikit-learn
├── models/                 # ML models
│   ├── training.py         # Model training
│   ├── evaluation.py       # Performance evaluation
│   └── model_registry.py   # Model registry and versioning system
├── features/               # Feature engineering
│   ├── creation.py         # Basic feature creation
│   ├── selection.py        # Feature selection
│   └── advanced_features.py # Advanced features for fraud detection
├── utils/                  # Utility functions and classes
│   └── config_utils.py     # Configuration utilities
├── requirements.txt        # Python dependencies
├── config.yaml             # Configuration file
└── fictitious_credit_card_transactions.csv  # Sample data file
```

## 🚀 Installation
```bash
# Clone the repository
git clone https://github.com/[seu-username]/fraud-credit.git
cd fraud-credit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🔧 Usage
```bash
# Run the Streamlit application
cd app
streamlit run app.py
```

## 🎬 Demo Videos

### 🎥 Initial Project Demo
This video demonstrates the initial version of the fraud detection system:
[Watch Initial Demo](https://drive.google.com/file/d/16K7kBDLYZsRexCD35zrR7IBiBnUaYxC1/view?usp=sharing)

### 🎥 Final Project Demo
This video shows the completed project with all features implemented:
[Watch Final Demo](https://drive.google.com/file/d/1729n_4bUDQtDTbXMlMNT2wHQIx2kc0w_/view?usp=sharing)

## 🔧 Technical Features

### 🔄 Processing Pipelines
The system uses `sklearn.pipeline.Pipeline` to create reproducible processing flows:
- ✅ Consistent data preprocessing
- 🔄 Transformation of categorical and numerical features
- ⏱️ Automatic extraction of temporal characteristics
- ⚖️ Use of SMOTE for class balancing

### 📊 Model Registry System
A complete model versioning system has been implemented:
- 💾 Storage of multiple model versions
- 📝 Registration of hyperparameters and performance metrics
- 🔍 Traceability of used features
- 🔌 API to select the best model by metric

### 🚩 Advanced Features for Fraud Detection
We implemented advanced features specific to fraud:
- ⚡ Transaction velocity analysis
- 👤 Behavioral pattern detection
- 🕒 Temporal anomaly identification
- 🎯 Composite fraud risk score

### 💻 Responsive Interface
The Streamlit application has a modern and responsive interface:
- 🌃 Interactive dashboard with dark theme
- 📊 Real-time monitoring capabilities
- 🎚️ Risk threshold adjustments with immediate visual feedback
- 🔢 Transaction fraud scoring with detailed explanations

## 📊 Dataset
The project includes a sample fictitious credit card transaction dataset containing:
- 💰 Transaction information (amount, date, status)
- 🏪 Merchant data (ID, category, location)
- 💳 Card information (brand, issuing bank)
- 🚨 Fraud indicator (target)

Users can upload their own data or use the built-in synthetic data generator.

## 🤖 Implemented Models
- 📊 Logistic Regression
- 🌲 Random Forest
- 🚀 XGBoost
- 💡 LightGBM
- 🤝 Model Ensemble (Voting)

## 🤝 How to Contribute
Contributions are welcome! Follow these steps:

1. 🍴 Fork the project
2. 🌿 Create a branch for your feature (`git checkout -b feature/new-feature`)
3. 💾 Commit the changes (`git commit -m 'Add new feature'`)
4. 📤 Push to the remote repository (`git push origin feature/new-feature`)
5. 🔃 Open a Pull Request

### 🎯 Areas for contribution
- 🧠 Implementation of new detection algorithms
- 📊 Improvement of existing visualizations
- ⚡ Performance optimization
- 📚 Documentation and tutorials

## 📄 License
This project is licensed under the MIT license - see the LICENSE file for details.
