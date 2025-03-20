# Credit Card Fraud Detection

## Description
This project implements a fraud detection system for credit card transactions using advanced data analysis and machine learning techniques. The system includes an interactive web interface built with Streamlit to facilitate data analysis, visualization, and modeling.

## Features
- Exploratory analysis of transaction data
- Feature engineering for fraud detection
- Training and comparison of multiple ML models
- Interactive visualizations and dashboards
- Model explainability with SHAP and other techniques
- "What-if" simulations to understand behaviors
- Generation of exportable reports

## Project Structure
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
│   ├── model_registry.py   # Model registry and versioning system
│   └── saved/              # Saved trained models
├── features/               # Feature engineering
│   ├── creation.py         # Basic feature creation
│   ├── selection.py        # Feature selection
│   └── advanced_features.py # Advanced features for fraud detection
├── visualization/          # Visualizations
│   ├── exploratory.py      # Exploratory visualizations
│   └── model_viz.py        # Model visualizations
├── utils/                  # Utility functions and classes
│   ├── metrics.py          # Custom metrics
│   ├── config_utils.py     # Configuration utilities
│   └── helpers.py          # Helper functions
├── tests/                  # Unit and integration tests
├── notebooks/              # Jupyter notebooks for analysis
├── requirements.txt        # Python dependencies
├── setup.py                # Package configuration
└── config.yaml             # Configuration file
```

## Installation
```bash
# Clone the repository
git clone https://github.com/your-username/fraud-credit.git
cd fraud-credit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
# Run the Streamlit application
streamlit run app/app.py

# To use a specific theme (light/dark)
# Add ?theme=dark to the browser URL (e.g., http://localhost:8501/?theme=dark)
```

## Technical Features

### Processing Pipelines
The system uses `sklearn.pipeline.Pipeline` to create reproducible processing flows:
- Consistent data preprocessing
- Transformation of categorical and numerical features
- Automatic extraction of temporal characteristics
- Use of SMOTE for class balancing

### Model Registry System
A complete model versioning system has been implemented:
- Storage of multiple model versions
- Registration of hyperparameters and performance metrics
- Traceability of used features
- API to select the best model by metric

### Advanced Features for Fraud Detection
We implemented advanced features specific to fraud:
- Transaction velocity analysis
- Behavioral pattern detection
- Temporal anomaly identification
- Composite fraud risk score

### Responsive Interface
The Streamlit application has a modern and responsive interface:
- Adaptable design for different devices
- Configurable light/dark theme
- Animations and visual feedback
- Interactive and intuitive components

## Dataset
The project uses a credit card transaction dataset containing:
- Transaction information (amount, date, status)
- Merchant data (ID, category, location)
- Card information (brand, issuing bank)
- Fraud indicator (target)

## Implemented Models
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- Model Ensemble (Voting)

## How to Contribute
Contributions are welcome! Follow these steps:

1. Fork the project
2. Create a branch for your feature (`git checkout -b feature/new-feature`)
3. Commit the changes (`git commit -m 'Add new feature'`)
4. Push to the remote repository (`git push origin feature/new-feature`)
5. Open a Pull Request

### Areas for contribution
- Implementation of new detection algorithms
- Improvement of existing visualizations
- Performance optimization
- Documentation and tutorials
- New automated test sets

## License
This project is licensed under the MIT license - see the LICENSE file for details.
