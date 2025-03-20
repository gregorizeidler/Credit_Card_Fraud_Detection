"""
Package for data loading, preprocessing and pipeline functionality for the fraud detection project.
"""

from data.preprocessing import (
    load_data,
    prepare_data,
    generate_synthetic_data,
    clean_data,
    encode_categorical_features,
    scale_numeric_features,
    handle_class_imbalance
)

from data.pipeline import (
    create_preprocessing_pipeline,
    create_model_pipeline
)

__all__ = [
    # Data preprocessing
    'load_data',
    'prepare_data',
    'generate_synthetic_data',
    'clean_data',
    'encode_categorical_features',
    'scale_numeric_features',
    'handle_class_imbalance',
    
    # Data pipeline
    'create_preprocessing_pipeline',
    'create_model_pipeline'
] 
