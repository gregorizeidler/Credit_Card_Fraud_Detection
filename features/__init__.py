"""
Package of functionalities for creation and selection of features for fraud detection.
"""

from features.creation import (
    create_amount_features,
    create_temporal_features,
    create_location_features,
    create_customer_behavior_features,
    create_merchant_features,
    create_payment_method_features,
    create_velocity_features,
    create_fraud_pattern_features,
    create_all_features
)

from features.selection import (
    remove_highly_correlated_features,
    select_features_univariate,
    select_features_model_based,
    select_features_rfe,
    apply_pca,
    select_best_features
)

from features.advanced_features import (
    create_transaction_velocity_features,
    create_behavioral_pattern_features,
    create_temporal_pattern_features,
    create_fraud_detection_score,
    create_all_advanced_features
)

__all__ = [
    # Basic feature creation
    'create_amount_features',
    'create_temporal_features',
    'create_location_features',
    'create_customer_behavior_features',
    'create_merchant_features',
    'create_payment_method_features',
    'create_velocity_features',
    'create_fraud_pattern_features',
    'create_all_features',
    
    # Feature selection
    'remove_highly_correlated_features',
    'select_features_univariate',
    'select_features_model_based',
    'select_features_rfe',
    'apply_pca',
    'select_best_features',
    
    # Advanced features
    'create_transaction_velocity_features',
    'create_behavioral_pattern_features',
    'create_temporal_pattern_features',
    'create_fraud_detection_score',
    'create_all_advanced_features'
] 
