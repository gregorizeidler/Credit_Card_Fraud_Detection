"""
Package of functionalities for training, evaluation, and inference of fraud detection models.
"""

from models.training import (
    create_model,
    train_model,
    tune_hyperparameters,
    create_ensemble_model,
    save_model,
    train_all_models,
    tune_all_models
)

from models.evaluation import (
    calculate_metrics,
    evaluate_model,
    compare_models,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_calibration_curve,
    plot_shap_summary,
    generate_classification_report
)

from models.model_registry import ModelRegistry

__all__ = [
    # Model training
    'create_model',
    'train_model',
    'tune_hyperparameters',
    'create_ensemble_model',
    'save_model',
    'train_all_models',
    'tune_all_models',
    
    # Model evaluation
    'calculate_metrics',
    'evaluate_model',
    'compare_models',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_precision_recall_curve',
    'plot_calibration_curve',
    'plot_shap_summary',
    'generate_classification_report',
    
    # Model registry
    'ModelRegistry'
] 
