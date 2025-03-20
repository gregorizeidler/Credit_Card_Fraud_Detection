"""
Model registry module for version control and traceability.
"""

import os
import sys
import json
import pickle
import logging
import datetime
import pandas as pd
import numpy as np
from pathlib import Path

# Add the root directory to the path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from utils.config_utils import load_config, get_project_root, ensure_dir_exists

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    Class to manage model registry and versioning.
    """
    
    def __init__(self, config=None):
        """
        Initialize the model registry.
        
        Args:
            config (dict, optional): Project configuration. If None, loads from file.
        """
        self.config = config if config is not None else load_config()
        
        # Directory to store models
        self.models_dir = os.path.join(get_project_root(), self.config['paths']['models_dir'])
        ensure_dir_exists(self.models_dir)
        
        # Model registry (metadata)
        self.registry_file = os.path.join(self.models_dir, 'model_registry.json')
        self.registry = self._load_registry()
        
        logger.info(f"Model registry initialized at {self.models_dir}")
    
    def _load_registry(self):
        """Loads the model registry from the JSON file."""
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading registry: {e}")
                return {}
        return {}
    
    def _save_registry(self):
        """Saves the model registry to the JSON file."""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.registry, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving registry: {e}")
    
    def register_model(self, model, model_name, model_type, metrics, features=None, params=None, description=None):
        """
        Registers a new model in the registry.
        
        Args:
            model: The trained model.
            model_name (str): Name of the model.
            model_type (str): Type of model (e.g., 'logistic_regression', 'random_forest', etc.).
            metrics (dict): Model performance metrics.
            features (list, optional): List of features used in training.
            params (dict, optional): Model parameters.
            description (str, optional): Model description.
            
        Returns:
            str: Unique ID of the registered model.
        """
        # Generate unique ID for the model
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"{model_type}_{timestamp}"
        
        # Extract model parameters, if not provided
        if params is None and hasattr(model, 'get_params'):
            params = model.get_params()
        
        # Create registry entry
        model_info = {
            'id': model_id,
            'name': model_name,
            'type': model_type,
            'timestamp': timestamp,
            'metrics': metrics,
            'params': params,
            'features': features,
            'description': description,
            'file_path': os.path.join(self.models_dir, f"{model_id}.pkl")
        }
        
        # Save model
        try:
            with open(model_info['file_path'], 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Model {model_id} saved at {model_info['file_path']}")
        except Exception as e:
            logger.error(f"Error saving model {model_id}: {e}")
            raise
        
        # Update registry
        self.registry[model_id] = model_info
        self._save_registry()
        
        logger.info(f"Model {model_id} registered successfully")
        return model_id
    
    def load_model(self, model_id):
        """
        Loads a model from the registry by ID.
        
        Args:
            model_id (str): ID of the model to load.
            
        Returns:
            tuple: (loaded model, model information)
        """
        if model_id not in self.registry:
            raise ValueError(f"Model with ID {model_id} not found in registry")
        
        model_info = self.registry[model_id]
        model_path = model_info['file_path']
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model {model_id} loaded successfully")
            return model, model_info
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            raise
    
    def get_best_model(self, metric='roc_auc', model_type=None):
        """
        Gets the best model from the registry based on a metric.
        
        Args:
            metric (str): Metric to use for comparison (default: 'roc_auc').
            model_type (str, optional): Filter by model type.
            
        Returns:
            tuple: (loaded model, model information)
        """
        if not self.registry:
            raise ValueError("Empty model registry")
        
        # Filter by model type, if specified
        models = self.registry.values()
        if model_type:
            models = [m for m in models if m['type'] == model_type]
        
        if not models:
            raise ValueError(f"No models of type {model_type} found")
        
        # Find the best model based on the metric
        best_model_info = max(models, key=lambda m: m['metrics'].get(metric, 0))
        
        return self.load_model(best_model_info['id'])
    
    def list_models(self, model_type=None):
        """
        Lists all registered models.
        
        Args:
            model_type (str, optional): Filter by model type.
            
        Returns:
            pandas.DataFrame: DataFrame with model information.
        """
        if not self.registry:
            return pd.DataFrame()
        
        # Extract relevant information
        model_list = []
        for model_id, model_info in self.registry.items():
            if model_type and model_info['type'] != model_type:
                continue
            
            model_data = {
                'id': model_id,
                'name': model_info['name'],
                'type': model_info['type'],
                'timestamp': model_info['timestamp'],
                'description': model_info.get('description', '')
            }
            
            # Add metrics
            for metric, value in model_info.get('metrics', {}).items():
                model_data[f"metric_{metric}"] = value
            
            model_list.append(model_data)
        
        return pd.DataFrame(model_list)
    
    def delete_model(self, model_id):
        """
        Removes a model from the registry.
        
        Args:
            model_id (str): ID of the model to remove.
            
        Returns:
            bool: True if the model was successfully removed, False otherwise.
        """
        if model_id not in self.registry:
            logger.warning(f"Model with ID {model_id} not found in registry")
            return False
        
        model_info = self.registry[model_id]
        
        # Delete model file
        try:
            if os.path.exists(model_info['file_path']):
                os.remove(model_info['file_path'])
            
            # Remove from registry
            del self.registry[model_id]
            self._save_registry()
            
            logger.info(f"Model {model_id} removed successfully")
            return True
        except Exception as e:
            logger.error(f"Error removing model {model_id}: {e}")
            return False 
