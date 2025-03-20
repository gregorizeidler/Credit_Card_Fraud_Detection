"""
Utility module for handling project configurations and paths.
"""

import os
import yaml
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_project_root():
    """
    Gets the project root path.
    
    Returns:
        str: Absolute path to the project root.
    """
    # Assuming this file is in utils/config_utils.py
    current_file = os.path.abspath(__file__)
    utils_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(utils_dir)
    return project_root

def load_config(config_path=None):
    """
    Loads settings from the config.yaml file.
    
    Args:
        config_path (str, optional): Path to the configuration file.
            If None, uses the default path relative to the project root.
    
    Returns:
        dict: Settings loaded from the YAML file.
    """
    if config_path is None:
        config_path = os.path.join(get_project_root(), 'config.yaml')
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Settings successfully loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading settings: {e}")
        raise

def get_data_path(config=None, data_key='raw_data'):
    """
    Gets the full path to a data file based on the configuration.
    
    Args:
        config (dict, optional): Project settings. If None, loads from file.
        data_key (str): Key in the configuration dictionary that specifies the relative path.
    
    Returns:
        str: Absolute path to the data file.
    """
    if config is None:
        config = load_config()
    
    relative_path = config['paths'].get(data_key)
    if not relative_path:
        raise ValueError(f"Path '{data_key}' not found in configuration.")
    
    full_path = os.path.join(get_project_root(), relative_path)
    logger.debug(f"Resolved data path: {full_path}")
    return full_path

def ensure_dir_exists(directory):
    """
    Ensures that a directory exists, creating it if necessary.
    
    Args:
        directory (str): Path of the directory to be checked/created.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Directory created: {directory}") 
