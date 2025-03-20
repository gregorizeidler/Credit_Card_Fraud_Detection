"""
Utilities package for the credit card fraud detection project.

This package contains utility functions for configuration management, 
file handling, and other general-purpose helpers.
"""

from utils.config_utils import (
    load_config,
    get_project_root,
    get_data_path,
    ensure_dir_exists
)

# Constants
VERSION = '1.0.0'
AUTHOR = 'Credit Fraud Detection Team'

__all__ = [
    # Configuration utilities
    'load_config',
    'get_project_root',
    'get_data_path',
    'ensure_dir_exists',
    
    # Constants
    'VERSION',
    'AUTHOR'
] 
