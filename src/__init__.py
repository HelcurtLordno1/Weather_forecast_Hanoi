# Hanoi Temperature Forecasting Project
# Source package initialization

__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "Machine Learning application for Hanoi temperature forecasting"

# Import main modules (only basic modules to avoid circular imports)
from . import data_utils
from . import feature_utils

__all__ = [
    'data_utils',
    'feature_utils'
]