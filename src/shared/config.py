"""
Shared Configuration for Weather Forecasting Project

Common constants, paths, and configuration settings used across daily and hourly forecasting.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Daily data paths
DAILY_DATA_FILE = RAW_DATA_DIR / "Hanoi-Daily-10-years.csv"
DAILY_PROCESSED_DIR = PROCESSED_DATA_DIR / "daily"

# Hourly data paths
HOURLY_DATA_FILE = RAW_DATA_DIR / "hanoi_weather_data_hourly.csv"
HOURLY_PROCESSED_DIR = PROCESSED_DATA_DIR / "hourly"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
DAILY_MODELS_DIR = MODELS_DIR / "daily_trained"
HOURLY_MODELS_DIR = MODELS_DIR / "hourly_trained"

# Output paths
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Notebook paths
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOKS_HOURLY_DIR = PROJECT_ROOT / "notebooks_hourly"

# App paths
APP_DIR = PROJECT_ROOT / "app"

# Common model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15

# Time series validation parameters
N_SPLITS = 5
MAX_TRAIN_SIZE = None

# Feature engineering parameters
FORECAST_DAYS_DAILY = 5  # 5 days ahead for daily forecasting
FORECAST_HOURS_HOURLY = 120  # 120 hours (5 days) ahead for hourly forecasting

# Weather feature categories
TEMPERATURE_FEATURES = ['temp', 'feelslike', 'dew']
HUMIDITY_FEATURES = ['humidity', 'precip', 'precipprob']
WIND_FEATURES = ['windspeed', 'winddir', 'windgust']
PRESSURE_FEATURES = ['sealevelpressure']
VISIBILITY_FEATURES = ['visibility', 'cloudcover']
SOLAR_FEATURES = ['solarradiation', 'solarenergy', 'uvindex']
CATEGORICAL_FEATURES = ['conditions', 'icon', 'preciptype']

# All numeric features (excluding categorical)
NUMERIC_FEATURES = (TEMPERATURE_FEATURES + HUMIDITY_FEATURES + WIND_FEATURES + 
                   PRESSURE_FEATURES + VISIBILITY_FEATURES + SOLAR_FEATURES)

# Model hyperparameter grids
HYPERPARAMETER_GRIDS = {
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'gradient_boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0]
    },
    'xgboost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0]
    }
}

# Evaluation metrics
PRIMARY_METRIC = 'rmse'  # Primary metric for model selection
METRICS_TO_TRACK = ['rmse', 'mae', 'r2', 'mape', 'directional_accuracy']

# Plotting configuration
PLOT_STYLE = 'seaborn-v0_8'
FIGURE_SIZE = (12, 8)
DPI = 300

# Color schemes
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e', 
    'success': '#2ca02c',
    'warning': '#d62728',
    'info': '#9467bd',
    'temperature': '#ff4444',
    'prediction': '#4444ff'
}

# Streamlit configuration
STREAMLIT_CONFIG = {
    'page_title': 'Hanoi Weather Forecasting',
    'page_icon': '🌡️',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'handlers': ['console', 'file']
}

# API configuration (if using external weather APIs)
API_CONFIG = {
    'timeout': 30,
    'retry_attempts': 3,
    'rate_limit': 100  # requests per hour
}

def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR,
        DAILY_PROCESSED_DIR, HOURLY_PROCESSED_DIR,
        MODELS_DIR, DAILY_MODELS_DIR, HOURLY_MODELS_DIR,
        OUTPUTS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def get_model_save_path(model_name: str, is_hourly: bool = False) -> Path:
    """
    Get the save path for a model.
    
    Args:
        model_name (str): Name of the model
        is_hourly (bool): Whether it's an hourly model
        
    Returns:
        Path: Path to save the model
    """
    if is_hourly:
        return HOURLY_MODELS_DIR / f"{model_name}.joblib"
    else:
        return DAILY_MODELS_DIR / f"{model_name}.joblib"

def get_feature_columns_path(is_hourly: bool = False) -> Path:
    """
    Get the path for feature columns file.
    
    Args:
        is_hourly (bool): Whether it's for hourly data
        
    Returns:
        Path: Path to feature columns file
    """
    if is_hourly:
        return HOURLY_MODELS_DIR / "feature_columns_hourly.joblib"
    else:
        return DAILY_MODELS_DIR / "feature_columns.joblib"

def get_metadata_path(is_hourly: bool = False) -> Path:
    """
    Get the path for model metadata file.
    
    Args:
        is_hourly (bool): Whether it's for hourly data
        
    Returns:
        Path: Path to metadata file
    """
    if is_hourly:
        return HOURLY_MODELS_DIR / "model_metadata_hourly.json"
    else:
        return DAILY_MODELS_DIR / "model_metadata.json"