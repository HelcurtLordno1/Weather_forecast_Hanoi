"""
Feature Engineering Utilities for Hanoi Temperature Forecasting

This module contains functions for creating features for temperature prediction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_lag_features(df: pd.DataFrame, target_column: str, lags: list = [1, 2, 3, 6, 12, 24]) -> pd.DataFrame:
    """
    Create lag features for time series prediction.
    
    Args:
        df (pd.DataFrame): Input data
        target_column (str): Name of target column
        lags (list): List of lag periods
        
    Returns:
        pd.DataFrame: Data with lag features
    """
    df_features = df.copy()
    
    for lag in lags:
        df_features[f'{target_column}_lag_{lag}'] = df_features[target_column].shift(lag)
    
    logger.info(f"Created {len(lags)} lag features")
    return df_features


def create_rolling_features(df: pd.DataFrame, target_column: str, windows: list = [3, 6, 12, 24]) -> pd.DataFrame:
    """
    Create rolling statistical features.
    
    Args:
        df (pd.DataFrame): Input data
        target_column (str): Name of target column
        windows (list): List of window sizes
        
    Returns:
        pd.DataFrame: Data with rolling features
    """
    df_features = df.copy()
    
    for window in windows:
        df_features[f'{target_column}_rolling_mean_{window}'] = df_features[target_column].rolling(window=window).mean()
        df_features[f'{target_column}_rolling_std_{window}'] = df_features[target_column].rolling(window=window).std()
        df_features[f'{target_column}_rolling_min_{window}'] = df_features[target_column].rolling(window=window).min()
        df_features[f'{target_column}_rolling_max_{window}'] = df_features[target_column].rolling(window=window).max()
    
    logger.info(f"Created rolling features for {len(windows)} windows")
    return df_features


def create_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create cyclical features for time components.
    
    Args:
        df (pd.DataFrame): Input data with time components
        
    Returns:
        pd.DataFrame: Data with cyclical features
    """
    df_features = df.copy()
    
    # Hour cyclical features (24 hours)
    if 'hour' in df_features.columns:
        df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
        df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
    
    # Month cyclical features (12 months)
    if 'month' in df_features.columns:
        df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
    
    # Day of year cyclical features (365 days)
    if 'day_of_year' in df_features.columns:
        df_features['day_of_year_sin'] = np.sin(2 * np.pi * df_features['day_of_year'] / 365)
        df_features['day_of_year_cos'] = np.cos(2 * np.pi * df_features['day_of_year'] / 365)
    
    logger.info("Created cyclical time features")
    return df_features


def create_season_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create season-based features.
    
    Args:
        df (pd.DataFrame): Input data with month column
        
    Returns:
        pd.DataFrame: Data with season features
    """
    df_features = df.copy()
    
    if 'month' in df_features.columns:
        def get_season(month):
            if month in [12, 1, 2]:
                return 'winter'
            elif month in [3, 4, 5]:
                return 'spring'
            elif month in [6, 7, 8]:
                return 'summer'
            else:
                return 'autumn'
        
        df_features['season'] = df_features['month'].apply(get_season)
        
        # One-hot encode seasons
        season_dummies = pd.get_dummies(df_features['season'], prefix='season')
        df_features = pd.concat([df_features, season_dummies], axis=1)
    
    logger.info("Created season features")
    return df_features


def create_temperature_difference_features(df: pd.DataFrame, temp_column: str) -> pd.DataFrame:
    """
    Create temperature difference features.
    
    Args:
        df (pd.DataFrame): Input data
        temp_column (str): Name of temperature column
        
    Returns:
        pd.DataFrame: Data with temperature difference features
    """
    df_features = df.copy()
    
    # Temperature differences
    df_features[f'{temp_column}_diff_1h'] = df_features[temp_column].diff(1)
    df_features[f'{temp_column}_diff_3h'] = df_features[temp_column].diff(3)
    df_features[f'{temp_column}_diff_6h'] = df_features[temp_column].diff(6)
    df_features[f'{temp_column}_diff_24h'] = df_features[temp_column].diff(24)
    
    logger.info("Created temperature difference features")
    return df_features


def scale_features(df: pd.DataFrame, feature_columns: list, scaler_type: str = 'standard') -> tuple:
    """
    Scale features using specified scaler.
    
    Args:
        df (pd.DataFrame): Input data
        feature_columns (list): List of columns to scale
        scaler_type (str): Type of scaler ('standard' or 'minmax')
        
    Returns:
        tuple: (scaled_data, fitted_scaler)
    """
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("scaler_type must be 'standard' or 'minmax'")
    
    df_scaled = df.copy()
    df_scaled[feature_columns] = scaler.fit_transform(df[feature_columns])
    
    logger.info(f"Scaled {len(feature_columns)} features using {scaler_type} scaler")
    return df_scaled, scaler


def prepare_features_for_modeling(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.
    
    Args:
        df (pd.DataFrame): Input data
        target_column (str): Name of target column
        
    Returns:
        pd.DataFrame: Data ready for modeling
    """
    logger.info("Starting complete feature engineering pipeline")
    
    # Create all features
    df_features = create_lag_features(df, target_column)
    df_features = create_rolling_features(df_features, target_column)
    df_features = create_cyclical_features(df_features)
    df_features = create_season_features(df_features)
    df_features = create_temperature_difference_features(df_features, target_column)
    
    # Drop rows with NaN values (from lag and rolling features)
    df_features = df_features.dropna()
    
    logger.info(f"Feature engineering completed. Final shape: {df_features.shape}")
    return df_features