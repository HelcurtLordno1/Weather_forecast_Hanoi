"""
Hourly Data Utilities Module for Hanoi Weather Forecasting

This module contains functions for data loading, cleaning, and preprocessing 
specifically designed for HOURLY weather forecasting with enhanced temporal features.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
import sys
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Add shared utilities to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
from config import HOURLY_DATA_FILE, HOURLY_PROCESSED_DIR, FORECAST_HOURS_HOURLY

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_hanoi_hourly_weather_data(file_path: str) -> pd.DataFrame:
    """
    Load Hanoi hourly weather data with proper datetime parsing and sorting.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded and sorted hourly weather data
    """
    try:
        df = pd.read_csv(file_path)
        
        # Convert datetime and sort
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Add hour-specific information
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        df['year'] = df['datetime'].dt.year
        
        logger.info(f"Successfully loaded {len(df)} hourly records from {file_path}")
        logger.info(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        logger.info(f"Temperature range: {df['temp'].min():.1f}°C to {df['temp'].max():.1f}°C")
        logger.info(f"Hours covered: {df['hour'].nunique()} unique hours")
        
        return df
    except Exception as e:
        logger.error(f"Error loading hourly data from {file_path}: {str(e)}")
        raise


def load_raw_hourly_data(file_path: str = None) -> pd.DataFrame:
    """
    Load raw hourly data using default path from config.
    
    Args:
        file_path (str, optional): Custom file path
        
    Returns:
        pd.DataFrame: Loaded hourly weather data
    """
    if file_path is None:
        file_path = str(HOURLY_DATA_FILE)
    
    return load_hanoi_hourly_weather_data(file_path)


def check_hourly_data_quality(df: pd.DataFrame) -> Dict[str, any]:
    """
    Comprehensive data quality check for hourly weather data.
    
    Args:
        df (pd.DataFrame): Hourly weather dataframe
        
    Returns:
        Dict: Data quality summary
    """
    quality_report = {
        'total_records': len(df),
        'date_range': {
            'start': df['datetime'].min(),
            'end': df['datetime'].max(),
            'duration_days': (df['datetime'].max() - df['datetime'].min()).days
        },
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        'duplicates': df.duplicated().sum(),
        'hourly_coverage': {
            'unique_hours': df['hour'].nunique(),
            'expected_hours': 24,
            'missing_hours': list(set(range(24)) - set(df['hour'].unique()))
        }
    }
    
    # Check for gaps in hourly sequence
    df_sorted = df.sort_values('datetime')
    time_diffs = df_sorted['datetime'].diff()
    expected_diff = timedelta(hours=1)
    gaps = time_diffs[time_diffs > expected_diff]
    
    quality_report['temporal_gaps'] = {
        'number_of_gaps': len(gaps),
        'largest_gap_hours': gaps.max().total_seconds() / 3600 if len(gaps) > 0 else 0,
        'gap_locations': df_sorted[time_diffs > expected_diff]['datetime'].tolist()
    }
    
    # Temperature statistics
    quality_report['temperature_stats'] = {
        'min': df['temp'].min(),
        'max': df['temp'].max(),
        'mean': df['temp'].mean(),
        'std': df['temp'].std(),
        'outliers_count': len(df[(df['temp'] < df['temp'].quantile(0.01)) | 
                                 (df['temp'] > df['temp'].quantile(0.99))])
    }
    
    logger.info("Hourly data quality check completed")
    logger.info(f"Total records: {quality_report['total_records']}")
    logger.info(f"Date range: {quality_report['date_range']['start']} to {quality_report['date_range']['end']}")
    logger.info(f"Missing values: {sum(quality_report['missing_values'].values())} total")
    logger.info(f"Temporal gaps: {quality_report['temporal_gaps']['number_of_gaps']}")
    
    return quality_report


def create_hourly_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create hourly-specific temporal features.
    
    Args:
        df (pd.DataFrame): Raw hourly data
        
    Returns:
        pd.DataFrame: Data with hourly features
    """
    df_features = df.copy()
    
    # Cyclical encoding for hours (24-hour cycle)
    df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
    df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
    
    # Day of week cyclical encoding
    df_features['dow_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
    df_features['dow_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
    
    # Month cyclical encoding
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
    
    # Time of day categories
    df_features['is_night'] = ((df_features['hour'] >= 22) | (df_features['hour'] <= 5)).astype(int)
    df_features['is_morning'] = ((df_features['hour'] >= 6) & (df_features['hour'] <= 11)).astype(int)
    df_features['is_afternoon'] = ((df_features['hour'] >= 12) & (df_features['hour'] <= 17)).astype(int)
    df_features['is_evening'] = ((df_features['hour'] >= 18) & (df_features['hour'] <= 21)).astype(int)
    
    # Peak hours
    df_features['is_peak_heat'] = ((df_features['hour'] >= 12) & (df_features['hour'] <= 16)).astype(int)
    df_features['is_coolest'] = ((df_features['hour'] >= 3) & (df_features['hour'] <= 7)).astype(int)
    
    # Weekend indicator
    df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)
    
    logger.info("Created hourly temporal features")
    logger.info(f"Added {len(df_features.columns) - len(df.columns)} new features")
    
    return df_features


def handle_missing_values_hourly(df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
    """
    Handle missing values in hourly weather data.
    
    Args:
        df (pd.DataFrame): Hourly weather data
        method (str): Method to handle missing values
        
    Returns:
        pd.DataFrame: Data with handled missing values
    """
    df_clean = df.copy()
    
    # Numeric columns for interpolation
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    if method == 'interpolate':
        # Time-based interpolation for numeric columns
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                # Forward fill first, then interpolate, then backward fill
                df_clean[col] = df_clean[col].fillna(method='ffill').interpolate(method='time').fillna(method='bfill')
        
        # Forward fill for categorical columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in ['datetime']:
                df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
                
    elif method == 'rolling_mean':
        # Use rolling mean for numeric columns
        window_size = 24  # 24 hours
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col] = df_clean[col].fillna(df_clean[col].rolling(window=window_size, center=True).mean())
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())  # Fill remaining with overall mean
    
    logger.info(f"Handled missing values using {method} method")
    logger.info(f"Remaining missing values: {df_clean.isnull().sum().sum()}")
    
    return df_clean


def detect_hourly_outliers(df: pd.DataFrame, columns: List[str] = None, method: str = 'iqr') -> Dict[str, List[int]]:
    """
    Detect outliers in hourly weather data.
    
    Args:
        df (pd.DataFrame): Hourly weather data
        columns (List[str]): Columns to check for outliers
        method (str): Method for outlier detection
        
    Returns:
        Dict: Dictionary mapping column names to outlier indices
    """
    if columns is None:
        columns = ['temp', 'humidity', 'windspeed', 'sealevelpressure']
    
    outliers = {}
    
    for col in columns:
        if col in df.columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
                outliers[col] = outlier_indices
                
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_indices = df[z_scores > 3].index.tolist()
                outliers[col] = outlier_indices
    
    total_outliers = sum(len(indices) for indices in outliers.values())
    logger.info(f"Detected {total_outliers} outliers using {method} method")
    
    return outliers


def prepare_hourly_time_series(df: pd.DataFrame, target_col: str = 'temp', 
                              forecast_horizon: int = FORECAST_HOURS_HOURLY) -> pd.DataFrame:
    """
    Prepare hourly time series data for forecasting.
    
    Args:
        df (pd.DataFrame): Hourly weather data
        target_col (str): Target column for forecasting
        forecast_horizon (int): Number of hours to forecast ahead
        
    Returns:
        pd.DataFrame: Prepared time series data
    """
    df_ts = df.copy()
    
    # Create lagged features (previous hours)
    lag_hours = [1, 2, 3, 6, 12, 24, 48, 72]  # 1h, 2h, 3h, 6h, 12h, 1d, 2d, 3d
    
    for lag in lag_hours:
        if lag <= len(df_ts):
            df_ts[f'{target_col}_lag_{lag}h'] = df_ts[target_col].shift(lag)
    
    # Create rolling statistics
    windows = [3, 6, 12, 24, 48]  # 3h, 6h, 12h, 1d, 2d
    
    for window in windows:
        if window <= len(df_ts):
            df_ts[f'{target_col}_rolling_mean_{window}h'] = df_ts[target_col].rolling(window=window).mean()
            df_ts[f'{target_col}_rolling_std_{window}h'] = df_ts[target_col].rolling(window=window).std()
            df_ts[f'{target_col}_rolling_min_{window}h'] = df_ts[target_col].rolling(window=window).min()
            df_ts[f'{target_col}_rolling_max_{window}h'] = df_ts[target_col].rolling(window=window).max()
    
    # Create future target (for training)
    df_ts[f'{target_col}_future_{forecast_horizon}h'] = df_ts[target_col].shift(-forecast_horizon)
    
    # Create change features
    df_ts[f'{target_col}_change_1h'] = df_ts[target_col].diff(1)
    df_ts[f'{target_col}_change_3h'] = df_ts[target_col].diff(3)
    df_ts[f'{target_col}_change_6h'] = df_ts[target_col].diff(6)
    df_ts[f'{target_col}_change_24h'] = df_ts[target_col].diff(24)
    
    logger.info(f"Prepared hourly time series with {forecast_horizon}h forecast horizon")
    logger.info(f"Created {len(df_ts.columns) - len(df.columns)} new time series features")
    
    return df_ts


def split_hourly_data(df: pd.DataFrame, train_ratio: float = 0.7, 
                     val_ratio: float = 0.15, test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split hourly data into train, validation, and test sets temporally.
    
    Args:
        df (pd.DataFrame): Hourly weather data
        train_ratio (float): Ratio for training data
        val_ratio (float): Ratio for validation data
        test_ratio (float): Ratio for test data
        
    Returns:
        Tuple: train_df, val_df, test_df
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()
    
    logger.info(f"Split hourly data: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    logger.info(f"Train period: {train_df['datetime'].min()} to {train_df['datetime'].max()}")
    logger.info(f"Val period: {val_df['datetime'].min()} to {val_df['datetime'].max()}")
    logger.info(f"Test period: {test_df['datetime'].min()} to {test_df['datetime'].max()}")
    
    return train_df, val_df, test_df


def save_processed_hourly_data(df: pd.DataFrame, filename: str) -> None:
    """
    Save processed hourly data to the processed directory.
    
    Args:
        df (pd.DataFrame): Processed data
        filename (str): Output filename
    """
    os.makedirs(HOURLY_PROCESSED_DIR, exist_ok=True)
    output_path = HOURLY_PROCESSED_DIR / filename
    
    df.to_csv(output_path, index=False)
    logger.info(f"Saved processed hourly data to {output_path}")
    logger.info(f"Shape: {df.shape}")


def load_processed_hourly_data(filename: str) -> pd.DataFrame:
    """
    Load processed hourly data from the processed directory.
    
    Args:
        filename (str): Filename to load
        
    Returns:
        pd.DataFrame: Loaded processed data
    """
    file_path = HOURLY_PROCESSED_DIR / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"Processed file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    logger.info(f"Loaded processed hourly data from {file_path}")
    logger.info(f"Shape: {df.shape}")
    
    return df