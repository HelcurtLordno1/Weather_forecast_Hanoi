"""
Daily Data Utilities Module for Hanoi Weather Forecasting

This module contains functions for data loading, cleaning, and preprocessing 
specifically designed for DAILY weather forecasting with 33-feature dataset.
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
from config import DAILY_DATA_FILE, DAILY_PROCESSED_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_hanoi_weather_data(file_path: str) -> pd.DataFrame:
    """
    Load Hanoi weather data with proper datetime parsing and sorting.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded and sorted weather data
    """
    try:
        df = pd.read_csv(file_path)
        
        # Convert datetime and sort
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        
        logger.info(f"Successfully loaded {len(df)} records from {file_path}")
        logger.info(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        logger.info(f"Temperature range: {df['temp'].min():.1f}°C to {df['temp'].max():.1f}°C")
        
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        raise


def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Alias for load_hanoi_weather_data to maintain compatibility.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded and sorted weather data
    """
    return load_hanoi_weather_data(file_path)


def analyze_data_quality(df: pd.DataFrame) -> Dict:
    """
    Comprehensive data quality analysis for weather dataset.
    
    Args:
        df (pd.DataFrame): Weather data
        
    Returns:
        Dict: Data quality report
    """
    quality_report = {
        'shape': df.shape,
        'date_range': (df['datetime'].min(), df['datetime'].max()),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'data_types': df.dtypes.to_dict(),
        'duplicates': df.duplicated().sum(),
        'temperature_stats': df['temp'].describe().to_dict()
    }
    
    # Calculate completeness
    total_cells = len(df) * len(df.columns)
    missing_cells = df.isnull().sum().sum()
    quality_report['completeness_percentage'] = ((total_cells - missing_cells) / total_cells) * 100
    
    logger.info(f"Data quality analysis completed. Completeness: {quality_report['completeness_percentage']:.2f}%")
    
    return quality_report


def classify_weather_features(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Classify weather features into meaningful categories.
    
    Args:
        df (pd.DataFrame): Weather dataframe
        
    Returns:
        Dict: Feature categories
    """
    feature_types = {
        'temporal': ['datetime', 'sunrise', 'sunset'],
        'target': ['temp'],
        'temperature': ['tempmax', 'tempmin', 'feelslike', 'feelslikemax', 'feelslikemin', 'dew'],
        'atmospheric': ['humidity', 'sealevelpressure', 'cloudcover', 'visibility'],
        'precipitation': ['precip', 'precipprob', 'precipcover', 'snow', 'snowdepth'],
        'wind': ['windspeed', 'windgust', 'winddir'],
        'solar': ['solarradiation', 'solarenergy', 'uvindex'],
        'cyclical': ['moonphase'],
        'risk': ['severerisk'],
        'categorical': ['conditions', 'description', 'icon', 'preciptype'],
        'identifier': ['name', 'stations']
    }
    
    # Filter to only include features that exist in the dataframe
    available_features = {}
    for category, features in feature_types.items():
        available_features[category] = [f for f in features if f in df.columns]
    
    # All numerical features
    available_features['numerical'] = (
        available_features['temperature'] + available_features['atmospheric'] +
        available_features['precipitation'] + available_features['wind'] +
        available_features['solar'] + available_features['cyclical'] + 
        available_features['risk']
    )
    
    return available_features


def detect_outliers_comprehensive(df: pd.DataFrame, columns: List[str], 
                                method: str = 'iqr') -> Dict[str, Dict]:
    """
    Detect outliers in specified columns using various methods.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (List[str]): Columns to analyze for outliers
        method (str): Method to use ('iqr', 'zscore', 'isolation')
        
    Returns:
        Dict: Outlier analysis results
    """
    outlier_results = {}
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = df[z_scores > 3]
            lower_bound = df[col].mean() - 3 * df[col].std()
            upper_bound = df[col].mean() + 3 * df[col].std()
        
        outlier_results[col] = {
            'count': len(outliers),
            'percentage': len(outliers) / len(df) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_indices': outliers.index.tolist()
        }
    
    return outlier_results


def handle_missing_values(df: pd.DataFrame, strategy: Dict[str, str] = None) -> pd.DataFrame:
    """
    Handle missing values with different strategies for different feature types.
    
    Args:
        df (pd.DataFrame): Input dataframe
        strategy (Dict): Strategy for each feature type
        
    Returns:
        pd.DataFrame: Dataframe with handled missing values
    """
    if strategy is None:
        strategy = {
            'numerical': 'interpolate',  # Linear interpolation for time series
            'categorical': 'mode',       # Most frequent value
            'temporal': 'drop'           # Drop rows with missing dates
        }
    
    df_processed = df.copy()
    feature_types = classify_weather_features(df)
    
    # Handle numerical features
    numerical_cols = feature_types['numerical']
    for col in numerical_cols:
        if col in df_processed.columns and df_processed[col].isnull().any():
            if strategy.get('numerical') == 'interpolate':
                df_processed[col] = df_processed[col].interpolate(method='linear')
            elif strategy.get('numerical') == 'mean':
                df_processed[col].fillna(df_processed[col].mean(), inplace=True)
            elif strategy.get('numerical') == 'median':
                df_processed[col].fillna(df_processed[col].median(), inplace=True)
    
    # Handle categorical features
    categorical_cols = feature_types['categorical']
    for col in categorical_cols:
        if col in df_processed.columns and df_processed[col].isnull().any():
            if strategy.get('categorical') == 'mode':
                mode_value = df_processed[col].mode().iloc[0] if not df_processed[col].mode().empty else 'unknown'
                df_processed[col].fillna(mode_value, inplace=True)
    
    # Handle temporal features (usually drop rows)
    if 'datetime' in df_processed.columns and df_processed['datetime'].isnull().any():
        if strategy.get('temporal') == 'drop':
            df_processed = df_processed.dropna(subset=['datetime'])
    
    logger.info(f"Missing value handling completed. Shape: {df_processed.shape}")
    return df_processed


def scale_features(df: pd.DataFrame, feature_columns: List[str], 
                  scaler_type: str = 'standard') -> Tuple[pd.DataFrame, object]:
    """
    Scale numerical features using specified scaler.
    
    Args:
        df (pd.DataFrame): Input dataframe
        feature_columns (List[str]): Columns to scale
        scaler_type (str): Type of scaler ('standard', 'minmax', 'robust')
        
    Returns:
        Tuple: (scaled_dataframe, fitted_scaler)
    """
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("scaler_type must be 'standard', 'minmax', or 'robust'")
    
    df_scaled = df.copy()
    
    # Only scale columns that exist and are numerical
    available_columns = [col for col in feature_columns if col in df.columns]
    
    if available_columns:
        df_scaled[available_columns] = scaler.fit_transform(df[available_columns])
        logger.info(f"Scaled {len(available_columns)} features using {scaler_type} scaler")
    
    return df_scaled, scaler


def create_train_test_split_temporal(df: pd.DataFrame, test_size: float = 0.2, 
                                   val_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create temporal train/validation/test split for time series data.
    
    Args:
        df (pd.DataFrame): Input dataframe (must be sorted by datetime)
        test_size (float): Proportion for test set
        val_size (float): Proportion for validation set
        
    Returns:
        Tuple: (train_df, val_df, test_df)
    """
    n_samples = len(df)
    test_start = int(n_samples * (1 - test_size))
    val_start = int(n_samples * (1 - test_size - val_size))
    
    train_df = df.iloc[:val_start].copy()
    val_df = df.iloc[val_start:test_start].copy()
    test_df = df.iloc[test_start:].copy()
    
    logger.info(f"Temporal split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    logger.info(f"Train period: {train_df['datetime'].min()} to {train_df['datetime'].max()}")
    logger.info(f"Val period: {val_df['datetime'].min()} to {val_df['datetime'].max()}")
    logger.info(f"Test period: {test_df['datetime'].min()} to {test_df['datetime'].max()}")
    
    return train_df, val_df, test_df


def save_processed_data(df: pd.DataFrame, file_path: str, 
                       metadata: Optional[Dict] = None) -> None:
    """
    Save processed data with metadata.
    
    Args:
        df (pd.DataFrame): Processed dataframe
        file_path (str): Output file path
        metadata (Dict): Optional metadata to save alongside
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save main data
        df.to_csv(file_path, index=False)
        
        # Save metadata if provided
        if metadata:
            metadata_path = file_path.replace('.csv', '_metadata.json')
            import json
            with open(metadata_path, 'w') as f:
                # Convert non-serializable types
                metadata_serializable = {}
                for key, value in metadata.items():
                    if isinstance(value, (pd.Timestamp, datetime)):
                        metadata_serializable[key] = value.isoformat()
                    elif isinstance(value, np.integer):
                        metadata_serializable[key] = int(value)
                    elif isinstance(value, np.floating):
                        metadata_serializable[key] = float(value)
                    else:
                        metadata_serializable[key] = value
                
                json.dump(metadata_serializable, f, indent=2)
        
        logger.info(f"Processed data saved to {file_path}")
        if metadata:
            logger.info(f"Metadata saved to {metadata_path}")
            
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {str(e)}")
        raise


def get_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate comprehensive feature summary statistics.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Feature summary
    """
    summary_data = []
    
    for col in df.columns:
        if col == 'datetime':
            continue
            
        col_info = {
            'feature': col,
            'dtype': str(df[col].dtype),
            'missing_count': df[col].isnull().sum(),
            'missing_pct': (df[col].isnull().sum() / len(df)) * 100,
            'unique_count': df[col].nunique()
        }
        
        if df[col].dtype in ['int64', 'float64']:
            col_info.update({
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'q25': df[col].quantile(0.25),
                'q50': df[col].quantile(0.50),
                'q75': df[col].quantile(0.75)
            })
        else:
            col_info.update({
                'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                'most_frequent_count': df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0
            })
        
        summary_data.append(col_info)
    
    return pd.DataFrame(summary_data)