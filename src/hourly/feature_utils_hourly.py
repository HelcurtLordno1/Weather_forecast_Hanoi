"""
Hourly Feature Engineering Utilities for Hanoi Weather Forecasting

Advanced feature engineering specifically designed for hourly weather data
with enhanced temporal patterns and diurnal cycles.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
import sys
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder

# Add shared utilities to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
from config import FORECAST_HOURS_HOURLY, CATEGORICAL_FEATURES, NUMERIC_FEATURES

# Set up logging
logger = logging.getLogger(__name__)


class HourlyFeatureEngineering:
    """
    Comprehensive feature engineering for hourly weather data.
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive temporal features for hourly data.
        
        Args:
            df (pd.DataFrame): Hourly weather data with datetime column
            
        Returns:
            pd.DataFrame: Data with temporal features
        """
        df_temp = df.copy()
        
        # Basic temporal components
        df_temp['hour'] = df_temp['datetime'].dt.hour
        df_temp['day'] = df_temp['datetime'].dt.day
        df_temp['month'] = df_temp['datetime'].dt.month
        df_temp['year'] = df_temp['datetime'].dt.year
        df_temp['day_of_week'] = df_temp['datetime'].dt.dayofweek
        df_temp['day_of_year'] = df_temp['datetime'].dt.dayofyear
        df_temp['week_of_year'] = df_temp['datetime'].dt.isocalendar().week
        
        # Cyclical encoding for better ML performance
        # Hour (24-hour cycle)
        df_temp['hour_sin'] = np.sin(2 * np.pi * df_temp['hour'] / 24)
        df_temp['hour_cos'] = np.cos(2 * np.pi * df_temp['hour'] / 24)
        
        # Day of week (7-day cycle)
        df_temp['dow_sin'] = np.sin(2 * np.pi * df_temp['day_of_week'] / 7)
        df_temp['dow_cos'] = np.cos(2 * np.pi * df_temp['day_of_week'] / 7)
        
        # Month (12-month cycle)
        df_temp['month_sin'] = np.sin(2 * np.pi * df_temp['month'] / 12)
        df_temp['month_cos'] = np.cos(2 * np.pi * df_temp['month'] / 12)
        
        # Day of year (365-day cycle)
        df_temp['doy_sin'] = np.sin(2 * np.pi * df_temp['day_of_year'] / 365.25)
        df_temp['doy_cos'] = np.cos(2 * np.pi * df_temp['day_of_year'] / 365.25)
        
        # Time of day categories
        df_temp['is_night'] = ((df_temp['hour'] >= 22) | (df_temp['hour'] <= 5)).astype(int)
        df_temp['is_dawn'] = ((df_temp['hour'] >= 5) & (df_temp['hour'] <= 7)).astype(int)
        df_temp['is_morning'] = ((df_temp['hour'] >= 8) & (df_temp['hour'] <= 11)).astype(int)
        df_temp['is_noon'] = ((df_temp['hour'] >= 11) & (df_temp['hour'] <= 13)).astype(int)
        df_temp['is_afternoon'] = ((df_temp['hour'] >= 14) & (df_temp['hour'] <= 17)).astype(int)
        df_temp['is_evening'] = ((df_temp['hour'] >= 18) & (df_temp['hour'] <= 21)).astype(int)
        
        # Peak temperature periods
        df_temp['is_peak_heat'] = ((df_temp['hour'] >= 12) & (df_temp['hour'] <= 16)).astype(int)
        df_temp['is_coolest'] = ((df_temp['hour'] >= 3) & (df_temp['hour'] <= 7)).astype(int)
        
        # Weekend and workday patterns
        df_temp['is_weekend'] = (df_temp['day_of_week'] >= 5).astype(int)
        df_temp['is_workday'] = (df_temp['day_of_week'] < 5).astype(int)
        
        # Season indicators (Vietnam seasons)
        df_temp['is_spring'] = ((df_temp['month'] >= 2) & (df_temp['month'] <= 4)).astype(int)
        df_temp['is_summer'] = ((df_temp['month'] >= 5) & (df_temp['month'] <= 7)).astype(int)
        df_temp['is_autumn'] = ((df_temp['month'] >= 8) & (df_temp['month'] <= 10)).astype(int)
        df_temp['is_winter'] = ((df_temp['month'] >= 11) | (df_temp['month'] <= 1)).astype(int)
        
        logger.info(f"Created {len(df_temp.columns) - len(df.columns)} temporal features")
        return df_temp
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'temp', 
                           lag_hours: List[int] = None) -> pd.DataFrame:
        """
        Create lagged features for time series forecasting.
        
        Args:
            df (pd.DataFrame): Input data
            target_col (str): Target column for lagging
            lag_hours (List[int]): List of lag hours
            
        Returns:
            pd.DataFrame: Data with lag features
        """
        if lag_hours is None:
            lag_hours = [1, 2, 3, 6, 12, 24, 48, 72, 96, 120]  # Up to 5 days
        
        df_lag = df.copy()
        
        # Temperature lag features
        for lag in lag_hours:
            if lag <= len(df_lag):
                df_lag[f'{target_col}_lag_{lag}h'] = df_lag[target_col].shift(lag)
        
        # Other important weather variables lag features
        important_vars = ['humidity', 'sealevelpressure', 'windspeed', 'cloudcover']
        short_lags = [1, 3, 6, 12, 24]
        
        for var in important_vars:
            if var in df_lag.columns:
                for lag in short_lags:
                    if lag <= len(df_lag):
                        df_lag[f'{var}_lag_{lag}h'] = df_lag[var].shift(lag)
        
        logger.info(f"Created lag features for {len(lag_hours)} hours")
        return df_lag
    
    def create_rolling_features(self, df: pd.DataFrame, target_col: str = 'temp',
                               windows: List[int] = None) -> pd.DataFrame:
        """
        Create rolling statistical features.
        
        Args:
            df (pd.DataFrame): Input data
            target_col (str): Target column for rolling stats
            windows (List[int]): List of window sizes (in hours)
            
        Returns:
            pd.DataFrame: Data with rolling features
        """
        if windows is None:
            windows = [3, 6, 12, 24, 48, 72]  # 3h to 3 days
        
        df_roll = df.copy()
        
        for window in windows:
            if window <= len(df_roll):
                # Temperature rolling statistics
                df_roll[f'{target_col}_rolling_mean_{window}h'] = df_roll[target_col].rolling(window=window).mean()
                df_roll[f'{target_col}_rolling_std_{window}h'] = df_roll[target_col].rolling(window=window).std()
                df_roll[f'{target_col}_rolling_min_{window}h'] = df_roll[target_col].rolling(window=window).min()
                df_roll[f'{target_col}_rolling_max_{window}h'] = df_roll[target_col].rolling(window=window).max()
                df_roll[f'{target_col}_rolling_range_{window}h'] = (df_roll[f'{target_col}_rolling_max_{window}h'] - 
                                                                   df_roll[f'{target_col}_rolling_min_{window}h'])
                
                # Other variables rolling statistics (shorter windows)
                if window <= 24:
                    for var in ['humidity', 'sealevelpressure', 'windspeed']:
                        if var in df_roll.columns:
                            df_roll[f'{var}_rolling_mean_{window}h'] = df_roll[var].rolling(window=window).mean()
                            df_roll[f'{var}_rolling_std_{window}h'] = df_roll[var].rolling(window=window).std()
        
        logger.info(f"Created rolling features for {len(windows)} windows")
        return df_roll
    
    def create_change_features(self, df: pd.DataFrame, target_col: str = 'temp') -> pd.DataFrame:
        """
        Create change/difference features.
        
        Args:
            df (pd.DataFrame): Input data
            target_col (str): Target column for change features
            
        Returns:
            pd.DataFrame: Data with change features
        """
        df_change = df.copy()
        
        # Temperature change features
        change_periods = [1, 2, 3, 6, 12, 24]
        
        for period in change_periods:
            df_change[f'{target_col}_change_{period}h'] = df_change[target_col].diff(period)
            df_change[f'{target_col}_pct_change_{period}h'] = df_change[target_col].pct_change(period)
        
        # Other variables change features
        for var in ['humidity', 'sealevelpressure', 'windspeed', 'cloudcover']:
            if var in df_change.columns:
                df_change[f'{var}_change_1h'] = df_change[var].diff(1)
                df_change[f'{var}_change_6h'] = df_change[var].diff(6)
        
        # Temperature acceleration (second derivative)
        df_change[f'{target_col}_acceleration_1h'] = df_change[f'{target_col}_change_1h'].diff(1)
        df_change[f'{target_col}_acceleration_3h'] = df_change[f'{target_col}_change_3h'].diff(3)
        
        logger.info("Created change and acceleration features")
        return df_change
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between weather variables.
        
        Args:
            df (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with interaction features
        """
        df_interact = df.copy()
        
        # Temperature-humidity interactions
        if 'temp' in df_interact.columns and 'humidity' in df_interact.columns:
            df_interact['temp_humidity_ratio'] = df_interact['temp'] / (df_interact['humidity'] + 1e-6)
            df_interact['heat_index_approx'] = df_interact['temp'] + 0.5 * df_interact['humidity']
        
        # Wind-temperature interactions
        if 'temp' in df_interact.columns and 'windspeed' in df_interact.columns:
            df_interact['wind_chill_approx'] = df_interact['temp'] - 0.7 * df_interact['windspeed']
            df_interact['temp_windspeed_product'] = df_interact['temp'] * df_interact['windspeed']
        
        # Pressure-temperature interactions
        if 'temp' in df_interact.columns and 'sealevelpressure' in df_interact.columns:
            df_interact['temp_pressure_ratio'] = df_interact['temp'] / (df_interact['sealevelpressure'] / 1000)
        
        # Solar radiation and cloud interactions
        if 'solarradiation' in df_interact.columns and 'cloudcover' in df_interact.columns:
            df_interact['solar_cloud_ratio'] = df_interact['solarradiation'] / (df_interact['cloudcover'] + 1e-6)
            
        # Visibility and humidity interactions
        if 'visibility' in df_interact.columns and 'humidity' in df_interact.columns:
            df_interact['visibility_humidity_ratio'] = df_interact['visibility'] / (df_interact['humidity'] + 1e-6)
        
        logger.info(f"Created {len(df_interact.columns) - len(df.columns)} interaction features")
        return df_interact
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical weather features.
        
        Args:
            df (pd.DataFrame): Input data
            fit (bool): Whether to fit encoders or use existing ones
            
        Returns:
            pd.DataFrame: Data with encoded categorical features
        """
        df_encoded = df.copy()
        
        # Weather condition encoding
        if 'conditions' in df_encoded.columns:
            if fit:
                self.encoders['conditions'] = LabelEncoder()
                df_encoded['conditions_encoded'] = self.encoders['conditions'].fit_transform(
                    df_encoded['conditions'].fillna('Unknown')
                )
            else:
                df_encoded['conditions_encoded'] = self.encoders['conditions'].transform(
                    df_encoded['conditions'].fillna('Unknown')
                )
        
        # Icon encoding (simplified weather state)
        if 'icon' in df_encoded.columns:
            if fit:
                self.encoders['icon'] = LabelEncoder()
                df_encoded['icon_encoded'] = self.encoders['icon'].fit_transform(
                    df_encoded['icon'].fillna('clear-day')
                )
            else:
                df_encoded['icon_encoded'] = self.encoders['icon'].transform(
                    df_encoded['icon'].fillna('clear-day')
                )
        
        # Precipitation type encoding
        if 'preciptype' in df_encoded.columns:
            # One-hot encode precipitation type
            precip_dummies = pd.get_dummies(df_encoded['preciptype'], prefix='precip', dummy_na=True)
            df_encoded = pd.concat([df_encoded, precip_dummies], axis=1)
        
        logger.info("Encoded categorical features")
        return df_encoded
    
    def create_weather_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create custom weather comfort and intensity indices.
        
        Args:
            df (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with weather indices
        """
        df_indices = df.copy()
        
        # Comfort index (simplified)
        if all(col in df_indices.columns for col in ['temp', 'humidity', 'windspeed']):
            df_indices['comfort_index'] = (
                (df_indices['temp'] - 20) ** 2 +  # Optimal temp around 20°C
                (df_indices['humidity'] - 50) ** 2 / 100 +  # Optimal humidity around 50%
                df_indices['windspeed'] ** 2 / 10  # Light wind preferred
            )
        
        # Weather severity index
        if all(col in df_indices.columns for col in ['windspeed', 'precip', 'cloudcover']):
            df_indices['weather_severity'] = (
                df_indices['windspeed'] / 10 +
                df_indices['precip'] * 2 +
                df_indices['cloudcover'] / 50
            )
        
        # Temperature stress index
        if 'temp' in df_indices.columns:
            df_indices['temp_stress'] = np.where(
                df_indices['temp'] > 30, (df_indices['temp'] - 30) ** 1.5,
                np.where(df_indices['temp'] < 10, (10 - df_indices['temp']) ** 1.5, 0)
            )
        
        logger.info("Created weather indices")
        return df_indices
    
    def create_diurnal_patterns(self, df: pd.DataFrame, target_col: str = 'temp') -> pd.DataFrame:
        """
        Create diurnal (daily cycle) pattern features.
        
        Args:
            df (pd.DataFrame): Input data
            target_col (str): Target column
            
        Returns:
            pd.DataFrame: Data with diurnal features
        """
        df_diurnal = df.copy()
        
        # Daily temperature statistics for each day
        df_diurnal['date'] = df_diurnal['datetime'].dt.date
        daily_stats = df_diurnal.groupby('date')[target_col].agg([
            'min', 'max', 'mean', 'std'
        ]).add_prefix(f'daily_{target_col}_')
        
        # Merge back to hourly data
        df_diurnal = df_diurnal.merge(daily_stats, left_on='date', right_index=True, how='left')
        
        # Diurnal temperature pattern features
        df_diurnal[f'{target_col}_diurnal_range'] = (df_diurnal[f'daily_{target_col}_max'] - 
                                                    df_diurnal[f'daily_{target_col}_min'])
        df_diurnal[f'{target_col}_from_daily_mean'] = (df_diurnal[target_col] - 
                                                      df_diurnal[f'daily_{target_col}_mean'])
        df_diurnal[f'{target_col}_normalized_in_day'] = (
            (df_diurnal[target_col] - df_diurnal[f'daily_{target_col}_min']) / 
            (df_diurnal[f'daily_{target_col}_max'] - df_diurnal[f'daily_{target_col}_min'] + 1e-6)
        )
        
        # Remove temporary date column
        df_diurnal = df_diurnal.drop('date', axis=1)
        
        logger.info("Created diurnal pattern features")
        return df_diurnal
    
    def prepare_features_for_forecast(self, df: pd.DataFrame, target_col: str = 'temp',
                                    forecast_horizon: int = FORECAST_HOURS_HOURLY) -> pd.DataFrame:
        """
        Create target variable for forecasting.
        
        Args:
            df (pd.DataFrame): Feature-engineered data
            target_col (str): Target column
            forecast_horizon (int): Hours ahead to predict
            
        Returns:
            pd.DataFrame: Data ready for model training
        """
        df_forecast = df.copy()
        
        # Create future target
        df_forecast[f'{target_col}_target_{forecast_horizon}h'] = df_forecast[target_col].shift(-forecast_horizon)
        
        # Remove rows where target is NaN (last forecast_horizon rows)
        df_forecast = df_forecast.dropna(subset=[f'{target_col}_target_{forecast_horizon}h'])
        
        logger.info(f"Prepared data for {forecast_horizon}h ahead forecasting")
        logger.info(f"Final dataset shape: {df_forecast.shape}")
        
        return df_forecast
    
    def get_feature_columns(self, df: pd.DataFrame, exclude_cols: List[str] = None) -> List[str]:
        """
        Get list of feature columns (excluding target, datetime, etc.).
        
        Args:
            df (pd.DataFrame): DataFrame with features
            exclude_cols (List[str]): Additional columns to exclude
            
        Returns:
            List[str]: List of feature column names
        """
        if exclude_cols is None:
            exclude_cols = []
        
        exclude_list = [
            'datetime', 'name', 'address', 'resolvedAddress', 
            'latitude', 'longitude', 'source'
        ] + exclude_cols
        
        # Exclude target columns
        target_cols = [col for col in df.columns if '_target_' in col]
        exclude_list.extend(target_cols)
        
        feature_cols = [col for col in df.columns if col not in exclude_list]
        
        logger.info(f"Selected {len(feature_cols)} feature columns")
        return feature_cols
    
    def fit_scalers(self, df: pd.DataFrame, feature_cols: List[str]) -> None:
        """
        Fit scalers for numeric features.
        
        Args:
            df (pd.DataFrame): Training data
            feature_cols (List[str]): Feature columns to scale
        """
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
        
        self.scalers['standard'] = StandardScaler()
        self.scalers['standard'].fit(df[numeric_cols])
        
        logger.info(f"Fitted scalers for {len(numeric_cols)} numeric features")
    
    def transform_features(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        Transform features using fitted scalers.
        
        Args:
            df (pd.DataFrame): Data to transform
            feature_cols (List[str]): Feature columns
            
        Returns:
            pd.DataFrame: Transformed data
        """
        df_scaled = df.copy()
        numeric_cols = df_scaled[feature_cols].select_dtypes(include=[np.number]).columns
        
        if 'standard' in self.scalers:
            df_scaled[numeric_cols] = self.scalers['standard'].transform(df_scaled[numeric_cols])
        
        return df_scaled