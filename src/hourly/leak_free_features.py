
"""
Helper functions for leak-free rolling statistics and seasonal anomalies.
These functions ensure train/test data integrity.
"""

import numpy as np
import pandas as pd

def create_rolling_features_safe(df_train, df_test, variables, windows):
    """
    Create rolling features WITHOUT data leakage

    Parameters:
    -----------
    df_train : DataFrame with datetime index
    df_test : DataFrame with datetime index
    variables : list of column names to create rolling features for
    windows : list of window sizes (in hours)

    Returns:
    --------
    df_train, df_test with new rolling feature columns
    """

    df_train = df_train.copy()
    df_test = df_test.copy()

    for var in variables:
        if var not in df_train.columns:
            continue

        for window in windows:
            print(f"   Creating {var}_rolling_{window}h features...")

            # Calculate rolling stats on TRAIN ONLY
            train_rolling_mean = df_train[var].rolling(window=window, min_periods=1).mean()
            train_rolling_std = df_train[var].rolling(window=window, min_periods=1).std()
            train_rolling_min = df_train[var].rolling(window=window, min_periods=1).min()
            train_rolling_max = df_train[var].rolling(window=window, min_periods=1).max()

            # Apply to train
            df_train[f'{var}_rolling_{window}h_mean'] = train_rolling_mean
            df_train[f'{var}_rolling_{window}h_std'] = train_rolling_std
            df_train[f'{var}_rolling_{window}h_min'] = train_rolling_min
            df_train[f'{var}_rolling_{window}h_max'] = train_rolling_max

            # For test: use expanding window from last train values
            last_train_window = df_train[var].iloc[-window:].tolist()

            test_means, test_stds, test_mins, test_maxs = [], [], [], []
            buffer = last_train_window.copy()

            for value in df_test[var]:
                buffer.append(value)
                if len(buffer) > window:
                    buffer.pop(0)

                test_means.append(np.mean(buffer))
                test_stds.append(np.std(buffer))
                test_mins.append(np.min(buffer))
                test_maxs.append(np.max(buffer))

            df_test[f'{var}_rolling_{window}h_mean'] = test_means
            df_test[f'{var}_rolling_{window}h_std'] = test_stds
            df_test[f'{var}_rolling_{window}h_min'] = test_mins
            df_test[f'{var}_rolling_{window}h_max'] = test_maxs

    return df_train, df_test


def create_seasonal_features_safe(df_train, df_test):
    """
    Create seasonal anomalies WITHOUT leakage
    Uses only training data to compute baselines

    Parameters:
    -----------
    df_train : DataFrame with datetime index and 'month', 'hour' columns
    df_test : DataFrame with datetime index and 'month', 'hour' columns

    Returns:
    --------
    df_train, df_test with seasonal anomaly columns
    """

    df_train = df_train.copy()
    df_test = df_test.copy()

    # Calculate seasonal baselines ONLY from training data
    seasonal_vars = ['temp', 'humidity', 'sealevelpressure']
    seasonal_baselines = df_train.groupby(['month', 'hour'])[seasonal_vars].mean().reset_index()
    seasonal_baselines.columns = ['month', 'hour'] + [f'{v}_seasonal_mean' for v in seasonal_vars]

    # Merge with train
    df_train_temp = df_train.reset_index()
    df_train_merged = df_train_temp.merge(seasonal_baselines, on=['month', 'hour'], how='left')

    for var in seasonal_vars:
        if var in df_train.columns:
            df_train[f'{var}_seasonal_anomaly'] = (
                df_train_merged[var].values - df_train_merged[f'{var}_seasonal_mean'].values
            )

    # Merge with test (using train baselines)
    df_test_temp = df_test.reset_index()
    df_test_merged = df_test_temp.merge(seasonal_baselines, on=['month', 'hour'], how='left')

    # Fill missing combinations with overall train mean
    for var in seasonal_vars:
        if var in df_test.columns:
            seasonal_col = f'{var}_seasonal_mean'
            if df_test_merged[seasonal_col].isnull().any():
                df_test_merged[seasonal_col].fillna(df_train[var].mean(), inplace=True)

            df_test[f'{var}_seasonal_anomaly'] = (
                df_test_merged[var].values - df_test_merged[seasonal_col].values
            )

    return df_train, df_test, seasonal_baselines


print("✅ Helper functions ready for notebook 03!")
