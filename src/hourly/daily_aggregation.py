
"""
Daily Aggregation Helper for 5-Day Temperature Forecasting

This module aggregates hourly weather data to daily level, preserving
rich hourly patterns while creating daily predictions.

Purpose: Enable 5-day ahead forecasting using hourly granularity data
"""

import pandas as pd
import numpy as np

def aggregate_hourly_to_daily_features(df_hourly):
    """
    Aggregate hourly features to daily level for 5-day forecasting

    This function converts hourly weather observations into daily-level features
    that capture diurnal patterns, intraday variability, and daily statistics.

    Parameters:
    -----------
    df_hourly : DataFrame
        Hourly features with datetime index

    Returns:
    --------
    df_daily : DataFrame
        Daily-aggregated features with date column
    """

    df = df_hourly.copy()
    df['date'] = df.index.date

    print(f"   📊 Aggregating {len(df)} hourly observations to daily...")

    # Define aggregation dictionary
    agg_dict = {}

    # Temperature statistics (PRIMARY TARGET)
    agg_dict['temp'] = ['mean', 'min', 'max', 'std']

    # Core weather variables - daily aggregations
    agg_dict['humidity'] = ['mean', 'min', 'max', 'std']
    agg_dict['sealevelpressure'] = ['mean', 'min', 'max', 'std']
    agg_dict['windspeed'] = ['mean', 'max']
    agg_dict['solarradiation'] = ['sum', 'max']  # Total daily solar energy
    agg_dict['cloudcover'] = ['mean']
    agg_dict['visibility'] = ['mean', 'min']

    # Add lag features if they exist (daily averages of hourly lags)
    lag_cols = [col for col in df.columns if '_lag_' in col]
    for col in lag_cols:
        agg_dict[col] = 'mean'

    # Temporal features (daily aggregations)
    temporal_cols = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 
                    'month_sin', 'month_cos', 'doy_sin', 'doy_cos']
    for col in temporal_cols:
        if col in df.columns:
            agg_dict[col] = 'mean'

    # Categorical indicators (first value of day)
    categorical_cols = ['is_workday', 'is_business_hours', 'hour', 'day_of_week', 
                       'month', 'day_of_year']
    for col in categorical_cols:
        if col in df.columns:
            agg_dict[col] = 'first'

    # Interaction features (daily averages)
    interaction_cols = [col for col in df.columns if any(x in col for x in 
                       ['_ratio', '_spread', '_index', '_factor', '_interaction'])]
    for col in interaction_cols:
        agg_dict[col] = 'mean'

    # Trend features (daily averages)
    trend_cols = [col for col in df.columns if any(x in col for x in 
                 ['_change_', '_tendency_', '_stability', '_acceleration'])]
    for col in trend_cols:
        agg_dict[col] = 'mean'

    # Perform aggregation
    daily_agg = df.groupby('date').agg(agg_dict)

    # Flatten column names
    daily_agg.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                         for col in daily_agg.columns]

    # Reset index to make date a column
    daily_agg.reset_index(inplace=True)
    daily_agg['date'] = pd.to_datetime(daily_agg['date'])

    print(f"   ✅ Aggregated to {len(daily_agg)} daily observations")
    print(f"   📈 Features created: {len(daily_agg.columns)}")

    return daily_agg


def create_5day_targets(df_daily, target_base='temp_mean'):
    """
    Create 5-day ahead targets for forecasting

    Parameters:
    -----------
    df_daily : DataFrame
        Daily features with date column
    target_base : str
        Base temperature column (default: 'temp_mean')

    Returns:
    --------
    df_daily : DataFrame with added target columns
    """

    df = df_daily.copy()

    print(f"\n   🎯 Creating 5-day ahead targets from '{target_base}'...")

    # Create targets for day 1 through day 5
    for day in range(1, 6):
        target_col = f'temp_target_day{day}'
        df[target_col] = df[target_base].shift(-day)
        print(f"      ✅ {target_col} created ({day} day{'s' if day > 1 else ''} ahead)")

    # Remove rows with NaN targets (last 5 days)
    initial_len = len(df)
    df = df.dropna(subset=[f'temp_target_day{d}' for d in range(1, 6)])
    removed = initial_len - len(df)

    print(f"\n   📊 Removed {removed} rows with NaN targets (last 5 days)")
    print(f"   ✅ Final dataset: {len(df)} samples")

    return df


print("📝 Functions defined:")
print("   1. aggregate_hourly_to_daily_features() - Hourly → Daily aggregation")
print("   2. create_5day_targets() - Create 5-day ahead targets")
