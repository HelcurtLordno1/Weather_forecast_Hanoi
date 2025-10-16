# Excluded Large Files

This repository excludes certain large files to comply with GitHub's 100MB file size limit.

## Excluded Files

### `data/processed/hanoi_weather_hourly_features.csv` (189.52 MB)
- **Size**: 189.52 MB  
- **Records**: 87,696 rows
- **Features**: 269 columns
- **Purpose**: Comprehensive engineered features for hourly weather forecasting

## How to Regenerate Excluded Files

To regenerate the excluded files locally:

1. **Generate hourly features dataset**:
   ```bash
   # Run the feature engineering notebook
   jupyter notebook notebooks_hourly/02_feature_engineering_hourly.ipynb
   ```
   
   Or run the notebook cells that:
   - Load `data/processed/hanoi_weather_hourly_processed.csv`
   - Apply comprehensive feature engineering (temporal, lag, rolling, interaction, trend features)
   - Save to `data/processed/hanoi_weather_hourly_features.csv`

2. **Alternative**: The hourly Streamlit app can still run with the base processed data if the features file is missing, but with reduced functionality.

## File Dependencies

- **Input**: `data/processed/hanoi_weather_hourly_processed.csv`
- **Output**: `data/processed/hanoi_weather_hourly_features.csv`
- **Used by**: 
  - `app/streamlit_app_hourly.py`
  - Hourly forecasting models

## Note

These files are tracked in `.gitignore` to prevent accidental commits of large files.