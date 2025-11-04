# 🌡️ Hanoi Weather Forecasting - Streamlit Applications

A comprehensive set of Streamlit web applications for weather forecasting in Hanoi, Vietnam using advanced machine learning models. This project provides both **daily** and **hourly** weather prediction capabilities with interactive visualizations and model monitoring.

## 📋 Table of Contents

- [Features](#-features)
- [Applications Available](#-applications-available)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Running the Applications](#-running-the-applications)
- [Application Overview](#-application-overview)
- [Technical Details](#-technical-details)
- [Troubleshooting](#-troubleshooting)
- [Project Structure](#-project-structure)

## 🚀 Features

### 🎯 Core Capabilities
- **Daily Forecasting**: 5-day ahead temperature predictions using 130 engineered features
- **Hourly Forecasting**: 5-day ahead temperature predictions using 91 features aggregated from hourly data
- **Single Target Variable**: Temperature (°C) - focused and accurate predictions
- **Interactive Visualizations**: Real-time charts with confidence intervals
- **Model Comparison**: Multiple ML models (Random Forest, XGBoost, LightGBM, AdaBoost, Gradient Boosting)
- **Historical Data Analysis**: Comprehensive weather data visualization
- **Model Monitoring**: Performance tracking with degradation analysis

### 🛠️ Advanced Features
- **Daily Features**: 150-180 engineered features with cyclical encoding and leak-free methodology
- **Hourly Features**: 91 aggregated features capturing diurnal patterns
- **Confidence Intervals**: Prediction uncertainty quantification (±2 RMSE)
- **Model Performance Dashboard**: Real-time accuracy metrics per forecast day
- **Feature Importance Analysis**: Understanding model decisions
- **Prediction History**: Track and compare past forecasts
- **Degradation Tracking**: Monitor performance decline across forecast horizons (12.7% threshold)

## 📱 Applications Available

### 1. Daily Weather Forecasting App
- **File**: `app/streamlit_app_daily.py`
- **Purpose**: 5-day temperature forecasting using daily data
- **Port**: 8501
- **Features**: 150-180 engineered features, multi-horizon forecasting, 12.7% degradation (Day 1→Day 5)
- **Best Model**: LightGBM (Day 1 R² 0.9113, RMSE 1.423°C)

### 2. Hourly Weather Forecasting App
- **File**: `app/streamlit_app_hourly.py`
- **Purpose**: 5-day temperature forecasting using hourly data aggregated to daily
- **Port**: 8502
- **Features**: 91 aggregated features from hourly observations, diurnal pattern capture
- **Best Model**: XGBoost (Day 1 R² 0.9362, RMSE 1.283°C)
- **Advantage**: Superior Day 1-2 accuracy due to hourly data resolution

### 3. Main Combined App
- **File**: `app_streamlit.py`
- **Purpose**: Unified interface for both daily and hourly forecasting
- **Port**: 8501
- **Features**: Combined dashboard with all capabilities

## 🚀 Quick Start

### Option 1: Using Launcher Scripts (Recommended)

#### For Daily Forecasting:
```bash
# Navigate to the app directory
cd app

# Run the daily forecasting app
python run_daily_app.py
```

#### For Hourly Forecasting:
```bash
# Navigate to the app directory  
cd app

# Run the hourly forecasting app
python run_hourly_app.py
```

### Option 2: Direct Streamlit Commands

#### Daily App:
```bash
streamlit run app/streamlit_app_daily.py --server.port=8501
```

#### Hourly App:
```bash
streamlit run app/streamlit_app_hourly.py --server.port=8502
```

#### Main Combined App:
```bash
streamlit run app_streamlit.py --server.port=8501
```

## 💻 Installation

### Prerequisites
- Python 3.8 or higher
- Git (for cloning the repository)

### 1. Clone the Repository
```bash
git clone https://github.com/HelcurtLordno1/Weather_forecast_Hanoi.git
cd Weather_forecast_Hanoi
```

### 2. Install Dependencies

#### Option A: Using requirements.txt (Recommended)
```bash
pip install -r requirements_streamlit.txt
```

#### Option B: Manual Installation
```bash
pip install streamlit pandas numpy plotly scikit-learn xgboost lightgbm joblib python-dateutil pytz matplotlib seaborn optuna
```

### 3. Verify Installation
```bash
streamlit --version
python -c "import pandas, numpy, plotly, sklearn; print('✅ All dependencies installed successfully')"
```

## 🏃‍♂️ Running the Applications

### Method 1: Using Launcher Scripts

The launcher scripts automatically handle dependencies and configuration:

```bash
# For Daily Forecasting
cd app
python run_daily_app.py

# For Hourly Forecasting  
cd app
python run_hourly_app.py
```

### Method 2: Manual Streamlit Commands

```bash
# Daily App (5-day temperature forecasting)
streamlit run app/streamlit_app_daily.py --server.port=8501 --server.address=localhost

# Hourly App (multi-horizon weather forecasting)
streamlit run app/streamlit_app_hourly.py --server.port=8502 --server.address=localhost

# Main Combined App
streamlit run app_streamlit.py --server.port=8501 --server.address=localhost
```

### Method 3: Using the Main Run Script

```bash
python run_app.py
```

## 📊 Application Overview

### Daily Forecasting App (`streamlit_app_daily.py`)

#### 🎯 Purpose
- Provides 5-day ahead temperature forecasting for Hanoi
- Uses 130 engineered features from daily weather data
- Optimized for medium-term weather planning

#### 🔧 Features
- **Input Parameters**:
  - Current temperature (°C)
  - Humidity (%)
  - Atmospheric pressure (hPa)
  - Wind speed (m/s)
  - Prediction start date

- **Forecasting Options**:
  - 1 to 5 days ahead temperature prediction
  - Confidence intervals (±2 RMSE ~95%)
  - Model comparison (Random Forest, XGBoost, LightGBM, AdaBoost, Gradient Boosting)

- **Model Performance** (from actual metadata):
  - **Production Model: LightGBM** (selected via Day 1 prioritized strategy):
    - Day 1: R² 0.9113, RMSE 1.423°C, MAE 1.121°C, MAPE 4.59%
    - Day 2: R² 0.8316, RMSE 1.958°C, MAE 1.548°C, MAPE 6.47%
    - Day 3: R² 0.8037, RMSE 2.113°C, MAE 1.695°C, MAPE 7.12%
    - Day 4: R² 0.8047, RMSE 2.106°C, MAE 1.707°C, MAPE 7.17%
    - Day 5: R² 0.7951, RMSE 2.155°C, MAE 1.728°C, MAPE 7.32%
  - **R² Degradation**: 12.7% (Day 1 → Day 5)

- **Visualizations**:
  - Interactive temperature trend charts with historical context
  - Confidence interval bands
  - Day-by-day forecast summary cards
  - Performance degradation analysis

#### 📱 Tabs Available
1. **🔮 Prediction**: Generate forecasts with current conditions
2. **📊 Historical Data**: Analyze past weather patterns (10+ years)
3. **🎯 Model Performance**: Compare all 5 models across all 5 days (25 combinations)
4. **🔬 Feature Importance**: Understand the 130 features used
5. **🚨 Monitoring & Alerts**: Track model health and degradation
6. **📜 Prediction History**: Review past forecasts
7. **ℹ️ About**: Project information and documentation

### Hourly Forecasting App (`streamlit_app_hourly.py`)

#### 🎯 Purpose
- **5-day ahead temperature forecasting** using hourly data aggregated to daily level
- Leverages diurnal (24-hour) patterns for improved short-term accuracy
- Designed for **1-3 day operational planning** where high accuracy is critical

#### 🔧 Features
- **Forecast Horizons**: Day 1 through Day 5 (same as daily app but with hourly data)
- **Target Variable**: Daily average temperature (°C) - single focused prediction
- **Data Source**: Hourly weather observations aggregated to daily statistics

- **Model Performance** (from actual metadata):
  - **XGBoost Models** (separate model per day):
    - Day 1: R² 0.9362, RMSE 1.283°C, MAE 0.994°C ⭐ **Best Day 1 Accuracy**
    - Day 2: R² 0.8504, RMSE 1.965°C, MAE 1.537°C
    - Day 3: R² 0.8047, RMSE 2.245°C, MAE 1.794°C
    - Day 4: R² 0.7759, RMSE 2.405°C, MAE 1.920°C
    - Day 5: R² 0.7661, RMSE 2.458°C, MAE 1.940°C
  - **Degradation**: 18.2% (Day 1 → Day 5) - slightly higher than daily model but superior Day 1-2

- **Advantages Over Daily-Only Model**:
  - 🏆 Day 1: ~6% better RMSE (1.283 vs 1.417°C)
  - 🏆 Day 2: ~0.5% better RMSE (1.965 vs 1.954°C)
  - 📊 Captures diurnal temperature cycles
  - 📊 Better detects rapid weather changes
  - 📊 Higher temporal resolution from hourly observations

#### 📱 Interface Sections
1. **🔮 Prediction**: Input current weather → Generate 5-day forecast
   - Uses 91 aggregated features from hourly data
   - Displays prediction with model R² and RMSE per day
   - Color-coded forecast cards (green = excellent R², orange = good, red = fair)

2. **🎯 Model Performance**: Three sub-tabs:
   - **Performance Table**: All 4 models × 5 days = 20 combinations
   - **Degradation Analysis**: R² decline visualization Day 1→5
   - **Model Comparison**: XGBoost, LightGBM, RandomForest, GradientBoosting rankings

3. **� Feature Importance**: Analysis of 91 aggregated features
   - Temperature statistics (mean, min, max, std)
   - Humidity, pressure, wind aggregations
   - Temporal features (hour, day, month encodings)
   - Lag features from previous days
   - Rolling statistics and trends

4. **ℹ️ About**: Technical details and when to use hourly vs daily model

## 🔧 Technical Details

### Model Architecture

#### Daily Forecasting Models
- **Features**: 150-180 engineered features with leak-free methodology
  - Temporal features with cyclical encoding (month_sin/cos, dayofweek_sin/cos, dayofyear_sin/cos)
  - 3 season indicators (winter, spring, summer - autumn implied)
  - Lag features (temperature, humidity, pressure, wind, precipitation lags 1-30 days)
  - Rolling features (3d, 7d, 14d, 30d rolling statistics) applied post-split
  - Removed: day, week, is_autumn to prevent data leakage
- **Target**: Temperature (5-day ahead) - separate target per day
- **Models**: LightGBM (production), XGBoost, Random Forest, AdaBoost, Gradient Boosting
- **Training**: Multi-horizon approach with separate models per forecast day
- **Validation**: Time series split (70% train, 15% val, 15% test)
- **Performance**: 
  - Production Model: LightGBM (Day 1 R² 0.9113, RMSE 1.423°C)
  - Degradation: 12.7% (Day 1 R² 0.9113 → Day 5 R² 0.7951)

#### Hourly Forecasting Models
- **Features**: 91 aggregated features from hourly data
  - Base weather statistics aggregated to daily (mean, min, max, std)
  - Temporal features from hourly observations
  - Lag features from previous days
  - Rolling statistics capturing diurnal patterns
  - Interaction and trend features
- **Target**: Daily average temperature (Days 1-5) - from hourly aggregation
- **Models**: XGBoost (best), LightGBM, RandomForest, GradientBoosting
- **Training**: 5 separate XGBoost models (one per forecast day)
- **Validation**: Temporal split (70% train, 10% val, 20% test)
- **Performance**: 
  - Day 1: R² 0.9362, RMSE 1.283°C ⭐ **Superior to daily model**
  - Day 5: R² 0.7661, RMSE 2.458°C
  - Degradation: 18.2% (higher than daily but better short-term)

### Feature Engineering

#### Daily Features (150-180 total)
**1. Temporal Features**:
- Cyclical encoding: month_sin/cos, dayofyear_sin/cos, dayofweek_sin/cos
- Seasonal indicators: is_winter, is_spring, is_summer (autumn implied)
- Temporal: year, month, dayofweek, dayofyear, quarter
- Trend: days_since_start, years_since_start
- **Removed**: day, week, is_autumn (to prevent data leakage)

**2. Lag Features**:
- Temperature lags: 1, 2, 3, 4, 5, 6, 7, 14, 30 days
- Tempmax/tempmin lags: 1, 2, 3, 7 days
- Weather variable lags: humidity, precip, windspeed, pressure (1, 2, 3, 7 days)
- Temperature differences: 1d, 2d, 7d

**3. Rolling Features (Leak-Free)**:
- Applied post-split to prevent data leakage
- Temperature rolling (3d, 7d, 14d, 30d): mean, std, min, max, range, position_in_range
- Humidity rolling (3d, 7d, 14d): mean, std
- Precipitation rolling (3d, 7d, 14d): mean, std
- Wind speed rolling (3d, 7d, 14d): mean, std
- Solar radiation rolling (3d, 7d, 14d): mean, std
- Temperature EMA (exponential moving average): 3d, 7d, 30d

**4. Additional Features**:
- Weather conditions and description analysis
- Interaction features and derived metrics
- Statistical aggregations and trend indicators

#### Hourly Features (91 total)
**Aggregated from Hourly to Daily Statistics**:
- **Base Weather** (aggregated): temp, humidity, pressure, windspeed (mean, min, max, std per day)
- **Temporal**: Hour sin/cos, day-of-week sin/cos, month sin/cos (daily averages)
- **Lag Features**: Daily aggregations of hourly lags (1h, 3h, 6h, 24h, 48h, 168h)
- **Rolling Statistics**: Daily averages of rolling hourly statistics
- **Interactions**: Daily averages of temp-humidity ratios, wind chill, pressure tendencies
- **Diurnal Patterns**: Morning, afternoon, evening, night temperature ranges
- **Trend Features**: Daily temperature amplitude, variability indices, stability metrics

### Data Sources
- **Historical Data**: 10+ years of Hanoi weather data (2015-2025)
- **Daily Dataset**: 3,649 daily records with 33 weather parameters
- **Hourly Dataset**: ~87,698 hourly records with 28 weather parameters (aggregated to daily)
- **Variables**: Temperature, humidity, pressure, wind speed, precipitation, solar radiation, cloud cover, weather conditions, and more
- **Quality**: Cleaned and validated weather station data from Hanoi, Vietnam

---

## 🎯 Model Selection Guide

### When to Use Daily Forecasting App:
✅ **Best for comprehensive 5-day forecasting** with production-grade accuracy
- Strategic planning and extended forecasts
- LightGBM model optimized for Day 1 accuracy (R² 0.9113)
- 150-180 leak-free engineered features
- Controlled degradation monitoring (12.7% threshold)
- Long-term trend analysis with 10+ years of daily data context

### When to Use Hourly Forecasting App:
✅ **Best for Days 1-2** (superior short-term accuracy)
- Operational planning and immediate decisions
- When highest Day 1-2 accuracy is critical (R² 0.936 vs 0.908)
- Capturing rapid weather changes
- Leveraging diurnal (24-hour) temperature patterns
- Real-time operations requiring precise near-term forecasts

### Performance Comparison:

| Forecast Day | Daily Model (LightGBM) | Hourly Model (XGBoost) | Winner |
|--------------|------------------------|------------------------|--------|
| **Day 1** | R² 0.911, RMSE 1.42°C | R² 0.936, RMSE 1.28°C | 🏆 **Hourly** |
| **Day 2** | R² 0.832, RMSE 1.96°C | R² 0.850, RMSE 1.96°C | ⚖️ **Similar** |
| **Day 3** | R² 0.804, RMSE 2.11°C | R² 0.805, RMSE 2.24°C | 🏆 **Daily** |
| **Day 4** | R² 0.805, RMSE 2.11°C | R² 0.776, RMSE 2.41°C | 🏆 **Daily** |
| **Day 5** | R² 0.795, RMSE 2.16°C | R² 0.766, RMSE 2.46°C | 🏆 **Daily** |

**Key Insights**:
- Hourly model: ~10% better Day 1 RMSE, captures short-term dynamics
- Daily model (LightGBM): Superior Days 3-5, controlled 12.7% degradation
- Daily model: 150-180 leak-free features with optimized Day 1 performance
- **Best Practice**: Use hourly for Day 1, daily (LightGBM) for Days 2-5

---

## 📊 Data Sources

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors
```bash
# Problem: Module not found errors
# Solution: Install missing dependencies
pip install -r requirements_streamlit.txt

# For specific missing modules:
pip install streamlit plotly pandas numpy scikit-learn
```

#### 2. Port Already in Use
```bash
# Problem: Port 8501/8502 already in use
# Solution: Use different port
streamlit run app/streamlit_app_daily.py --server.port=8503

# Or kill existing process
lsof -ti:8501 | xargs kill -9  # macOS/Linux
netstat -ano | findstr :8501   # Windows (find PID and kill)
```

#### 3. Model Files Not Found
```bash
# Problem: Trained models not available
# Solution: Run model training notebooks first
# 1. Navigate to notebooks/ directory
# 2. Run 03_model_training_comprehensive.ipynb (for daily)
# 3. Run notebooks_hourly/03_model_training_hourly.ipynb (for hourly)
```

#### 4. Data Files Missing
```bash
# Problem: Raw data files not found
# Solution: Ensure data files are in correct directories
# Daily: data/raw/Hanoi-Daily-10-years.csv
# Hourly: data/raw/hanoi_weather_data_hourly.csv
```

#### 5. Memory Issues
```bash
# Problem: App runs slowly or crashes
# Solution: Increase available memory or reduce data size
# Option 1: Restart the app
# Option 2: Clear Streamlit cache
streamlit cache clear

# Option 3: Reduce forecast horizon or variables
```

### Performance Optimization

#### For Better Performance:
1. **Close unused tabs** in the Streamlit app
2. **Limit forecast horizons** to necessary timeframes
3. **Reduce number of target variables** in hourly app
4. **Clear browser cache** if app loads slowly
5. **Restart the app** periodically for memory cleanup

#### System Requirements:
- **RAM**: Minimum 4GB, Recommended 8GB+
- **CPU**: Multi-core processor recommended
- **Storage**: 2GB free space for models and data
- **Browser**: Chrome, Firefox, Safari, or Edge (latest versions)

### Debug Mode

To run in debug mode with more detailed error messages:

```bash
# Enable debug logging
export STREAMLIT_LOG_LEVEL=debug

# Run with verbose output
streamlit run app/streamlit_app_daily.py --logger.level=debug
```

## 📁 Project Structure

```
weather_forcast_project/
├── 📱 STREAMLIT APPLICATIONS
│   ├── app_streamlit.py              # Main combined app
│   ├── app/
│   │   ├── streamlit_app_daily.py    # Daily forecasting app
│   │   ├── streamlit_app_hourly.py   # Hourly forecasting app
│   │   ├── run_daily_app.py          # Daily app launcher
│   │   └── run_hourly_app.py         # Hourly app launcher
│   ├── run_app.py                    # Main app launcher
│   └── requirements_streamlit.txt    # Streamlit dependencies
│
├── 🧠 MODELS & DATA
│   ├── models/
│   │   ├── daily_trained/            # Daily forecasting models
│   │   └── hourly_trained/           # Hourly forecasting models (if available)
│   ├── data/
│   │   ├── raw/                      # Original weather data
│   │   └── processed/                # Cleaned and feature-engineered data
│   └── src/                          # Source code modules
│
├── 📓 NOTEBOOKS
│   ├── notebooks/                    # Daily forecasting notebooks
│   └── notebooks_hourly/             # Hourly forecasting notebooks
│
└── 📋 DOCUMENTATION
    ├── README.md                     # Main project README
    ├── README_STREAMLIT.md           # This file - Streamlit guide
    ├── README_Daily.md               # Daily forecasting documentation
    └── README_HOURLY.md              # Hourly forecasting documentation
```

## 🔗 Related Documentation

- **[Main Project README](README.md)**: Overall project overview
- **[Daily Forecasting Guide](README_Daily.md)**: Daily model details
- **[Hourly Forecasting Guide](README_HOURLY.md)**: Hourly model details
- **[Excluded Files Guide](README_EXCLUDED_FILES.md)**: Development notes

## 🎯 Usage Examples

### Example 1: Quick Daily Forecast
1. Run: `python app/run_daily_app.py`
2. Open: http://localhost:8501
3. Navigate to "🔮 Prediction" tab
4. Enter current weather conditions
5. Click "🔮 Generate 5-Day Forecast"

### Example 2: Hourly Weather Analysis
1. Run: `python app/run_hourly_app.py`
2. Open: http://localhost:8502
3. Select forecast horizon (e.g., "24 Hours")
4. Choose variables (Temperature, Humidity, Pressure)
5. Set start date/time
6. View interactive forecasts

### Example 3: Model Comparison
1. Use either daily or hourly app
2. Navigate to "🎯 Model Performance" tab
3. Compare different models (XGBoost, LightGBM, etc.)
4. Analyze accuracy metrics and performance trends

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test the Streamlit applications
5. Submit a pull request

## 📞 Support

For issues or questions:
- Check the troubleshooting section above
- Review the application logs in the terminal
- Ensure all dependencies are properly installed
- Verify data and model files are in correct locations

## 📄 License

This project is developed as part of a Machine Learning course project. Please refer to the repository for licensing information.

---

**🌡️ Hanoi Weather Forecasting Project** | Advanced ML-powered weather predictions for Hanoi, Vietnam