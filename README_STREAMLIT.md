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
- **Daily Forecasting**: 5-day ahead temperature predictions
- **Hourly Forecasting**: Multi-horizon predictions (1h, 6h, 24h, 72h, 168h)
- **Multiple Weather Variables**: Temperature, humidity, pressure, wind speed, cloud cover
- **Interactive Visualizations**: Real-time charts with confidence intervals
- **Model Comparison**: Multiple ML models (XGBoost, LightGBM, CatBoost, Ensemble)
- **Historical Data Analysis**: Comprehensive weather data visualization
- **Model Monitoring**: Performance tracking and alerts

### 🛠️ Advanced Features
- **Feature Engineering**: 79+ engineered features for daily, 269+ for hourly
- **Confidence Intervals**: Prediction uncertainty quantification
- **Model Performance Dashboard**: Real-time accuracy metrics
- **Feature Importance Analysis**: Understanding model decisions
- **Prediction History**: Track and compare past forecasts
- **Data Upload**: Custom weather data integration

## 📱 Applications Available

### 1. Daily Weather Forecasting App
- **File**: `app/streamlit_app_daily.py`
- **Purpose**: 5-day temperature forecasting
- **Port**: 8501
- **Features**: Daily temperature trends, seasonal patterns, extended forecasts

### 2. Hourly Weather Forecasting App
- **File**: `app/streamlit_app_hourly.py`
- **Purpose**: Multi-horizon hourly predictions
- **Port**: 8502
- **Features**: Short-term weather changes, multiple variables, high-frequency updates

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
- Optimized for daily temperature trend analysis
- Ideal for medium-term weather planning

#### 🔧 Features
- **Input Parameters**:
  - Current temperature (°C)
  - Humidity (%)
  - Atmospheric pressure (hPa)
  - Wind speed (m/s)
  - Prediction start date

- **Forecasting Options**:
  - 1 to 5 days ahead
  - Confidence intervals (95%)
  - Model comparison (AdaBoost, XGBoost, LightGBM, Random Forest, Gradient Boosting)

- **Visualizations**:
  - Interactive temperature trend charts
  - Historical data analysis
  - Forecast vs. actual comparisons
  - Feature importance analysis

#### 📱 Tabs Available
1. **🔮 Prediction**: Generate forecasts with current conditions
2. **📊 Historical Data**: Analyze past weather patterns
3. **🎯 Model Performance**: Compare model accuracies
4. **🔬 Feature Importance**: Understand model decisions
5. **🚨 Monitoring & Alerts**: Track model health
6. **📜 Prediction History**: Review past forecasts
7. **ℹ️ About**: Project information and documentation

### Hourly Forecasting App (`streamlit_app_hourly.py`)

#### 🎯 Purpose
- Multi-horizon hourly weather predictions
- Covers temperature, humidity, pressure, wind speed, and cloud cover
- Designed for short-term and precise weather forecasting

#### 🔧 Features
- **Forecast Horizons**:
  - 1 Hour: Immediate forecast
  - 6 Hours: Short-term planning
  - 24 Hours: Daily forecast
  - 3 Days (72h): Extended forecast
  - 1 Week (168h): Weekly trend

- **Weather Variables**:
  - Temperature (°C)
  - Humidity (%)
  - Sea Level Pressure (hPa)
  - Wind Speed (m/s)
  - Cloud Cover (%)

- **Model Options**:
  - XGBoost: Gradient boosting with tree-based learning
  - LightGBM: Fast gradient boosting with optimized memory
  - CatBoost: Categorical boosting with automatic feature handling
  - Ensemble: Combination of multiple models

#### 📱 Interface Sections
1. **🌡️ Current Conditions**: Real-time weather display
2. **📈 Forecast Visualization**: Interactive multi-variable charts
3. **🎯 Model Performance**: Accuracy metrics across horizons
4. **🔍 Feature Importance**: Analysis of 269+ engineered features
5. **🚨 Monitoring & Alerts**: Model health and drift detection

## 🔧 Technical Details

### Model Architecture

#### Daily Forecasting Models
- **Features**: 79+ engineered features
- **Target**: Temperature (5-day ahead)
- **Models**: AdaBoost (optimized), XGBoost, LightGBM, Random Forest, Gradient Boosting
- **Validation**: Time series cross-validation
- **Performance**: RMSE ~2.3°C, R² ~0.92

#### Hourly Forecasting Models
- **Features**: 269+ engineered features
- **Targets**: 5 weather variables
- **Models**: XGBoost, LightGBM, CatBoost, Ensemble
- **Horizons**: 1h, 6h, 24h, 72h, 168h
- **Performance**: RMSE 1-7°C depending on horizon

### Feature Engineering

#### Daily Features (79+)
- Temperature lag features (1-5 days)
- Rolling statistics (3, 7, 14 days)
- Seasonal decomposition
- Temporal encoding (cyclical)
- Weather variable interactions
- Trend and anomaly detection

#### Hourly Features (269+)
- Short-term lag features (1-24 hours)
- Rolling statistics (1h, 3h, 6h, 12h, 24h)
- Atmospheric stability indices
- Diurnal cycle encoding
- Pressure tendency calculations
- Wind vector components

### Data Sources
- **Historical Data**: 10+ years of Hanoi weather data
- **Variables**: 25+ meteorological parameters
- **Frequency**: Daily for daily models, hourly for hourly models
- **Quality**: Cleaned and validated weather station data

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