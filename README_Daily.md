# 🌡️ Hanoi Daily Temperature Forecasting - Core Implementation

## 📊 Project Overview

This is the **core implementation** of the Hanoi Weather Forecasting project, focusing on **daily temperature prediction** using advanced machine learning techniques. This module processes over **10 years of daily weather observations** to forecast temperature up to **5 days ahead** with high precision and reliability.

### 🎯 Key Objectives

- **Strategic Weather Planning**: Process daily data for long-term weather planning and analysis
- **Seasonal Pattern Analysis**: Capture year-round temperature trends and seasonal cycles
- **Multi-day Forecasting**: Predict temperature for multiple future days with confidence intervals
- **Advanced Feature Engineering**: Leverage daily-specific patterns and historical trends

---

## 📂 Directory Structure

```
weather_forecast_project/
├── notebooks/                          # Daily analysis notebooks
│   ├── 00_data_exploration_comprehensive.ipynb # Daily data exploration
│   ├── 01_data_processing_comprehensive.ipynb  # Daily data cleaning
│   ├── 02_feature_engineering_comprehensive.ipynb # Daily feature engineering
│   ├── 03_model_training_comprehensive.ipynb   # Daily model training
│   └── 04_model_monitoring_retraining.ipynb    # Daily model monitoring
│
├── src/daily/                          # Daily-specific utilities
│   ├── data_utils_daily.py            # Daily data processing
│   ├── feature_utils_daily.py         # Daily feature engineering
│   ├── model_utils_daily.py           # Daily model training
│   └── visualization_daily.py         # Daily visualizations
│
├── models/daily_trained/               # Daily model artifacts
│   ├── best_model_adaboost_optimized.joblib # Best daily model
│   ├── feature_columns.joblib         # Daily feature list
│   └── model_metadata.json            # Daily model metadata
│
├── app/                               # Streamlit applications
│   ├── streamlit_app_daily.py        # Daily forecasting app
│   └── run_daily_app.py              # Daily app launcher
│
└── data/raw/
    └── Hanoi-Daily-10-years.csv      # Daily weather dataset
```

---

## 📈 Dataset Information

### **Daily Weather Dataset**
- **File**: `Hanoi-Daily-10-years.csv`
- **Records**: ~3,660 daily observations
- **Time Period**: 2013-2024 (10+ years)
- **Features**: 28 weather parameters per day
- **Frequency**: Daily observations (365 records per year)

### **Key Features**
```python
# Core weather parameters
- temp, feelslike, dew          # Temperature metrics
- humidity, precip, precipprob  # Moisture metrics
- windspeed, winddir, windgust  # Wind metrics
- sealevelpressure             # Pressure
- cloudcover, visibility       # Visibility metrics
- solarradiation, solarenergy  # Solar metrics
- conditions, icon             # Weather conditions

# Daily-specific advantages
- Long-term seasonal trends
- Monthly and yearly patterns
- Climate change indicators
- Seasonal transition timing
```

---

## 🔬 Step-by-Step Analysis Workflow

### **Step 1: Daily Data Exploration** 
📓 **Notebook**: `00_data_exploration_comprehensive.ipynb`

**Objectives**:
- Analyze 3,660+ daily records for long-term patterns and trends
- Explore seasonal temperature cycles and yearly variations
- Identify correlations between weather variables and temperature
- Assess data quality, completeness, and temporal consistency

**Key Analyses**:
```python
# Seasonal pattern analysis
monthly_avg_temp = df.groupby('month')['temp'].mean()
yearly_trends = df.groupby('year')['temp'].mean()

# Climate change detection
temp_trend_analysis = df.set_index('date')['temp'].resample('Y').mean()
long_term_slope = calculate_trend_slope(temp_trend_analysis)

# Extreme weather identification
heat_waves = df[df['temp'] > df['temp'].quantile(0.95)]
cold_snaps = df[df['temp'] < df['temp'].quantile(0.05)]
```

### **Step 2: Daily Data Processing**
📓 **Notebook**: `01_data_processing_comprehensive.ipynb`

**Enhanced Processing**:
- **Temporal Gap Detection**: Identify missing days in sequence and assess impact
- **Daily Interpolation**: Season-aware missing value handling with climate context
- **Outlier Detection**: Daily-specific outlier identification using seasonal baselines
- **Quality Assessment**: Daily data completeness metrics and validation

**Unique Challenges**:
```python
# Handle seasonal variations in data quality
# Detect weather station maintenance periods  
# Account for leap years and calendar effects
# Process irregular reporting patterns during extreme weather
```

### **Step 3: Advanced Daily Feature Engineering**
📓 **Notebook**: `02_feature_engineering_comprehensive.ipynb`

**Daily-Specific Features** (~79 features):

#### **Temporal Features**
```python
# Cyclical encoding for seasonality
month_sin = sin(2π × month / 12)
month_cos = cos(2π × month / 12)
day_of_year_sin = sin(2π × day_of_year / 365)
day_of_year_cos = cos(2π × day_of_year / 365)

# Calendar-based features
day_of_year, week_of_year, season, quarter
is_weekend, is_holiday, is_summer, is_winter
```

#### **Lag Features**
```python
# Multi-day lags (crucial for daily forecasting)
temp_lag_1d, temp_lag_2d, temp_lag_3d
temp_lag_7d, temp_lag_14d, temp_lag_30d

# Seasonal lags (year-over-year patterns)
temp_lag_365d, temp_lag_730d  # Previous years same date
```

#### **Rolling Statistics**
```python
# Multiple time windows for trend analysis
temp_rolling_mean_7d, temp_rolling_mean_14d
temp_rolling_mean_30d, temp_rolling_mean_90d
temp_rolling_std_7d, temp_rolling_range_30d
temp_rolling_min_7d, temp_rolling_max_7d
```

#### **Change Features**
```python
# Daily change rates and trend indicators
temp_change_1d, temp_change_7d, temp_change_30d
temp_trend_7d = temp_rolling_mean_7d.diff()
temp_acceleration_7d = temp_trend_7d.diff()

# Weather transition indicators
pressure_change_3d, humidity_change_7d
wind_speed_change_3d, cloud_cover_change_7d
```

#### **Seasonal Pattern Features**
```python
# Climate context and anomaly detection
temp_from_seasonal_mean = temp - seasonal_temp_mean
temp_normalized_in_season = (temp - seasonal_mean) / seasonal_std
seasonal_temp_anomaly, position_in_season
temp_percentile_in_month, temp_rank_in_year
```

#### **Interaction Features**
```python
# Season-specific interactions
temp_season_interaction = temp × season_encoded
humidity_temp_summer = humidity × temp × is_summer
wind_chill_winter = temp - 0.7 × windspeed × is_winter
pressure_temp_correlation = pressure × temp_normalized
```

### **Step 4: Daily Model Training**
📓 **Notebook**: `03_model_training_comprehensive.ipynb`

**Multi-Model Approach**:
```python
# Different models optimized for daily forecasting
models = {
    'adaboost': AdaBoostRegressor(),        # Best performer for daily
    'xgboost': XGBRegressor(),             # Gradient boosting power
    'lightgbm': LGBMRegressor(),           # Fast training with efficiency
    'random_forest': RandomForestRegressor(), # Robust ensemble method
    'gradient_boost': GradientBoostingRegressor() # Traditional boosting
}

# Hyperparameter optimization for each model
optimized_models = {}
for name, model in models.items():
    optimized_models[name] = optimize_hyperparameters(model, X_train, y_train)
```

**Daily-Specific Validation**:
```python
# Time series validation with daily awareness
# Ensure no future data leakage in temporal features
# Account for seasonal patterns in train/test splits
# Year-based stratification for robust validation
tscv = TimeSeriesSplit(n_splits=5, test_size=365)  # 1 year test sets
```

### **Step 5: Daily Model Monitoring**
📓 **Notebook**: `04_model_monitoring_retraining.ipynb`

**Daily Performance Metrics**:
- **Seasonal Accuracy**: Performance analysis by season and month
- **Year-over-year Patterns**: Annual performance consistency tracking
- **Forecast Horizon Degradation**: Accuracy vs. prediction distance analysis
- **Climate Trend Adaptation**: Model stability over changing climate patterns

---

## 🎯 Performance Results with Daily Data

### **Accuracy Achievements**

| Metric | Best Model (AdaBoost) | XGBoost | LightGBM | Random Forest | Gradient Boost |
|--------|----------------------|---------|----------|---------------|----------------|
| **1-day RMSE** | **2.17°C** | 2.34°C | 2.41°C | 2.48°C | 2.52°C |
| **3-day RMSE** | **2.45°C** | 2.58°C | 2.63°C | 2.71°C | 2.78°C |
| **5-day RMSE** | **2.67°C** | 2.82°C | 2.89°C | 2.95°C | 3.02°C |
| **R² Score** | **0.923** | 0.908 | 0.902 | 0.896 | 0.889 |
| **MAE** | **1.87°C** | 1.95°C | 2.01°C | 2.08°C | 2.12°C |

### **Model Capabilities**

1. **Multi-day Forecasting**: Reliable predictions up to 5 days ahead
2. **Seasonal Adaptation**: Captures complex seasonal temperature patterns
3. **Climate Monitoring**: Tracks long-term temperature trends and changes
4. **Strategic Planning**: Supports medium-term weather-dependent decisions

### **Business Applications**

- **Agricultural Planning**: Crop scheduling, planting, and harvest timing optimization
- **Energy Management**: Strategic HVAC planning and grid load forecasting
- **Tourism Industry**: Event planning, seasonal preparation, and activity scheduling
- **Climate Research**: Temperature trend analysis and climate change monitoring

---

## 🚀 Getting Started with Daily Analysis

### **Prerequisites**
```bash
# Install core dependencies for daily analysis
pip install scikit-learn>=1.3.0 xgboost>=1.7.0 lightgbm>=4.0.0 
pip install optuna>=3.4.0 streamlit>=1.28.0 plotly>=5.17.0
pip install pandas>=1.5.0 numpy>=1.24.0 joblib>=1.3.0
```

### **Quick Start**
```bash
# 1. Explore daily data patterns and trends
jupyter notebook notebooks/00_data_exploration_comprehensive.ipynb

# 2. Engineer daily-specific features  
jupyter notebook notebooks/02_feature_engineering_comprehensive.ipynb

# 3. Train and optimize daily models
jupyter notebook notebooks/03_model_training_comprehensive.ipynb

# 4. Launch interactive daily forecasting app
python app/run_daily_app.py
# Or directly: streamlit run app/streamlit_app_daily.py
```

### **Data Loading Example**
```python
from src.daily.data_utils_daily import load_raw_daily_data
from src.daily.feature_utils_daily import DailyFeatureEngineering

# Load daily weather data
df_daily = load_raw_daily_data()
print(f"✅ Loaded {len(df_daily)} daily records from {df_daily['datetime'].min()} to {df_daily['datetime'].max()}")

# Create comprehensive daily features
feature_engineer = DailyFeatureEngineering()
df_features = feature_engineer.create_all_features(df_daily)
print(f"🔧 Engineered {len(df_features.columns)} features for daily forecasting")

# Display feature summary
feature_engineer.display_feature_summary()
```

---

## 📊 Streamlit Web Application

### **Daily Forecasting Application Interface**

The daily forecasting web application provides a comprehensive interface with multiple interactive sections optimized for daily temperature prediction:

### 🔮 **Prediction Tab**
- **Input Interface**: Enter current weather conditions (temperature, humidity, pressure, wind speed)
- **Forecast Configuration**: Select forecast horizon (1-5 days ahead) and confidence intervals
- **Instant Predictions**: Generate real-time temperature predictions with uncertainty quantification
- **Interactive Visualizations**: Advanced Plotly charts showing multi-day forecast trends
- **Model Comparison**: Switch between different models (AdaBoost, XGBoost, LightGBM, etc.)

### 📈 **Model Performance Tab** 
- **Multi-Model Comparison**: Compare all 5 trained models with detailed metrics
- **Performance Metrics**: View RMSE, MAE, R², MAPE across different forecast horizons
- **Training Analysis**: Training vs validation performance with overfitting detection
- **Seasonal Performance**: Model accuracy by season and month
- **Hyperparameter Insights**: Optimized parameters and model configurations

### 🔬 **Feature Importance Tab**
- **Feature Ranking**: Explore the most influential features for daily predictions
- **Interactive Plots**: Feature importance visualization with filtering capabilities
- **Correlation Analysis**: Heatmaps showing feature relationships and dependencies
- **Category Breakdown**: Features grouped by type (temporal, lag, rolling, interaction)
- **SHAP Analysis**: Model interpretability with SHAP value explanations

### 🚨 **Monitoring & Alerts Tab**
- **Model Health Dashboard**: Real-time model performance monitoring
- **Drift Detection**: Data drift alerts and distribution changes
- **Performance Alerts**: Accuracy degradation warnings and thresholds
- **Retraining Schedule**: Automated retraining recommendations and triggers
- **System Status**: Model availability and prediction service health

### 📜 **Prediction History Tab**
- **Historical Forecasts**: Review and analyze past prediction performance
- **Accuracy Tracking**: Compare predictions with actual outcomes
- **Trend Analysis**: Long-term model performance trends
- **Export Capabilities**: Download prediction history and performance reports

### ℹ️ **About Tab**
- **Technical Specifications**: Detailed model architecture and methodology
- **Feature Engineering**: Comprehensive explanation of 79 engineered features
- **Performance Benchmarks**: Accuracy metrics and comparison with baselines
- **Use Cases**: Real-world applications and daily forecasting benefits

---

## 🛠️ Technical Implementation Details

### **Feature Engineering Pipeline**
```python
class DailyFeatureEngineering:
    """Comprehensive daily feature engineering for temperature forecasting."""
    
    def __init__(self):
        self.feature_groups = {
            'temporal': ['month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos'],
            'lag': ['temp_lag_1d', 'temp_lag_2d', 'temp_lag_3d', 'temp_lag_7d'],
            'rolling': ['temp_rolling_7d', 'temp_rolling_14d', 'temp_rolling_30d'],
            'change': ['temp_change_1d', 'temp_change_7d', 'temp_trend_7d'],
            'seasonal': ['temp_from_seasonal_mean', 'temp_percentile_in_month'],
            'interaction': ['humidity_temp_summer', 'wind_chill_winter']
        }
    
    def create_all_features(self, df):
        """Create all daily-specific features."""
        df = self.create_temporal_features(df)      # Time-based features
        df = self.create_lag_features(df)           # Historical lag features  
        df = self.create_rolling_features(df)       # Statistical rolling features
        df = self.create_change_features(df)        # Change and trend indicators
        df = self.create_seasonal_features(df)      # Seasonal pattern features
        df = self.create_interaction_features(df)   # Feature interactions
        return df
    
    def get_feature_importance_summary(self):
        """Return feature importance analysis for interpretability."""
        return self.feature_importance_analysis
```

### **Multi-Model Training Framework**
```python
# Comprehensive model training with hyperparameter optimization
daily_models = {}
model_configs = {
    'adaboost': {
        'model': AdaBoostRegressor(),
        'param_space': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1.0],
            'loss': ['linear', 'square', 'exponential']
        }
    },
    'xgboost': {
        'model': XGBRegressor(),
        'param_space': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    }
}

for model_name, config in model_configs.items():
    # Optimize hyperparameters using Optuna
    optimized_model = optimize_hyperparameters(
        config['model'], 
        config['param_space'], 
        X_train, y_train,
        cv_strategy='time_series'
    )
    daily_models[model_name] = optimized_model
```

### **Daily Validation Strategy**
```python
# Time-aware cross-validation for daily forecasting
from sklearn.model_selection import TimeSeriesSplit

def daily_time_series_validation(X, y, n_splits=5):
    """Custom validation strategy for daily temperature forecasting."""
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=365)  # 1 year test sets
    
    validation_scores = []
    for train_idx, test_idx in tscv.split(X):
        # Ensure no data leakage across temporal boundaries
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train model and evaluate
        model = train_model(X_train, y_train)
        predictions = model.predict(X_test)
        score = calculate_metrics(y_test, predictions)
        validation_scores.append(score)
    
    return validation_scores
```

---

## 📈 Success Metrics & KPIs

### **Primary Performance Metrics**
- **RMSE by Season**: Model accuracy assessment for spring, summer, autumn, winter
- **MAE by Month**: Monthly performance evaluation for seasonal variation detection
- **R² by Forecast Horizon**: Predictive power analysis vs. prediction distance (1-5 days)
- **Directional Accuracy**: Temperature trend prediction success rate and reliability

### **Secondary Quality Metrics**
- **Seasonal Transition Accuracy**: Performance during season changes and weather pattern shifts
- **Extreme Weather Detection**: Heat wave and cold snap prediction capability
- **Long-term Trend Tracking**: Climate change indicator accuracy and drift detection
- **Agricultural Timing**: Support for crop management and farming decision accuracy

### **Operational Metrics**
- **Prediction Latency**: Time required to generate forecasts (target: <100ms)
- **Model Availability**: System uptime and prediction service reliability (target: 99.9%)
- **Data Freshness**: Time lag between data acquisition and model updates
- **Retraining Frequency**: Model update schedule and performance maintenance

---

## 🎉 Expected Outcomes

By completing the daily temperature forecasting implementation, you will achieve:

1. ✅ **Robust Forecasting System**: Production-ready 5-day temperature prediction capability
2. ✅ **Advanced Feature Engineering**: 79+ sophisticated daily-specific features with domain expertise
3. ✅ **Multi-Model Framework**: 5 optimized models for different use cases and performance requirements
4. ✅ **Comprehensive Analysis**: Deep understanding of seasonal patterns and long-term climate trends
5. ✅ **Production-Ready Application**: Professional Streamlit web app for daily forecasting
6. ✅ **Performance Benchmarks**: Quantified accuracy metrics and competitive model comparisons

---

## 🔗 Integration with Hourly System

This daily implementation **complements** the hourly forecasting system for comprehensive weather intelligence:

### **System Synergy**
- **Daily System**: Optimal for strategic planning, seasonal analysis, and medium-term forecasting
- **Hourly System**: Optimal for operational decisions, short-term optimization, and tactical planning
- **Combined Approach**: Leverage both systems for complete temporal coverage and decision support

### **When to Use Daily vs Hourly Forecasting**

| Use Case | Daily System | Hourly System | Rationale |
|----------|-------------|---------------|-----------|
| **Agricultural Planning** | ✅ Primary | ➖ Secondary | Crop cycles operate on daily+ timescales |
| **Energy Grid Planning** | ✅ Primary | ➖ Secondary | Strategic capacity planning needs daily trends |
| **Event Planning** | ✅ Primary | ➖ Secondary | Events planned days/weeks in advance |
| **HVAC Operations** | ➖ Secondary | ✅ Primary | Real-time optimization needs hourly precision |
| **Emergency Response** | ➖ Secondary | ✅ Primary | Rapid response requires immediate forecasts |
| **Transportation** | ➖ Secondary | ✅ Primary | Route planning needs real-time conditions |

---

## 📚 References & Further Reading

- **Time Series Forecasting**: Hyndman & Athanasopoulos - "Forecasting: Principles and Practice" (3rd Edition)
- **Climate Data Analysis**: World Meteorological Organization Guidelines on Climate Data and Information
- **Seasonal Forecasting**: NOAA Climate Prediction Center Methodologies and Best Practices
- **Agricultural Weather**: FAO Guidelines on Agricultural Meteorology for Weather-Dependent Decision Making
- **Machine Learning for Weather**: Nielsen et al. - "Practical Time Series Analysis" for Weather Prediction
- **Feature Engineering**: Kuhn & Johnson - "Feature Engineering and Selection" for Predictive Models

---

## 🤝 Contributing & Support

### **Contributing Guidelines**
1. Fork the repository and create a feature branch
2. Implement changes with comprehensive testing
3. Update documentation and examples
4. Submit pull request with detailed description

### **Getting Help**
- 📖 Check this documentation and notebook implementations first
- 🐛 Report bugs through GitHub issues with reproduction steps
- 💡 Request features through GitHub discussions
- 📧 Contact the development team for technical support

---

*This README provides comprehensive guidance for implementing daily temperature forecasting for Hanoi. The daily system forms the foundation for strategic weather planning and complements the hourly system for complete weather intelligence coverage.*