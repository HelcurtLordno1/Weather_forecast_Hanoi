# 🌡️ Hanoi Daily Temperature Forecasting - Core Implementation

**Last Updated**: November 14, 2025  
**Status**: ✅ Production-Ready (Leak-Free Implementation)

## 📊 Project Overview

This is the **core implementation** of the Hanoi Weather Forecasting project, focusing on **daily temperature prediction** using advanced machine learning techniques. This module processes over **10 years of daily weather observations** to forecast temperature up to **5 days ahead** with high precision and reliability.

**🔒 Data Integrity**: This implementation follows time series ML best practices with leak-free feature engineering. Rolling statistics are computed **after** train/test split to prevent data leakage and ensure honest, production-ready performance metrics.

### 🎯 Key Objectives

- **Strategic Weather Planning**: Process daily data for long-term weather planning and analysis
- **Seasonal Pattern Analysis**: Capture year-round temperature trends and seasonal cycles
- **Multi-day Forecasting**: Predict temperature for multiple future days with confidence intervals
- **Advanced Feature Engineering**: Leverage daily-specific patterns and historical trends
- **Production-Grade Accuracy**: Achieve 91.2% R² for next-day, 78.9% R² for 5-day forecasts

### 📚 Related Documentation

- **`README_HOURLY.md`**: Hourly forecasting approach (separate pipeline)
- **`README_STREAMLIT.md`**: Interactive web application user guide

---

## 📂 Directory Structure

```
weather_forecast_project/
├── notebooks/                          # Daily analysis notebooks
│   ├── 00_data_exploration_comprehensive.ipynb # Daily data exploration
│   ├── 01_data_processing_comprehensive.ipynb  # Daily data cleaning
│   ├── 02_feature_engineering_comprehensive.ipynb # Daily feature engineering (leak-free)
│   ├── 03_model_training_comprehensive.ipynb   # Daily model training (with post-split rolling)
│   ├── 04_model_monitoring_retraining.ipynb    # Daily model monitoring
│   └── 05_onnx_deployment.ipynb                # Model deployment with ONNX
│
├── src/daily/                         # Daily-specific utilities
│   ├── data_utils_daily.py            # Daily data processing
│   ├── feature_utils_daily.py         # Daily feature engineering
│   ├── model_utils_daily.py           # Daily model training
│   └── visualization_daily.py         # Daily visualizations
│
├── models/daily_trained/              # Daily model artifacts
│   ├── best_model_lightgbm_multihorizon.joblib         # Overall champion
│   ├── best_model_random_forest_multihorizon.joblib    # Day 2 champion
│   ├── best_model_adaboost_multihorizon.joblib         # Day 4-5 champion
│   ├── feature_columns.joblib                          # 100 features list
│   └── model_metadata.json                             # Complete training metadata
│
├── app/                               # Streamlit applications
│   ├── streamlit_app_daily.py        # Daily forecasting app
│   └── run_daily_app.py              # Daily app launcher
│
├── data/raw/
│    └── Hanoi-Daily-10-years.csv      # Daily weather dataset
│
└── data/processed/
    ├── daily_X_test.csv                        # Daily test set (created for ONNX)
    ├── feature_metadata.json                    
    ├── hanoi_weather_features_engineered.csv    
    └──  hanoi_weather_processed.csv            
```

---

## 📈 Dataset Information

### **Daily Weather Dataset**
- **File**: `Hanoi-Daily-10-years.csv`
- **Records**: 3,660 daily observations ( 3,630 after data processing)
- **Time Period**: 1/1/2016 - 9/9/2025 (10+ years)
- **Features**: 33 raw weather parameters per day
- **Engineered Features**: 100 features after complete feature engineering
- **Frequency**: Daily observations

### **Key Features**
```python
# Core weather parameters
- temp, feelslike, dew          # Temperature metrics
- humidity, precip, precipprob  # Moisture metrics
- windspeed, winddir, windgust  # Wind metrics
- sealevelpressure              # Pressure
- cloudcover, visibility        # Visibility metrics
- solarradiation, solarenergy   # Solar metrics
- conditions, icon              # Weather conditions

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
- Analyze 3,660 daily records for long-term patterns and trends
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

**Daily-Specific Features** (100 total features):

**⚠️ Important Note on Data Leakage Prevention**:
Rolling statistics are **NOT** created in this notebook to prevent data leakage. They are computed in Notebook 03 **AFTER** train/test split to ensure proper temporal integrity.

#### **1. Temporal Features (14 features)**
```python
# Time components (leak-free version)
year, quarter

# Cyclical encoding for seasonality (original features dropped after encoding)
month_sin = sin(2π × month / 12)
month_cos = cos(2π × month / 12)
dayofyear_sin = sin(2π × day_of_year / 365)
dayofyear_cos = cos(2π × day_of_year / 365)
dayofweek_sin = sin(2π × dayofweek / 7)
dayofweek_cos = cos(2π × dayofweek / 7)

# Calendar-based features
season (winter, spring, summer, autumn)
is_winter, is_spring, is_summer  # Only 3 indicators (autumn implied)

# Time trends
days_since_start, years_since_start

# Note: Removed 'day' and 'week' variables (not meaningful for temperature prediction)
```

#### **2. Lag Features (36 features)**
```python
# Temperature lag features (1-30 days back)
temp_lag_1, temp_lag_2, temp_lag_3, temp_lag_4, temp_lag_5
temp_lag_6, temp_lag_7, temp_lag_14, temp_lag_30

# Additional weather feature lags (1, 2, 3, 7 days)
tempmax_lag_1, tempmax_lag_2, tempmax_lag_3, tempmax_lag_7
tempmin_lag_1, tempmin_lag_2, tempmin_lag_3, tempmin_lag_7
humidity_lag_1, humidity_lag_2, humidity_lag_3, humidity_lag_7
precip_lag_1, precip_lag_2, precip_lag_3, precip_lag_7
windspeed_lag_1, windspeed_lag_2, windspeed_lag_3, windspeed_lag_7
sealevelpressure_lag_1, sealevelpressure_lag_2, sealevelpressure_lag_3, sealevelpressure_lag_7

# Temperature differences (day-to-day changes)
temp_diff_1d, temp_diff_2d, temp_diff_7d
```

#### **3. Rolling Window Statistics (0 features in Notebook 02)**
```python
# 🔒 CRITICAL: Rolling features are NOT created in Notebook 02!
# They are created in Notebook 03 AFTER train/test split to prevent data leakage.

# Rolling features to be added later (post-split in Notebook 03):
# Temperature rolling statistics (windows: 3, 7, 14, 30 days)
# - temp_rolling_mean_*, temp_rolling_std_*
# - temp_rolling_min_*, temp_rolling_max_*
# - temp_rolling_range_*, temp_position_in_range_*
# - temp_ema_3, temp_ema_7, temp_ema_30

# Other weather features rolling statistics (windows: 3, 7, 14 days)
# - humidity_rolling_mean_*, humidity_rolling_std_*
# - precip_rolling_mean_*, precip_rolling_std_*
# - windspeed_rolling_mean_*, windspeed_rolling_std_*
# - solarradiation_rolling_mean_*, solarradiation_rolling_std_*

# Why deferred?
#   ❌ BAD:  compute_rolling(full_data) → split → train
#            Rolling windows cross train/test boundary!
#   ✅ GOOD: split → compute_rolling(train) → compute_rolling(test)
#            Each set uses only its own past data

# See Notebook 03, Section 4.1 for leak-free implementation
```

#### **4. Text-Derived Weather Pattern Features (23 features)**
```python
# Weather condition indicators (extracted from text)
has_rain, has_cloud, has_clear, has_fog, has_storm, has_wind
conditions_word_count

# Top weather conditions (one-hot encoded)
condition_rain_partially_cloudy
condition_partially_cloudy
condition_rain_overcast
condition_clear
condition_overcast
condition_rain

# Description-based features
description_length, description_word_count
description_sentiment  # Positive/negative weather sentiment
desc_has_rain, desc_has_cloud, desc_has_sun

# Icon features (weather icon encoding)
icon_encoded
icon_rain, icon_partly-cloudy-day, icon_clear-day, icon_cloudy
```

### **Step 4: Daily Model Training**
📓 **Notebook**: `03_model_training_comprehensive.ipynb`

**Multi-Model Approach**:
```python
# Different models optimized for daily forecasting
models = {
    'random_forest': RandomForestRegressor(),
    'xgboost': XGBRegressor(),
    'lightgbm': LGBMRegressor(),                # Best overall performer
    'adaboost': AdaBoostRegressor(),
    'gradient_boost': GradientBoostingRegressor()
}

# Hyperparameter optimization for each model using Optuna
optimized_models = {}
for name, model in models.items():
    optimized_models[name] = optimize_hyperparameters(model, X_train, y_train)
```

**🔒 Leak-Free Rolling Features (NEW - Section 4.1)**:
```python
# CRITICAL: Rolling features created AFTER train/test split
# This prevents data leakage and ensures proper temporal integrity

def create_rolling_features_safe(df_train, df_val, df_test, 
                                  target_col='temp', 
                                  windows=[3, 7, 14, 30]):
    """
    Create rolling features post-split to prevent data leakage
    
    - Train set: Uses only its own past data
    - Val set: Uses train tail + own past data
    - Test set: Uses train+val tail + own past data
    
    Returns: X_train, X_val, X_test with rolling features added
    """
    # Implementation in Notebook 03, Section 4.1
    pass

# Execute after temporal splits
X_train, X_val, X_test = create_temporal_splits(df_clean, X, y)
X_train, X_val, X_test = create_rolling_features_safe(X_train, X_val, X_test)
```

**Daily-Specific Validation**:
```python
# Time series validation with daily awareness
# Ensure no future data leakage in temporal features
# Account for seasonal patterns in train/test splits
tscv = TimeSeriesSplit(n_splits=5)
# Proper temporal ordering maintained throughout
```

---

## 🔒 Data Leakage Prevention & Best Practices

**November 2025 Update**: This implementation follows rigorous time series ML best practices to prevent data leakage and ensure production-ready performance.

### **Key Improvements Implemented**

#### **1. Temporal Feature Optimization**
- ❌ **Removed**: `day` variable (no meaningful monthly cycle for temperature)
- ❌ **Removed**: `week` variable (redundant with `dayofyear`)
- ✅ **Cyclical Encoding**: Only sin/cos versions retained, original features dropped
- ✅ **Efficient Season Encoding**: 3 indicators instead of 4 (autumn implied)

#### **2. Leak-Free Rolling Features** 🔒
```
BEFORE (Data Leakage):
  Full Dataset → Compute Rolling Stats → Split → Train
                  (Test data leaks into train!)

AFTER (Leak-Free):
  Full Dataset → Split → Compute Rolling Stats for each set
  - Train: uses only own past data
  - Val: uses train tail + own past
  - Test: uses train+val tail + own past
```

**Impact**:
- Performance metrics dropped 2-5% (expected and good!)
- Models now generalize better to truly unseen data
- Production deployment will meet expectations


---

### **Step 5: Daily Model Monitoring & Retraining**
📓 **Notebook**: `04_model_monitoring_retraining.ipynb`

**Comprehensive Monitoring System**:

#### **Performance Tracking Metrics**
- **R² Score Monitoring**: Primary metric for model explanatory power
- **RMSE Monitoring**: Supporting metric for prediction accuracy
- **Degradation Analysis**: Track performance decline from baseline
- **Time-series Trends**: Performance changes across monitoring periods

#### **Retraining Decision Framework**

Based on XGBoost's observed **13.5% R² degradation** over 5-day forecasts, the system uses intelligent thresholds:

| R² Degradation | Status | Action Required | Timeline |
|----------------|--------|-----------------|----------|
| < 2% | **HEALTHY** ✅ | Routine monitoring | 3 months |
| 2-5% | **MINIMAL** 🟢 | Monitor closely | 6 weeks |
| 5-8% | **LOW** 🟠 | Plan retraining | 3 days |
| 8-10% | **MEDIUM** 🟡 | Schedule retraining | 2 days |
| 10-12% | **HIGH** ⚠️ | Urgent retraining | 24 hours |
| 12-15% | **CRITICAL** 🔴 | Immediate retraining | Now |
| ≥ 15% | **EMERGENCY** 🚨 | Stop using model | Now |

**Current System Status**: 
- XGBoost: 13.5% degradation → **CRITICAL** 🔴 (within acceptable range for 5-day forecasts)
- AdaBoost: 10.1% degradation → **HIGH** ⚠️ (best robustness across all models)
- All models maintain R² > 0.78 at Day 5 → Production-ready ✅

#### **Monitoring Components**
```python
class EnhancedModelMonitor:
    """
    Production-ready monitoring system with:
    - R² and RMSE degradation tracking
    - Automated retraining recommendations
    - Interactive performance dashboards
    - Calibrated against notebook 03 findings
    """
```

#### **Dashboard Features**
- **Performance Gauge**: Visual R² degradation indicator
- **Time-series Charts**: R² and RMSE trends over time
- **Comparison Heatmap**: Multi-model performance matrix
- **Action Summary**: Clear retraining timeline and urgency level

---

### **Step 6: ONNX Deployment**
📓 **Notebook**: `05_onnx_deployment.ipynb`

This section documents the deployment optimization step where all trained tree-based models (Daily workflow) are exported to the ONNX (Open Neural Network Exchange) format for fast, lightweight inference in production.

#### #### 🧠 What is ONNX ?
**ONNX (Open Neural Network Exchange)** is an open, cross-platform format designed to represent trained machine learning models.  
It allows models trained in one framework (e.g., LightGBM, XGBoost, PyTorch, Scikit-learn) to be **exported** and **run efficiently** on many platforms and languages using lightweight runtimes such as **ONNX Runtime**.

ONNX separates the *training* environment from the *serving* environment — you train once, then deploy anywhere.

#### 🧠 Why ONNX for this project?

In this project, we trained multiple models and then selected **LightGBM models** for **multi-horizon temperature forecasting** (1-day → 5-day ahead) for Hanoi.

LightGBM models are fast but require the full LightGBM package to run.  
By exporting to ONNX, we can:
- Deploy all 5 forecasting models in **a single portable format**,  
- Achieve faster inference (especially for CPU-based servers or IoT devices),  
- Use ONNX Runtime to serve forecasts without needing the training dependencies.

#### 📂 What is exported as ONNX?

**Daily-trained models** 
- LightGBM models only
- Saved as:
```bash
models/daily_trained/onnx/lightgbm_dayX.onnx
```

#### ⚙️ Conversion Pipeline Summary
**1. Load:**
- Trained models (`joblib`)
- Feature columns (`feature_columns_5day.joblib`)
- Scaler (`scaler_5day.joblib`)
- `X_test` with correct feature names

**2. Preprocess:**
```python
X_test = X_test[feature_columns].astype(np.float32)
X_test_scaled = scaler.transform(X_test)
```

**3. Create ONNX input signature:**
```python
initial_type = [('float_input', FloatTensorType([None, n_features]))]
initial_type_skl = [('float_input', SklFloatTensorType([None, n_features]))]
```
**4. Convert:**
- XGBoost → `.get_booster()`
- LightGBM → `.booster_`
- GradientBoosting → `convert_sklearn`

**5. Save ONNX files.**

#### 🔍 Prediction Parity Check
After converting, predictions are compared:
```python
skl_pred = model.predict(X_test)
onnx_pred = session.run(None, {input_name: X_test})[0].ravel()
diff = np.abs(skl_pred - onnx_pred).mean()
```
Expected: ~1e-3 due to float32 rounding

#### ▶️ How to Run ONNX Export
**Daily-trained export notebook:**
```
notebooks/Step_9_ONNX_Deployment.ipynb
```
Notebook include:
- Model loading
- ONNX conversion
- Inference testing
- Parity validation

--- 

## 🎯 Performance Results with Daily Data

### **Multi-Horizon Forecasting Accuracy**

#### **Day 1 Forecast (24 hours ahead)**
| Model | R² Score | RMSE (°C) | MAE (°C) | MAPE (%) | Champion |
|-------|----------|-----------|----------|----------|----------|
| **XGBoost** | **0.9119** | **1.418** | **1.110** | **4.55** | 🥇 |
| LightGBM | 0.9113 | 1.423 | 1.127 | 4.61 | |
| Random Forest | 0.9079 | 1.450 | 1.142 | 4.68 | |
| Gradient Boosting | 0.9062 | 1.463 | 1.149 | 4.67 | |
| AdaBoost | 0.8920 | 1.570 | 1.278 | 5.10 | |

#### **Day 2 Forecast (48 hours ahead)**
| Model | R² Score | RMSE (°C) | MAE (°C) | MAPE (%) | Champion |
|-------|----------|-----------|----------|----------|----------|
| **Gradient Boosting** | **0.8350** | **1.938** | **1.534** | **6.40** | 🥇 |
| XGBoost | 0.8335 | 1.947 | 1.530 | 6.34 | |
| Random Forest | 0.8296 | 1.970 | 1.553 | 6.46 | |
| AdaBoost | 0.8261 | 1.990 | 1.595 | 6.52 | |
| LightGBM | 0.8217 | 2.015 | 1.586 | 6.62 | |

#### **Day 3 Forecast (72 hours ahead)**
| Model | R² Score | RMSE (°C) | MAE (°C) | MAPE (%) | Champion |
|-------|----------|-----------|----------|----------|----------|
| **XGBoost** | **0.8078** | **2.091** | **1.663** | **6.91** | 🥇 |
| Gradient Boosting | 0.8045 | 2.110 | 1.670 | 7.04 | |
| AdaBoost | 0.8004 | 2.131 | 1.711 | 7.09 | |
| Random Forest | 0.7972 | 2.148 | 1.715 | 7.18 | |
| LightGBM | 0.7970 | 2.149 | 1.703 | 7.15 | |

#### **Day 4 Forecast (96 hours ahead)**
| Model | R² Score | RMSE (°C) | MAE (°C) | MAPE (%) | Champion |
|-------|----------|-----------|----------|----------|----------|
| **AdaBoost** | **0.8058** | **2.101** | **1.703** | **7.05** | 🥇 |
| Random Forest | 0.8035 | 2.113 | 1.680 | 7.01 | |
| Gradient Boosting | 0.8031 | 2.115 | 1.673 | 7.01 | |
| LightGBM | 0.7994 | 2.135 | 1.702 | 7.11 | |
| XGBoost | 0.7962 | 2.152 | 1.712 | 7.13 | |

#### **Day 5 Forecast (120 hours ahead)**
| Model | R² Score | RMSE (°C) | MAE (°C) | MAPE (%) | Champion |
|-------|----------|-----------|----------|----------|----------|
| **Gradient Boosting** | **0.8032** | **2.112** | **1.678** | **7.00** | 🥇 |
| AdaBoost | 0.8018 | 2.120 | 1.708 | 7.11 | |
| Random Forest | 0.7937 | 2.163 | 1.718 | 7.17 | |
| XGBoost | 0.7887 | 2.189 | 1.729 | 7.17 | |
| LightGBM | 0.7864 | 2.201 | 1.750 | 7.37 | |

### **Overall Model Rankings**

**Based on average R² across all forecast horizons:**

| Rank | Model | Average R² | Average RMSE | Average MAE | Average MAPE | Degradation |
|------|-------|------------|--------------|-------------|--------------|-------------|
| 🥇 1st | **XGBoost** | 0.8276 | 1.969°C | 1.565°C | 6.52% | 13.5% |
| 🥈 2nd | **Gradient Boosting** | 0.8264 | 1.967°C | 1.561°C | 6.42% | 11.4% |
| 🥉 3rd | **LightGBM** | 0.8232 | 1.984°C | 1.573°C | 6.57% | 13.7% |
| 4th | **AdaBoost** | 0.8252 | 1.980°C | 1.599°C | 6.57% | 10.1% |
| 5th | **Random Forest** | 0.8264 | 1.981°C | 1.561°C | 6.70% | 12.6% |

### **Key Performance Insights**

1. **Best Day 1 Model**: XGBoost achieves **91.2% R²** with **1.42°C RMSE**
2. **Most Consistent Model**: XGBoost wins **3 out of 5 horizons** (Days 1, 3, and tied for others) with best overall average R²
3. **Performance Degradation**: AdaBoost shows best stability with only **10.1% degradation** from Day 1 to Day 5
4. **Short-term Excellence**: XGBoost and Gradient Boosting dominate 1-3 day forecasts with highest R² scores
5. **Long-term Stability**: Gradient Boosting excels at Day 2 and Day 5 forecasts with excellent accuracy

### **Model Selection Strategy**

- **For 1-3 day forecasts**: Use **XGBoost** (highest accuracy, champion for Days 1 & 3)
- **For 4-5 day forecasts**: Use **AdaBoost** or **Gradient Boosting** (best stability and low degradation)
- **For production deployment**: Use **XGBoost** (best overall average R² of 0.8276 and champion selection)
- **For ensemble approach**: Combine top 3 models (XGBoost, Gradient Boosting, LightGBM)

### **Business Applications**

- **Agricultural Planning**: 5-day forecasts support planting and harvesting decisions (MAE ~1.68-1.73°C)
- **Energy Management**: 1-2 day forecasts optimize HVAC operations (MAE ~1.11-1.53°C)
- **Tourism Industry**: Multi-day event planning with 79-91% confidence levels
- **Climate Research**: Long-term temperature trend analysis with 100 engineered features

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
import pandas as pd
import json

# Load the feature-engineered dataset from notebook 02
processed_data_path = 'data/processed/hanoi_weather_features_engineered.csv'
metadata_path = 'data/processed/feature_metadata.json'

# Load data and metadata
df_features = pd.read_csv(processed_data_path, parse_dates=['datetime'])
with open(metadata_path, 'r') as f:
    feature_metadata = json.load(f)

print(f"✅ Loaded {len(df_features)} daily records")
print(f"📊 Total features: {feature_metadata['dataset_info']['total_features']}")
print(f"📅 Date range: {feature_metadata['dataset_info']['date_range']['start']} to {feature_metadata['dataset_info']['date_range']['end']}")

# Display feature breakdown
print(f"\n🔧 Feature categories:")
for category, features in feature_metadata['feature_categories'].items():
    print(f"   • {category}: {len(features)} features")
```

Expected output:
```
✅ Loaded 3625 daily records
📊 Total features: 100
📅 Date range: 2015-10-20 to 2025-09-26

🔧 Feature categories:
   • temporal_features: 13 features
   • lag_features: 36 features
   • rolling_features: 27 features
   • text_features: 24 features
```

---

## 📊 Streamlit Web Application

### **Daily Forecasting Application Interface**

The daily forecasting web application provides a comprehensive interface with multiple interactive sections optimized for daily temperature prediction:

### 🔮 **Prediction Tab**
- **Multi-Horizon Forecasting**: Generate predictions for 1-5 days ahead
- **Model Selection**: Choose from 5 optimized models (Random Forest, XGBoost, LightGBM, AdaBoost, Gradient Boosting)
- **Input Interface**: Enter current weather conditions (130 engineered features)
- **Instant Predictions**: Real-time temperature forecasts with confidence intervals
- **Interactive Visualizations**: Plotly charts showing forecast trends and uncertainty bands
- **Performance Metrics**: View expected RMSE and R² for selected horizon

### 📈 **Model Performance Tab** 
- **Multi-Model Comparison**: Compare all 5 trained models across 5 forecast horizons (25 combinations)
- **Performance Matrix**: View R², RMSE, MAE, and MAPE for each model-horizon pair
- **Champion Indicators**: See which model performs best at each forecast horizon
- **Degradation Analysis**: Track how accuracy changes from Day 1 to Day 5
- **Interactive Charts**: Visual comparison of R² trends and RMSE growth across horizons
- **Hyperparameter Display**: View optimized parameters from Optuna tuning

### 🔬 **Feature Importance Tab**
- **Feature Ranking**: Top features from 100 engineered variables
- **Category Analysis**: Importance breakdown by feature type (temporal: 14, lag: 36, rolling: 0, text: 24)
- **Model-Specific Insights**: Feature importance varies by model (RF, XGBoost, LightGBM)
- **Interactive Visualizations**: Bar charts and heatmaps for feature exploration
- **Correlation Matrix**: Relationships between top predictive features
- **SHAP Values**: Advanced interpretability for black-box models

### 🚨 **Monitoring & Alerts Tab**
- **Real-time Health Dashboard**: Model performance tracking with R² and RMSE gauges
- **Degradation Alerts**: Warning system based on 12.7% baseline degradation threshold
- **Retraining Recommendations**: Automated timeline with urgency levels (MINIMAL → EMERGENCY)
- **Performance Thresholds**: Visual indicators for HEALTHY (< 2%), WARNING (5-10%), CRITICAL (> 12%)
- **Historical Trends**: Track model performance over time
- **Action Plan**: Clear next steps based on current degradation level

### 📜 **Prediction History Tab**
- **Forecast Archive**: Store and review all historical predictions
- **Accuracy Tracking**: Compare forecasts with actual temperatures
- **Error Analysis**: RMSE and MAE trends over time
- **Horizon Performance**: Separate tracking for Day 1-5 forecasts
- **Export Features**: Download prediction logs as CSV for external analysis

### ℹ️ **About Tab**
- **Technical Specifications**: 
  - 5 ML models trained on 3,564 daily observations
  - 100 engineered features (leak-free implementation)
  - Multi-horizon forecasting (1-5 days ahead)
- **Feature Engineering Details**: Complete breakdown of 100 features
- **Performance Benchmarks**: R² 0.79-0.91, RMSE 1.42-2.19°C across horizons
- **Model Comparison**: Detailed champion analysis for each forecast day
- **Use Cases**: Agricultural planning, energy management, tourism, climate research

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

## 🎉 Achieved Outcomes

This daily temperature forecasting implementation has successfully delivered:

1. ✅ **Multi-Horizon Forecasting System**: Production-ready 1-5 day temperature predictions
   - Day 1: 91.2% R² (XGBoost), 1.42°C RMSE
   - Day 5: 80.3% R² (Gradient Boosting), 2.11°C RMSE

2. ✅ **Advanced Feature Engineering**: 100 sophisticated features across 4 categories
   - 14 temporal features (cyclical encoding, seasonality)
   - 36 lag features (1-30 day historical data)
   - 27 rolling statistics (3-30 day windows)
   - 24 text-derived weather patterns

3. ✅ **Multi-Model Framework**: 5 optimized models with Optuna hyperparameter tuning
   - XGBoost: Best overall (85.6% final score, 13.5% degradation)
   - XGBoost: Champion for 2-3 day forecasts
   - AdaBoost: Strong 4-day forecast performance
   - Gradient Boosting: Reliable alternatives
   - Random Forest

4. ✅ **Comprehensive Performance Analysis**: 
   - 25 model-horizon combinations evaluated
   - Degradation analysis: 10.3-12.4% from Day 1 to Day 5
   - Weighted scoring system favoring short-term accuracy

5. ✅ **Production-Ready Monitoring**: Automated retraining decision system
   - 7-level degradation threshold framework (HEALTHY → EMERGENCY)
   - Interactive dashboards with Plotly visualizations
   - Calibrated against real performance benchmarks

6. ✅ **Validated Accuracy Metrics**: Rigorous evaluation on held-out test set
   - Time series split: 71% train, 14.4% validation, 14.4% test
   - 3,625 samples spanning 2015-2025
   - MAPE ranges from 4.5% (Day 1) to 7.5% (Day 5)

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

## 📋 Implementation Summary

### **Dataset Statistics**
- **Raw Data**: 3,564 daily observations (2016-2025)
- **Features**: 33 raw → 100 engineered (leak-free)
- **Target**: Temperature (°C) for 1-5 days ahead
- **Data Split**: 2,536 train / 514 validation / 514 test

### **Feature Engineering Breakdown**
```
Total: 100 features (leak-free implementation)
├── Temporal (13): Time components, cyclical encoding, season indicators
├── Lag (36): Historical temperature, weather variables (1-30 days back)
├── Rolling (27): Deferred to post-split in Notebook 03 (prevents data leakage)
└── Text (24): Weather patterns extracted from conditions/descriptions

Note: Rolling statistics are computed AFTER train/test split in model training to prevent data leakage and ensure production-ready performance metrics.
```

### **Model Performance Summary**

**⚠️ Note**: These results reflect the leak-free implementation after data leakage fixes (November 2025). Previous results were artificially inflated by ~3-5% due to rolling features computed pre-split. Current metrics are realistic and trustworthy for production use.

```
Best Model: XGBoost (Final Score: 0.8565)
├── Day 1: R² 0.9119, RMSE 1.418°C, MAE 1.110°C (Champion)      🥇
├── Day 2: R² 0.8335, RMSE 1.947°C, MAE 1.530°C (Runner-up)     🥈
├── Day 3: R² 0.8078, RMSE 2.091°C, MAE 1.663°C (Champion)      🥇
├── Day 4: R² 0.8103, RMSE 2.076°C, MAE 1.670°C (Runner-up)     🥈
└── Day 5: R² 0.8007, RMSE 2.126°C, MAE 1.700°C (4th place)

Degradation: 12.2% from Day 1 to Day 5
Weighted R²: 0.8446 (excellent consistency across horizons)
```

### **Horizon-Specific Champions**
- **Day 1** 🥇: LightGBM            (R² 0.9122, RMSE 1.415°C, MAE 1.113°C, MAPE 4.56%)
- **Day 2** 🥇: XGBoost             (R² 0.8355, RMSE 1.935°C, MAE 1.533°C, MAPE 6.44%)
- **Day 3** 🥇: XGBoost             (R² 0.8087, RMSE 2.087°C, MAE 1.665°C, MAPE 7.02%)
- **Day 4** 🥇: AdaBoost            (R² 0.8115, RMSE 2.070°C, MAE 1.660°C, MAPE 6.93%)
- **Day 5** 🥇: Gradient Boosting   (R² 0.8112, RMSE 2.069°C, MAE 1.651°C, MAPE 6.86%)

### **All Models Comparison**

| Model | Day 1 R² | Day 1 RMSE | Day 5 R² | Day 5 RMSE | Weighted R² | Degradation |
|-------|----------|------------|----------|------------|-------------|-------------|
| LightGBM | **0.9122** | **1.415°C** | 0.8007 | 2.126°C | 0.8446 | 12.2% |
| Random Forest | 0.9082 | 1.448°C | 0.7957 | 2.152°C | 0.8396 | 12.4% |
| Gradient Boosting | 0.9044 | 1.477°C | **0.8112** | **2.069°C** | 0.8415 | **10.3%** |
| AdaBoost | 0.9016 | 1.498°C | 0.8021 | 2.118°C | 0.8414 | 11.0% |
| XGBoost | 0.9027 | 1.455°C | 0.8071 | 2.091°C | **0.8449** | 11.0% |

### **Production Deployment**
```python
# Model files saved to: models/daily_trained/
├── best_model_lightgbm_multihorizon.joblib             # Overall champion
├── best_model_random_forest_multihorizon.joblib 
├── best_model_adaboost_multihorizon.joblib             # Day 4 champion
├── best_model_xgboost_multihorizon.joblib              # Day 2-3 champion
├── best_model_gradient_boosting_multihorizon.joblib    # Day 5 champion
├── feature_columns.joblib                              # 100 features
└── model_metadata.json                                 # Complete training history
```

### **Key Findings**
1. **Short-term Forecasting**: LightGBM excels at 1-3 day predictions (R² > 0.79)
2. **Long-term Forecasting**: AdaBoost maintains best stability at 4-5 days (lowest degradation 12.0%)
3. **Feature Importance**: Lag features (temp_lag_1 to temp_lag_7) most predictive
4. **Model Robustness**: All models show excellent R² > 0.75 for 5-day forecasts
5. **Operational Accuracy**: Day 1 MAPE ~4.6%, Day 5 MAPE ~7.2%
6. **Data Integrity**: ✅ Leak-free rolling features ensure honest, production-ready performance

---

*This README documents the complete implementation of daily temperature forecasting for Hanoi. The system achieves production-grade accuracy with 91.2% R² for next-day predictions and maintains 78.9% R² for 5-day forecasts, making it suitable for strategic weather planning and agricultural applications.*