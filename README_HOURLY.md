# 🌡️ Hanoi Hourly Temperature Forecasting - Step 8 Implementation

## 📊 Project Overview

This is the **Step 8** implementation of the Hanoi Weather Forecasting project, leveraging **hourly weather observations aggregated to daily level** for 5-day temperature prediction. This module processes over **87,000 hourly weather observations** (aggregated to 3,649 daily records) to forecast temperature for **Days 1-5 ahead** with enhanced precision by capturing diurnal patterns.

### 🎯 Key Objectives

- **Rich Temporal Features**: Leverage hourly granularity to capture diurnal patterns, then aggregate to daily
- **Diurnal Pattern Analysis**: Extract within-day temperature cycles, solar radiation patterns, and hourly variability
- **Multi-horizon Forecasting**: Predict daily average temperature for 5 consecutive future days
- **Advanced Feature Engineering**: 91 features from hourly aggregations (mean, min, max, std, sum)
- **Data Leakage Prevention**: Careful feature engineering using only past information

---

## 📂 Directory Structure

```
weather_forecast_project/
├── notebooks_hourly/                    # Hourly analysis notebooks
│   ├── 00_data_exploration_hourly.ipynb    # Hourly data exploration
│   ├── 01_data_processing_hourly.ipynb     # Hourly data cleaning
│   ├── 02_feature_engineering_hourly.ipynb # Hourly feature engineering
│   ├── 03_model_training_hourly.ipynb      # Hourly model training
│   ├── 04_model_monitoring_hourly.ipynb    # Hourly model monitoring
│   └── 05_onnx_deployment_hourly.ipynb      # Hourly model deployment with ONNX
│
├── src/hourly/                     # Hourly-specific utilities
│   ├── data_utils_hourly.py            # Hourly data processing
│   ├── data_aggregation.py             # Daily data aggregated from hourly data
│   ├── feature_utils_hourly.py         # Hourly feature engineering
│   └── leak_free_features.py           
│
├── models/hourly_trained/              # Hourly model artifacts
│   ├── best_model_hourly.joblib       # Best hourly model
│   ├── feature_columns_hourly.joblib  # Hourly feature list
│   └── model_metadata_hourly.json     # Hourly model metadata
│
├── app/                             # Streamlit applications
│   ├── streamlit_app_hourly.py        # Hourly forecasting app
│   └── run_hourly_app.py              # Hourly app launcher
│
└── data/raw/
    └── hanoi_weather_data_hourly.csv  # Hourly weather dataset
```

---

## 📈 Dataset Information

### **Hourly Weather Dataset**
- **File**: `hanoi_weather_data_hourly.csv`
- **Records**: ~87,698 hourly observations
- **Time Period**: 2015-2024 (10 years)
- **Features**: 28 raw weather parameters per hour
- **Aggregated Daily Records**: 3,649 days (for 5-day forecasting)
- **Final Features**: 91 features after aggregation and engineering
- **Frequency**: Hourly observations aggregated to daily for forecasting

### **Key Features**
```python
# Core weather parameters (same as daily)
- temp, feelslike, dew          # Temperature metrics
- humidity, precip, precipprob  # Moisture metrics
- windspeed, winddir, windgust  # Wind metrics
- sealevelpressure             # Pressure
- cloudcover, visibility       # Visibility metrics
- solarradiation, solarenergy  # Solar metrics
- conditions, icon             # Weather conditions

# Hourly-specific advantages
- Diurnal temperature cycles
- Rush hour patterns
- Peak heating/cooling periods
- Weather transition timing
```

---

## 🔬 Step-by-Step Analysis Workflow

### **Step 8.1: Hourly Data Exploration** 
📓 **Notebook**: `00_data_exploration_hourly.ipynb`

**Objectives**:
- Analyze 87K+ hourly records for patterns
- Explore diurnal (24-hour) temperature cycles
- Identify hourly weather correlations
- Assess data quality and completeness

**Key Analyses**:
```python
# Diurnal pattern analysis
hourly_avg_temp = df.groupby('hour')['temp'].mean()
seasonal_hourly_patterns = df.groupby(['month', 'hour'])['temp'].mean()

# Weather transition analysis
temp_change_1h = df['temp'].diff(1)
rapid_changes = temp_change_1h[abs(temp_change_1h) > 3]  # >3°C/hour

# Peak period identification
peak_heat_hours = df[df.groupby('date')['temp'].transform('max') == df['temp']]
coolest_hours = df[df.groupby('date')['temp'].transform('min') == df['temp']]
```

### **Step 8.2: Hourly Data Processing**
📓 **Notebook**: `01_data_processing_hourly.ipynb`

**Enhanced Processing**:
- **Temporal Gap Detection**: Identify missing hours in sequence
- **Hourly Interpolation**: Time-aware missing value handling
- **Outlier Detection**: Hour-specific outlier identification
- **Quality Assessment**: Hourly data completeness metrics

**Unique Challenges**:
```python
# Handle daylight saving time transitions
# Detect instrument downtime periods  
# Account for weather station maintenance windows
# Process irregular reporting intervals
```

### **Step 8.3: Advanced Hourly Feature Engineering**
📓 **Notebook**: `02_feature_engineering_hourly.ipynb`

**Hourly-Level Features** (created from 87,698 hourly observations):

#### **1. Temporal Features (Cyclical Encoding)**
```python
# 24-hour cycle encoding
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)

# 7-day cycle encoding
dow_sin = sin(2π × day_of_week / 7)
dow_cos = cos(2π × day_of_week / 7)

# 365-day cycle encoding
doy_sin = sin(2π × day_of_year / 365.25)
doy_cos = cos(2π × day_of_year / 365.25)

# Month cycle encoding
month_sin = sin(2π × month / 12)
month_cos = cos(2π × month / 12)

# Categorical time features
hour_category (Night/Morning/Afternoon/Evening)
is_workday, is_business_hours
season (Winter/Spring/Summer/Autumn)
hours_since_start, days_since_start
```

#### **2. Optimized Lag Features (6 key lags)**
```python
# Short-term hourly lags (1-6 hours)
temp_lag_1h, temp_lag_3h, temp_lag_6h

# Daily pattern lags (24-48 hours)
temp_lag_24h, temp_lag_48h

# Weekly pattern lag (168 hours = 1 week)
temp_lag_168h

# Applied to 5 core variables:
# temp, humidity, sealevelpressure, windspeed, solarradiation
# Total: 5 variables × 6 lags = 30 lag features
```

#### **3. Weather Interaction Features**
```python
# Temperature-based interactions
temp_humidity_ratio = temp / (humidity + 1)
temp_dewpoint_spread = temp - dew
heat_index = temp + 0.5 × (humidity - 50)

# Wind-related interactions
wind_chill_factor = temp - 0.7 × windspeed
wind_pressure_gradient = windspeed × sealevelpressure / 1000
wind_north = windspeed × cos(wind_direction)
wind_east = windspeed × sin(wind_direction)

# Solar interactions
solar_efficiency = temp / (solarradiation + 1)
cloud_solar_interaction = cloudcover × solarradiation / 100

# Atmospheric stability
stability_index = (temp - dew) / max(windspeed, 0.1)
convection_potential = temp × humidity / 100

# Comfort indices
comfort_index, thermal_sensation
cooling_degree_hours, heating_degree_hours
```

#### **4. Trend & Change Detection Features**
```python
# Pressure tendencies (weather front detection)
pressure_tendency_1h = sealevelpressure.diff(1)
pressure_tendency_3h = sealevelpressure.diff(3)
pressure_tendency_6h = sealevelpressure.diff(6)

# Temperature trends
temp_change_1h, temp_change_3h, temp_change_6h
temp_acceleration_1h = temp_change_1h.diff()
temp_trend_6h (rising/falling/stable indicator)

# Daily comparisons
temp_daily_change = temp.diff(24)
temp_vs_yesterday = temp - temp.shift(24)

# Rapid weather changes (early warning)
temp_rapid_drop (>3°C drop in 3h)
temp_rapid_rise (>3°C rise in 3h)
pressure_rapid_drop, pressure_rapid_rise
weather_stability = combined change metric
```

#### **5. Daily Aggregation (91 Final Features)**
```python
# Hourly features aggregated to daily using:
# - mean: Average diurnal patterns
# - min: Daily minimums
# - max: Daily maximums
# - std: Intraday variability
# - sum: Total daily values (e.g., solar energy)

# Temperature statistics (PRIMARY TARGET)
temp_mean, temp_min, temp_max, temp_std

# Core weather daily aggregations
humidity_mean, humidity_min, humidity_max, humidity_std
sealevelpressure_mean, sealevelpressure_min, sealevelpressure_max, sealevelpressure_std
windspeed_mean, windspeed_max
solarradiation_sum, solarradiation_max
cloudcover_mean, visibility_mean, visibility_min

# Lag features (daily averages of hourly lags)
# Temporal features (daily averages of cyclical encodings)
# Interaction features (daily averages)
# Trend features (daily averages)

# Result: 91 features ready for 5-day forecasting
```

**⚠️ Data Leakage Prevention:**
- ❌ **NO rolling statistics** computed before train/test split
- ❌ **NO seasonal anomalies** using full dataset statistics
- ✅ **Only lag features** (past information)
- ✅ **Only current-time calculations** (physics-based interactions)
- ✅ **Safe aggregations** (hourly → daily without future information)

### **Step 8.4: Hourly Model Training**
📓 **Notebook**: `03_model_training_hourly.ipynb`

**Training Approach: Hourly → Daily 5-Day Forecasting**:
```python
# Step 1: Aggregate hourly data to daily with rich statistics
df_daily = aggregate_hourly_to_daily_features(df_hourly)
# Result: 3,649 daily observations with 91 features

# Step 2: Create 5-day ahead targets
df_daily = create_5day_targets(df_daily, target_base='temp_mean')
# Creates: temp_target_day1 through temp_target_day5

# Step 3: Train separate model for each forecast day
forecast_horizons = [1, 2, 3, 4, 5]
models = {}
for day in forecast_horizons:
    # Optuna hyperparameter optimization (15 trials per model)
    best_model = optimize_model(
        X_train, y_train_day[day],
        model_type='XGBoost',  # Also tested: LightGBM, RandomForest, GradientBoosting
        n_trials=15
    )
    models[f'day_{day}'] = best_model
```

**Model Architecture**:
- **5 separate models** (one for each forecast day 1-5)
- **XGBoost** selected as best performer across all horizons
- **Optuna TPE Sampler** for hyperparameter tuning (15 trials each)
- **RobustScaler** for feature scaling
- **Total models tested**: 4 algorithms × 5 days = 20 model-horizon combinations

**Time Series Validation Strategy**:
```python
# Temporal split (NO random shuffling)
# Train: 2015-09-27 to 2022-09-23 (2,554 days, 70%)
# Val:   2022-09-23 to 2023-09-23 (365 days, 10%)
# Test:  2023-09-23 to 2025-09-22 (730 days, 20%)

# Ensures:
# ✅ No data leakage across time boundaries
# ✅ Future predictions validated on truly unseen data
# ✅ Realistic performance assessment
```

### **Step 8.5: Hourly Model Monitoring & Retraining Strategy**
📓 **Notebook**: `04_model_monitoring_retraining.ipynb`

**Model Monitoring System Components**:

#### **1. Performance Tracking**
```python
class EnhancedModelMonitor:
    """
    Tracks R² and RMSE degradation over time periods.
    Based on XGBoost degradation of 18.2% from Day 1 to Day 5.
    """
    def __init__(self, model, feature_columns, baseline_performance):
        self.baseline_performance = baseline_performance
        self.thresholds = {
            'r2_minimal': 2.0,     # <2% degradation = stable
            'r2_low': 5.0,         # 2-5% = early warning
            'r2_medium': 8.0,      # 5-8% = moderate concern
            'r2_high': 10.0,       # 8-10% = high priority
            'r2_critical': 12.0,   # 10-12% = critical
            'r2_emergency': 15.0,  # ≥15% = emergency
        }
```

#### **2. Retraining Thresholds (Based on Notebook 03)**

From notebook 03, we observed XGBoost degrades **18.2% in R² from Day 1 to Day 5**:

| R² Degradation | Status | Action Required | Timeline |
|----------------|--------|----------------|----------|
| **< 2%** | ✅ STABLE | Routine monitoring | 3 months |
| **2-5%** | 🟢 MINIMAL | Monitor closely | 7 days |
| **5-8%** | 🟠 LOW | Plan retraining | 3 days |
| **8-10%** | 🟡 MEDIUM | Retrain within 2 days | 2 days |
| **10-12%** | ⚠️ HIGH | Retrain within 24 hours | 1 day |
| **12-15%** | 🔴 CRITICAL | Retrain immediately | 0 days |
| **≥ 15%** | 🚨 EMERGENCY | Stop using model | 0 days |

#### **3. Monitoring Metrics**

**Primary Metrics**:
- **R² Score**: Model explanatory power (0 to 1 scale)
  - R² = 0.90 means model explains 90% of temperature patterns
  - Higher R² = better understanding of patterns
- **RMSE (Root Mean Square Error)**: Average prediction error in °C
  - RMSE < 2.0°C = Excellent accuracy
  - RMSE 2.0-3.0°C = Good accuracy
  - RMSE > 4.0°C = Poor accuracy
- **Degradation Percentage**: Performance decline from baseline
  - Formula: `((Current - Baseline) / Baseline) × 100`

**Secondary Metrics**:
- **MAE (Mean Absolute Error)**: Average absolute error
- **MAPE (Mean Absolute Percentage Error)**: Percentage-based error
- **Time-series Trends**: Performance changes across monitoring periods

#### **4. Automated Decision Logic**

```python
def should_retrain(self, performance_data):
    """
    Determines if retraining is needed based on R² degradation.
    Returns urgency level and recommended timeline.
    """
    r2_degradation_pct = abs(performance_data['r2_degradation_pct'])
    
    if r2_degradation_pct >= 15.0:
        return {
            'urgency': 'EMERGENCY',
            'days': 0,
            'action': 'STOP USING MODEL - RETRAIN IMMEDIATELY'
        }
    elif r2_degradation_pct >= 12.0:
        return {
            'urgency': 'CRITICAL',
            'days': 0,
            'action': 'RETRAIN IMMEDIATELY'
        }
    # ... (see notebook for full decision tree)
```

#### **5. Visual Dashboard**

The monitoring system provides an interactive dashboard with:
- **R² Degradation Gauge**: Visual indicator of model health (0-20% scale)
- **R² Score Over Time**: Trend chart with baseline and warning thresholds
- **RMSE Over Time**: Error metric trends across monitoring periods
- **Summary Table**: Current status, actions, urgency, and deadlines

#### **6. Business Impact Assessment**

| Degradation Level | Risk | Business Impact |
|-------------------|------|----------------|
| **≥ 12%** | CRITICAL | Model severely compromised |
| **10-12%** | HIGH | Reliability significantly compromised |
| **8-10%** | MEDIUM | Some quality loss detected |
| **5-8%** | LOW | Minor performance decline |
| **< 5%** | MINIMAL | Normal operational variance |

#### **7. Hourly-Specific Monitoring**

**Additional Metrics for Hourly Models**:
- **Hour-of-day accuracy**: Performance by time of day
- **Seasonal hourly patterns**: Monthly × hourly performance matrix
- **Forecast horizon degradation**: Day 1 vs Day 5 accuracy
- **Diurnal error patterns**: When does the model struggle?
- **Rapid change detection**: Performance during weather transitions

---

## 🔍 Understanding Monitoring Metrics

### **What is R² (R-squared)?**
**R²** measures how well the model explains temperature variations (scale: 0 to 1).

- **R² = 0.90**: Model explains 90% of temperature patterns (Excellent)
- **R² = 0.75**: Model explains 75% of temperature patterns (Good)
- **R² < 0.60**: Model explains less than 60% (Poor)

**Why it matters**: Higher R² means the model understands the patterns better.

### **What is RMSE (Root Mean Square Error)?**
**RMSE** measures the average prediction error in degrees Celsius.

- **RMSE < 2.0°C**: Excellent accuracy
- **RMSE 2.0-3.0°C**: Good accuracy
- **RMSE > 4.0°C**: Poor accuracy

**Why it matters**: Lower RMSE means predictions are closer to actual temperatures.

### **What is Performance Degradation?**
**Performance degradation** shows how much worse the model performs compared to its baseline (when first trained).

**Calculation**: `((Current Metric - Baseline Metric) / Baseline Metric) × 100`

**Example**: 
- Baseline R² = 0.90, Current R² = 0.81
- Degradation = ((0.81 - 0.90) / 0.90) × 100 = **-10%**

### **When Should We Retrain?**

Based on **notebook 03 findings** (XGBoost degrades 18.2% from Day 1 to Day 5):

1. **< 5% degradation**: Model is stable ✅
2. **5-10% degradation**: Early warning - plan retraining 💛
3. **10-12% degradation**: High priority - urgent action ⚠️
4. **≥ 12% degradation**: Critical - immediate retraining 🚨

### **Why These Metrics Matter**

- **Reliability**: Weather predictions affect safety and planning decisions
- **Accuracy**: Even 1-2°C errors can impact agriculture, energy, and transportation
- **Timeliness**: Weather patterns change seasonally - regular monitoring catches drift early
- **Business Impact**: Poor forecasts lead to inefficient resource allocation

---

## 📊 Production Monitoring Dashboard

### **Streamlit App Monitoring Features**

The `streamlit_app_hourly.py` includes a comprehensive **Monitoring & Alerts** tab:

#### **1. Model Health Status**
Displays 7 key health indicators:
- Model Accuracy (R²)
- Model Degradation
- Data Drift
- Prediction Latency
- Last Retrain
- Data Quality
- Feature Stability

#### **2. Recent Alerts**
Shows timestamped alerts with severity levels:
- ℹ️ **Info**: Routine operational updates
- ⚠️ **Warning**: Early degradation detected
- ✅ **Success**: Successful operations (retraining, validation)

#### **3. Performance Charts**
Interactive visualizations:
- **Accuracy Over Time** (30-day rolling)
- **RMSE Over Time** (30-day rolling)
- Target lines showing baseline performance
- Warning thresholds (±10% and ±20%)

#### **4. Alert Configuration**
Configurable thresholds:
- Accuracy Threshold: 80-98%
- RMSE Threshold: 0.5-5.0°C
- Data Drift Threshold: 1-10%

---

## 🚀 Getting Started with Monitoring

### **Step 1: Initial Setup**
```python
from src.hourly.model_utils_hourly import EnhancedModelMonitor

# Load trained model and baseline
trained_model = joblib.load('models/hourly_trained/best_model_day_1_xgboost.joblib')
baseline_performance = {
    'rmse': 1.295,  # From notebook 03
    'r2': 0.9373
}

# Initialize monitor
monitor = EnhancedModelMonitor(trained_model, feature_columns, baseline_performance)
```

### **Step 2: Evaluate Performance**
```python
# Evaluate on new data
period_info = {
    'period': 'Week 1',
    'sample_count': 7,
    'start_date': '2025-10-22',
    'end_date': '2025-10-29'
}

performance = monitor.evaluate_performance(X_new, y_new, period_info)
print(f"Current R²: {performance['r2']:.4f}")
print(f"Degradation: {performance['r2_degradation_pct']:.1f}%")
```

### **Step 3: Check Retraining Decision**
```python
decision = monitor.should_retrain(performance)

if decision['should_retrain']:
    print(f"⚠️ {decision['severity'].upper()} - {decision['timeline']['action']}")
    print(f"Deadline: {decision['timeline']['deadline']}")
else:
    print("✅ Performance acceptable - continue monitoring")
```

### **Step 4: Launch Monitoring Dashboard**
```bash
# Run the Streamlit app with monitoring tab
streamlit run app/streamlit_app_hourly.py --server.port 8502
```

Then navigate to the **🚨 Monitoring & Alerts** tab to view:
- Real-time model health status
- Performance trends over time
- Alert history and recommendations

---

#### **8. Monitoring Schedule Recommendations**

| Deployment Stage | Monitoring Frequency | Alert Threshold |
|------------------|---------------------|-----------------|
| **First Month** | Daily | 5% degradation |
| **Months 2-6** | Weekly | 8% degradation |
| **Production (6+ months)** | Bi-weekly | 10% degradation |
| **Stable System** | Monthly | 12% degradation |

#### **9. Retraining Decision Framework**

**When to Retrain**:
1. **R² degrades ≥ 12%** from baseline
2. **RMSE increases ≥ 20%** from baseline
3. **Consecutive alerts** (2+ periods in a row)
4. **Seasonal changes** (every 3-4 months)
5. **Data drift detected** (distribution changes)
6. **New weather patterns** (climate events)

**What to Check Before Retraining**:
- Investigate root cause of degradation
- Check data quality and completeness
- Verify feature stability
- Assess need for new features
- Review seasonal effects
- Consider ensemble approach

**Retraining Best Practices**:
- Use latest 70% of available data
- Maintain temporal split (no data leakage)
- Re-run Optuna hyperparameter optimization
- Validate on held-out test set
- Compare new model vs old model
- A/B test before full deployment

---

### **Step 9: ONNX Deployment**
📓 **Notebook**: `05_onnx_deployment_hourly.ipynb`

This section documents the deployment optimization step where all trained tree-based models (Hourly workflow) are exported to the ONNX (Open Neural Network Exchange) format for fast, lightweight inference in production.

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
**Hourly-trained models (using hourly data → daily forecast)**
Five models of different types:

| Day |	Best Model | Framework | Export Method |
| 1 | XGBoost | booster | onnxmltools.convert_xgboost |
| 2 | LightGBM | booster_ | onnxmltools.convert_lightgbm |
| 3 | LightGBM | booster_ | onnxmltools.convert_lightgbm |
| 4 | GradientBoosting | sklearn | skl2onnx.convert_sklearn |
| 5 | XGBoost |	booster | onnxmltools.convert_xgboost |

All exported to:
```bash
models/hourly_trained/onnx_exports/day_X_model.onnx
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
- Rescaling
- Inference testing
- Parity validation

---

## 🎯 Performance Results with Hourly-Aggregated Data

### **5-Day Forecast Accuracy (XGBoost - Best Performer)**

| Forecast Day | R² Score | RMSE (°C) | MAE (°C) | MAPE (%) | Performance |
|--------------|----------|-----------|----------|----------|-------------|
| **Day 1** | **0.9369** | **1.299** | **1.000** | **4.35** | 🥇 Excellent |
| **Day 2** | **0.8521** | **1.990** | **1.550** | **6.95** | 🥈 Very Good |
| **Day 3** | **0.8074** | **2.271** | **1.814** | **8.27** | 🥉 Good |
| **Day 4** | **0.7799** | **2.428** | **1.938** | **8.93** | ✅ Solid |
| **Day 5** | **0.7701** | **2.482** | **1.962** | **9.03** | ✅ Solid |

### **Model Comparison Across All Horizons**

| Model | Avg R² | Avg RMSE | Day 1 R² | Day 5 R² | Degradation |
|-------|--------|----------|----------|----------|-------------|
| **XGBoost** | 0.8293 | 2.094°C | 0.9369 | 0.7701 | 17.81% |
| **LightGBM** | 0.8220 | 2.116°C | 0.9355 | 0.7593 | 18.83% |
| **Gradient Boosting** | 0.8170 | 2.127°C | 0.9359 | 0.7640 | 18.36% |
| **Random Forest** | 0.8151 | 2.160°C | 0.9309 | 0.7531 | 19.11% |

### **Comparison: Daily vs Hourly-Aggregated Approach**

| Metric | Daily Model (Daily Raw) | Hourly Model (Hourly → Daily) | Improvement |
|--------|-------------------------|-------------------------------|-------------|
| **Day 1 R²** | 0.9119 (XGBoost) | **0.9369 (XGB)** | +2.7% |
| **Day 1 RMSE** | 1.418°C (XGBoost) | **1.299°C (XGB)** | **8.4% better** |
| **Day 5 R²** | **0.8032 (GB)** | 0.7701 (XGB) | -4.1% |
| **Day 5 RMSE** | **2.112°C (GB)** | 2.482°C (XGB) | -17.5% |
| **Avg R²** | 0.8276 (XGB) | 0.8293 (XGB) | **+0.2% better** |
| **Features** | 100 | 91 | 9% fewer |

### **Key Performance Insights**

1. **Best Day 1 Accuracy**: XGBoost achieves **93.7% R²** with **1.299°C RMSE** (8.4% better than daily model)
2. **Hourly Granularity Advantage**: Capturing diurnal patterns improves short-term forecasts significantly
3. **Feature Efficiency**: 91 features (vs 100 in daily model) achieve slightly better overall performance
4. **Degradation**: 17.81% from Day 1 to Day 5 (vs daily model's 13.5%)
5. **Trade-off**: Better 1-2 day forecasts, slightly worse 4-5 day forecasts compared to daily raw data

### **Model Selection Strategy**

- **For 1-2 day forecasts**: Use **Hourly → Daily XGBoost** (best short-term accuracy)
- **For 3-5 day forecasts**: Use **Daily Raw Random Forest** (better long-term stability)
- **For production**: **Hourly XGBoost** provides excellent balance with fewer features
- **For ensemble**: Combine both approaches (hourly + daily) for robust predictions

### **Business Applications**

- **Energy Management**: Superior 1-day forecasts enable better HVAC scheduling (1.299°C RMSE)
- **Agricultural Planning**: Improved short-term forecasts for irrigation and harvesting decisions
- **Event Planning**: Reliable 2-day ahead predictions (R² 0.852) for outdoor activities
- **Transportation**: Better next-day road condition and safety forecasts

---

## 🚀 Getting Started with Hourly Analysis

### **Prerequisites**
```bash
# Install additional dependencies for hourly analysis
pip install optuna scikit-learn>=1.3.0 xgboost lightgbm
```

### **Quick Start**
```bash
# 1. Explore hourly data
jupyter notebook notebooks_hourly/00_data_exploration_hourly.ipynb

# 2. Process hourly features  
jupyter notebook notebooks_hourly/02_feature_engineering_hourly.ipynb

# 3. Train hourly models
jupyter notebook notebooks_hourly/03_model_training_hourly.ipynb

# 4. Launch hourly app
python app/run_hourly_app.py
```

### **Data Loading Example**
```python
from src.hourly.data_utils_hourly import load_raw_hourly_data
from src.hourly.feature_utils_hourly import HourlyFeatureEngineering

# Load hourly data
df_hourly = load_raw_hourly_data()
print(f"Loaded {len(df_hourly)} hourly records")

# Create hourly features
feature_engineer = HourlyFeatureEngineering()
df_features = feature_engineer.create_temporal_features(df_hourly)
df_features = feature_engineer.create_lag_features(df_features)
print(f"Created {len(df_features.columns)} features")
```

---

## 📊 Performance Comparison: Daily vs Hourly Approach

### **Model Complexity - Actual Implementation**
- **Daily Model (Raw Daily Data)**: 130 features, 3,625 training samples
- **Hourly Model (Hourly → Daily Aggregation)**: 91 features, 3,649 training samples

### **Computational Requirements**
- **Training Time**: Similar (~2-3 minutes with Optuna optimization)
- **Memory Usage**: Comparable (both use daily-level aggregated data)
- **Storage**: Hourly model slightly larger (5 models × 5 days)
- **Preprocessing**: Hourly approach adds aggregation step (~10 seconds)

### **Prediction Quality - Real Results**
- **Day 1**: 🏆 **Hourly approach 12% better** (1.28°C vs 1.46°C RMSE)
- **Day 2-3**: ✅ **Hourly approach 5-8% better** (captures short-term patterns)
- **Day 4-5**: ⚠️ **Daily approach 8-14% better** (benefits from direct daily signals)
- **Overall**: 🤝 **Similar average performance** (0.8267 vs 0.8289 R²)

---

## 🛠️ Technical Implementation Details

### **Feature Engineering Pipeline**
```python
class HourlyFeatureEngineering:
    """Comprehensive hourly feature engineering."""
    
    def create_all_features(self, df):
        df = self.create_temporal_features(df)      # Time-based features
        df = self.create_lag_features(df)           # Historical features  
        df = self.create_rolling_features(df)       # Statistical features
        df = self.create_change_features(df)        # Change indicators
        df = self.create_interaction_features(df)   # Feature interactions
        df = self.create_diurnal_patterns(df)       # Daily cycle features
        return df
```

### **Multi-Horizon Training**
```python
# Train separate models for different forecast horizons
horizon_models = {}
for horizon in [1, 6, 12, 24, 48, 72, 120]:  # hours
    X_train, y_train = prepare_data(df, horizon)
    model = optimize_model_for_horizon(X_train, y_train, horizon)
    horizon_models[horizon] = model
```

### **Hourly Validation Strategy**
```python
# Time-aware cross-validation
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5, test_size=24*7)  # 1 week test sets
for train_idx, test_idx in tscv.split(X):
    # Ensure no data leakage across time boundaries
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
```

---

## 📈 Success Metrics & KPIs

### **Primary Metrics**
- **RMSE by Hour**: Model accuracy for each hour of day
- **MAE by Season**: Seasonal performance assessment  
- **R² by Forecast Horizon**: Predictive power vs. distance
- **Directional Accuracy**: Trend prediction success rate

### **Secondary Metrics**
- **Peak Temperature Timing**: Daily max/min prediction accuracy
- **Rapid Change Detection**: Alert system performance
- **Comfort Index Forecasting**: Human comfort predictions
- **Energy Load Correlation**: HVAC demand prediction accuracy

---

## 🎉 Actual Implementation Achievements

This hourly analysis system has successfully delivered:

1. ✅ **Hourly → Daily Aggregation System**: 87,698 hourly records → 3,649 daily observations
2. ✅ **Efficient Feature Engineering**: 91 optimized features (30% fewer than daily model)
3. ✅ **5-Day Multi-Horizon Models**: Separate XGBoost models for each forecast day (1-5)
4. ✅ **Superior Short-Term Accuracy**: 93.6% R² for Day 1 (12% better RMSE than daily model)
5. ✅ **Production-Ready Application**: Streamlit app with multi-tab interface for hourly forecasting
6. ✅ **Comprehensive Performance Benchmarks**: XGBoost outperforms 3 other algorithms across all horizons
7. ✅ **Data Leakage Prevention**: Leak-free features with temporal split validation

---

## 🔗 Integration with Existing Work

This hourly implementation **complements** rather than **replaces** the daily forecasting system:

### **When to Use Each System**

| Scenario | Recommended System | Reason |
|----------|-------------------|---------|
| **Next-day planning** | Hourly → Daily XGBoost | Best Day 1 accuracy (93.6% R², 1.28°C RMSE) |
| **2-3 day forecasts** | Hourly → Daily XGBoost | 5-8% better short-term RMSE |
| **4-5 day forecasts** | Daily Raw Random Forest | 8-14% better long-term stability |
| **Real-time operations** | Hourly → Daily XGBoost | Captures recent diurnal patterns |
| **Long-term planning** | Daily Raw Random Forest | Better with direct daily signals |
| **Production deployment** | Hourly → Daily XGBoost | Fewer features (91 vs 130), excellent balance |

### **Combined Ensemble Approach** (Recommended for Critical Applications)

```python
# Weighted ensemble for optimal accuracy
def ensemble_predict(date, horizon):
    if horizon <= 2:
        # Short-term: 70% hourly, 30% daily
        pred = 0.7 * hourly_model.predict(date) + 0.3 * daily_model.predict(date)
    else:
        # Long-term: 40% hourly, 60% daily
        pred = 0.4 * hourly_model.predict(date) + 0.6 * daily_model.predict(date)
    return pred
```

**Expected Ensemble Performance**:
- Day 1: R² ~0.94, RMSE ~1.26°C (best of both worlds)
- Day 5: R² ~0.79, RMSE ~2.30°C (balanced approach)

---

## � Implementation Summary

### **Final Project Statistics**

| Aspect | Specification |
|--------|------------|
| **Total Hourly Records** | 87,698 observations (2015-2025) |
| **Aggregated Daily Records** | 3,649 daily observations |
| **Training Period** | 2015-09-27 to 2022-09-23 (2,554 days, 70%) |
| **Validation Period** | 2022-09-23 to 2023-09-23 (328 days, 9%) |
| **Test Period** | 2023-09-23 to 2025-09-22 (693 days, 19%) |
| **Total Features** | 91 (from hourly aggregations) |
| **Feature Categories** | Temporal, Lag, Rolling, Cyclical, Interactions |
| **Forecast Horizons** | 5 days (separate model per day) |
| **Best Algorithm** | XGBoost (wins all 5 horizons) |
| **Optimization Method** | Optuna TPE (15 trials per model) |
| **Total Models Trained** | 20 (4 algorithms × 5 horizons) |

### **Key Technical Achievements**

1. **🏆 Best-in-Class Day 1 Accuracy**: 93.7% R² (1.299°C RMSE) - 8.4% better than daily model
2. **⚡ Feature Efficiency**: 91 features vs 100 in daily model (9% reduction) with better performance
3. **🔒 Data Leakage Prevention**: Strict temporal validation, no future information in features
4. **🎯 Consistent Performance**: XGBoost dominates across all 5 forecast horizons
5. **📊 Realistic Validation**: 693-day test set (19% of data) for robust evaluation
6. **🌡️ Low Error Rates**: Day 1 MAPE 4.35%, Day 5 MAPE 9.03% (acceptable for 5-day forecasts)

### **Model Performance Summary**

```
╔═══════════════════════════════════════════════════════════════════════╗
║                    XGBoost 5-Day Forecast Performance                 ║
╠════════════╦════════╦═══════════╦══════════╦══════════╦══════════════╣
║ Forecast   ║   R²   ║ RMSE (°C) ║ MAE (°C) ║ MAPE (%) ║ Assessment   ║
╠════════════╬════════╬═══════════╬══════════╬══════════╬══════════════╣
║ Day 1      ║ 0.9369 ║   1.299   ║   1.000  ║   4.35   ║ 🥇 Excellent ║
║ Day 2      ║ 0.8521 ║   1.990   ║   1.550  ║   6.95   ║ 🥈 Very Good ║
║ Day 3      ║ 0.8074 ║   2.271   ║   1.814  ║   8.27   ║ 🥉 Good      ║
║ Day 4      ║ 0.7799 ║   2.428   ║   1.938  ║   8.93   ║ ✅ Solid     ║
║ Day 5      ║ 0.7701 ║   2.482   ║   1.962  ║   9.03   ║ ✅ Solid     ║
╠════════════╬════════╬═══════════╬══════════╬══════════╬══════════════╣
║ AVERAGE    ║ 0.8293 ║   2.094   ║   1.653  ║   7.51   ║ 🏆 Very Good ║
╚════════════╩════════╩═══════════╩══════════╩══════════╩══════════════╝
```

### **Comparison with Daily Model**

| Metric | Daily Model (Raw) | Hourly Model (Aggregated) | Winner |
|--------|------------------|---------------------------|--------|
| Best Algorithm | XGBoost | XGBoost | - |
| Day 1 R² | 0.9119 | **0.9369** (+2.7%) | 🏆 Hourly |
| Day 1 RMSE | 1.418°C | **1.299°C** (-8.4%) | 🏆 Hourly |
| Day 5 R² | **0.8032** | 0.7701 (-4.1%) | 🏆 Daily |
| Day 5 RMSE | **2.112°C** | 2.482°C (+17.5%) | 🏆 Daily |
| Average R² | 0.8276 | **0.8293** (+0.2%) | 🏆 Hourly |
| Features | 100 | **91** (-9%) | 🏆 Hourly |
| Training Time | ~2 min | ~2 min | 🤝 Tie |

**Conclusion**: Hourly aggregation approach excels for 1-2 day forecasts with fewer features, while daily model is better for 4-5 day predictions. **Ensemble approach recommended** for production.

---

## �📚 References & Further Reading

- **Time Series Forecasting**: Hyndman & Athanasopoulos - "Forecasting: Principles and Practice"
- **Weather Pattern Analysis**: WMO Guidelines on Climate Data and Information
- **Hourly Energy Forecasting**: ASHRAE Standards for Building Energy Modeling
- **Diurnal Temperature Modeling**: Meteorological research on urban heat islands

---

*This README provides comprehensive guidance for implementing Step 8 of the Hanoi Weather Forecasting project. For questions or support, refer to the notebook implementations and utility modules.*
