# 🌡️ Hanoi Weather Forecasting System

**Advanced ML-powered 5-day ahead temperature prediction for Hanoi, Vietnam**

An interactive machine learning application that provides accurate temperature forecasting using 10+ years of historical weather data. Built with state-of-the-art ML models and a professional Streamlit web interface.

## 🎯 Features

- **🤖 5 Optimized ML Models**: Random Forest, XGBoost, LightGBM, AdaBoost, Gradient Boosting
- **🔮 5-Day Forecasting**: Predict temperatures up to 5 days ahead with confidence intervals
- **📊 79 Advanced Features**: Time series features, lag variables, rolling statistics, seasonal patterns
- **🌐 Interactive Web Interface**: Professional Streamlit dashboard with real-time predictions
- **📈 Performance Analytics**: Model comparison, accuracy metrics, prediction history
- **🎯 High Accuracy**: Best model achieves 2.17°C RMSE with 76.1% R² score

## � Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation & Running

1. **Clone/Download the project**
   ```bash
   cd weather_forcast_project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements_streamlit.txt
   ```

3. **Launch the application**
   ```bash
   # Option 1: Using launcher script
   python run_app.py
   
   # Option 2: Direct command
   streamlit run app_streamlit.py
   ```

4. **Open your browser**
   - Navigate to `http://localhost:8501`
   - Start making temperature predictions!

### For Developers
To explore the complete ML pipeline:
```bash
pip install -r requirements.txt
jupyter notebook
```
Run notebooks 00-03 in sequence to understand the model development process.

## 🌐 Streamlit Web Application

### Application Interface

The web application provides an intuitive interface with 5 main sections:

### 🔮 **Prediction Tab**
- Input current weather conditions (temperature, humidity, pressure, wind speed)
- Select forecast horizon (1-5 days ahead)
- Generate instant temperature predictions with confidence intervals
- Interactive Plotly charts showing forecast trends
- Real-time prediction updates (<1 second response)

### 📈 **Historical Data Tab** 
- Explore 10+ years of Hanoi weather data (2013-2024)
- Interactive temperature trend visualization
- Date range filtering and statistical summaries
- Weather pattern analysis and seasonal insights

### 🎯 **Model Performance Tab**
- Compare all 5 trained models (Random Forest, XGBoost, LightGBM, AdaBoost, Gradient Boosting)
- View performance metrics: RMSE, MAE, R², MAPE
- Training vs validation performance analysis
- Hyperparameter configurations and model details

### 📜 **Prediction History Tab**
- Track all predictions made during the session
- Input-output comparison tables
- Export prediction history for analysis
- Trend visualization of forecasting patterns

### ℹ️ **About Tab**
- Technical specifications and model architecture
- Feature engineering methodology (79 features)
- Performance benchmarks and accuracy metrics
- Use case examples and applications

## 🤖 Machine Learning Models

The system uses 5 optimized algorithms trained on 10+ years of Hanoi weather data:

| Model | RMSE (°C) | MAE (°C) | R² Score | Description |
|-------|-----------|----------|----------|-------------|
| **AdaBoost** ⭐ | **2.27** | **1.78** | **0.761** | Adaptive boosting - best performer |
| XGBoost | 2.34 | 1.85 | 0.745 | Gradient boosting with regularization |
| LightGBM | 2.41 | 1.91 | 0.728 | Fast gradient boosting |
| Random Forest | 2.48 | 1.96 | 0.712 | Ensemble of decision trees |
| Gradient Boosting | 2.55 | 2.02 | 0.695 | Traditional gradient descent |

### Key Features:
- **79 Engineered Features**: Temporal patterns, lag variables, rolling statistics
- **Hyperparameter Optimization**: Optuna Bayesian optimization (50-100 trials per model)
- **Time Series Validation**: Proper temporal splits to prevent data leakage
- **5-Day Forecasting**: Predicts temperature up to 5 days ahead

## � Usage

```bash
streamlit run app_streamlit.py
```

### **2. Navigate to:** `http://localhost:8501`

### **3. Use the 5-tab interface:**
- **🏠 Home**: Project overview and quick start
- **📊 EDA**: Explore weather data patterns  
- **🤖 Model Training**: Train and optimize ML models
- **🔮 Predictions**: Generate temperature forecasts
- **📈 Results**: View model performance metrics

## 🛠️ Technical Requirements

### **Dependencies:**
- Python 3.8+
- streamlit
- scikit-learn, xgboost, lightgbm
- pandas, numpy, plotly
### **System Requirements:**
- RAM: 8GB minimum
- Storage: 2GB for complete project
- CPU: Multi-core recommended

---

*This project provides a complete temperature forecasting system with professional web interface and optimized ML models.*


- **Hyperparameter Optimization:**
  - **Optuna Bayesian Optimization** with 50-100 trials per model
  - **Time Series Cross-Validation** to prevent data leakage
  - **Grid search** for optimal parameters

### **Stage 5: Model Evaluation** (`04_model_evaluation.ipynb`)
- **Performance Metrics:**
  - **RMSE**: Root Mean Square Error (primary metric)
  - **MAE**: Mean Absolute Error
  - **R²**: Coefficient of determination
  - **MAPE**: Mean Absolute Percentage Error

- **Validation Strategy:**
  - **Temporal Train/Validation/Test splits** (70%/15%/15%)
  - **No data leakage** - future data never used for training
  - **Model generalization testing** on unseen data

## � Model Performance Results

### **Champion Model: AdaBoost (Optimized)**
- **🎯 RMSE**: 2.17°C (Test), 2.27°C (Validation)
- **📊 R² Score**: 76.1% (Validation)
- **⚡ MAE**: 1.73°C (Test), 1.78°C (Validation)
- **📈 MAPE**: 5.77% (Test), 9.22% (Validation)

### **All Models Comparison:**
| Model | RMSE (°C) | MAE (°C) | R² Score | Training Time |
|-------|-----------|----------|----------|---------------|
| **AdaBoost** ⭐ | **2.27** | **1.78** | **0.761** | ~5 min |
| XGBoost | 2.34 | 1.85 | 0.745 | ~8 min |
| LightGBM | 2.41 | 1.91 | 0.728 | ~3 min |
| Random Forest | 2.48 | 1.96 | 0.712 | ~12 min |
| Gradient Boosting | 2.55 | 2.02 | 0.695 | ~15 min |

### **Key Achievements:**
- **🎯 High Accuracy**: Best model achieves <2.2°C prediction error
- **⚡ Fast Training**: All models trained in <15 minutes with optimization
- **🔄 Robust Validation**: Consistent performance across validation and test sets
- **📈 Feature Importance**: Temperature lags and seasonal patterns most predictive

## 💻 Usage Examples

### **1. Quick Prediction via Streamlit:**
1. Open the Streamlit app: `streamlit run app_streamlit.py`
2. Navigate to "🔮 Prediction" tab
3. Input current conditions:
   - Temperature: 25°C
   - Humidity: 70%
   - Pressure: 1013 hPa
4. Select forecast horizon (1-5 days)
5. Click "Generate Forecast" → Get instant predictions!

### **2. Programmatic Usage:**
```python
# Load trained model
import joblib
model = joblib.load('models/trained/best_model_adaboost_optimized.joblib')

# Prepare features (simplified example)
import pandas as pd
current_conditions = pd.DataFrame({
    'temp_lag_1': [24.5],  # Yesterday's temperature
    'temp_lag_7': [23.8],  # Last week's temperature  
    'month_sin': [0.5],    # Seasonal encoding
    'humidity': [65],      # Current humidity
    # ... (79 total features)
})

# Make prediction
forecast = model.predict(current_conditions)
print(f"5-day forecast: {forecast[0]:.1f}°C")
```

### **3. Batch Processing:**
```python
from src.data_utils import load_hanoi_weather_data
from src.feature_utils import prepare_features_for_modeling

# Load new data
df = load_hanoi_weather_data('path/to/new_data.csv')

# Engineer features
df_features = prepare_features_for_modeling(df, target_col='temp')

# Generate forecasts
forecasts = model.predict(df_features[feature_columns])
```

## �️ Technical Specifications

### **System Requirements:**
- **OS**: Windows 10+, macOS 10.14+, Linux Ubuntu 18.04+
- **Python**: 3.8+ (tested with 3.13)
- **RAM**: 8GB minimum, 16GB recommended for training
- **Storage**: 2GB for complete project
- **CPU**: Multi-core recommended for Optuna optimization

### **Dependencies:**
- **Core ML**: scikit-learn, xgboost, lightgbm, optuna
- **Data**: pandas, numpy, joblib
- **Visualization**: plotly, matplotlib, seaborn
- **Web App**: streamlit, altair
- **Utils**: datetime, json, os

### **Performance Benchmarks:**
- **Prediction Time**: <100ms per forecast
- **Model Loading**: <2 seconds
- **Streamlit Startup**: <10 seconds
- **Training Time**: 15-60 minutes (full pipeline)

## 🎯 Use Cases & Applications

### **Academic & Research:**
- **Time Series Forecasting** course projects
- **Machine Learning** pipeline demonstrations
- **Feature Engineering** technique showcase
- **Model Comparison** studies

### **Professional Applications:**
- **Weather Service Enhancement** - Improve local forecasting
- **Agricultural Planning** - Crop management based on temperature
- **Energy Management** - HVAC optimization and demand forecasting
- **Tourism Industry** - Activity planning and recommendations

### **Learning & Development:**
- **Streamlit Tutorial** - Learn web app development
- **ML Pipeline** - Understand end-to-end model development
- **Hyperparameter Tuning** - Explore Optuna optimization
- **Time Series Analysis** - Master temporal data handling

## 🧹 Project Cleanup Recommendations

### **Files to Keep (Essential):**
```
✅ KEEP THESE FILES:
├── app_streamlit.py              # Main application ⭐
├── run_app.py                    # App launcher
├── .streamlit/config.toml        # Configuration
├── data/raw/Hanoi-Daily-10-years.csv  # Core dataset
├── models/trained/               # All trained models
├── src/                          # Utility functions
├── notebooks/00-03*.ipynb       # Core ML pipeline
├── README.md                     # This guide
├── requirements_streamlit.txt    # Streamlit dependencies
└── requirements.txt              # Full dependencies
```


### **Commands to Clean Up:**
```bash
# Navigate to project directory
cd weather_forcast_project

# Remove test and redundant files
rm app_test.py
rm README_STREAMLIT.md

# Remove optional notebooks (keep if you want detailed analysis)
rm notebooks/04_model_evaluation.ipynb
rm notebooks/05_ui_design.ipynb  
rm notebooks/06_final_report.ipynb

# Remove generated outputs (they can be recreated)
rm -rf outputs/
rm -rf data/processed/

# Keep only essential files for clean distribution
```

## 🤝 Contributing & Customization

### **Adding New Models:**
1. Implement in `03_model_training_comprehensive.ipynb`
2. Add to the model comparison pipeline
3. Update Streamlit model loading logic

### **Extending Features:**
1. Add feature engineering in `02_feature_engineering_comprehensive.ipynb`
2. Update feature list in model training
3. Retrain models with new features

### **Customizing Streamlit UI:**
1. Edit `app_streamlit.py` for new components
2. Modify CSS in the styling section
3. Add new tabs or visualization types
4. Update the configuration in `.streamlit/config.toml`

## 📄 License & Acknowledgments

### **License:**
This project is created for educational purposes. Feel free to use, modify, and distribute for academic and research purposes.

### **Acknowledgments:**
- **Weather Data**: Historical Hanoi weather dataset providers
- **ML Libraries**: scikit-learn, XGBoost, LightGBM teams
- **Streamlit**: For the amazing web framework
- **Optuna**: For hyperparameter optimization
- **Python Community**: For the incredible ecosystem

## 📞 Support & Contact

### **Getting Help:**
- **📖 Documentation**: Check this README and notebook comments
- **🐛 Issues**: Create issues for bugs or feature requests
- **💡 Ideas**: Suggest improvements and new features
- **📧 Contact**: Reach out for collaboration or questions

### **Learning Resources:**
- **Streamlit**: [docs.streamlit.io](https://docs.streamlit.io)
- **Machine Learning**: [scikit-learn.org](https://scikit-learn.org)
- **Time Series**: [pandas time series documentation](https://pandas.pydata.org/docs/user_guide/timeseries.html)
- **Hyperparameter Optimization**: [optuna.org](https://optuna.org)

---

## 🎉 **Ready to Forecast the Future?**

### **Quick Start Commands:**
```bash
# 1. Install dependencies
pip install -r requirements_streamlit.txt

# 2. Launch the amazing Streamlit app
streamlit run app_streamlit.py

# 3. Open browser and start predicting! 🌡️
```

**Happy Forecasting! May your predictions be accurate and your temperatures comfortable! 🌡️✨**