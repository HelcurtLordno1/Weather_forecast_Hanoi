# 🌡️ Hanoi Temperature Forecasting - Streamlit Application

A comprehensive web application for 5-day ahead temperature forecasting in Hanoi, Vietnam using advanced machine learning models.

## 🚀 Quick Start

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


## 📋 Prerequisites

Before running the application, ensure you have:

1. **Completed Model Training**: Run the `03_model_training_comprehensive.ipynb` notebook to generate trained models
2. **Python Environment**: Python 3.8+ with required packages
3. **Data Files**: Weather data in the `data/raw/` directory

## 🎯 Features

### 🔮 Interactive Prediction
- Input current weather conditions
- Generate 1-5 day temperature forecasts
- View confidence intervals
- Interactive visualizations

### 📊 Historical Data Analysis
- Explore 10+ years of Hanoi weather data
- Interactive temperature trend charts
- Statistical summaries and insights
- Date range filtering

### 🎯 Model Performance Dashboard
- View model accuracy metrics (RMSE, MAE, R²)
- Compare validation vs test performance
- Model training details and metadata
- Feature engineering insights

### 📜 Prediction History
- Track all predictions made during the session
- Compare input conditions and results
- Export prediction history

### ℹ️ Comprehensive Documentation
- Model architecture details
- Technical specifications
- Use case examples
- Performance benchmarks

## 🛠️ Application Structure

```
├── app_streamlit.py          # Main Streamlit application
├── run_app.py               # Application launcher script
├── requirements_streamlit.txt # Python dependencies
├── .streamlit/
│   └── config.toml          # Streamlit configuration
└── models/trained/          # Trained ML models (generated)
    ├── best_model_*.joblib
    ├── feature_columns.joblib
    └── model_metadata.json
```

## 📊 Application Interface

### Main Tabs

1. **🔮 Prediction Tab**
   - Input current weather conditions
   - Generate multi-day forecasts
   - Interactive forecast visualization
   - Feature engineering demonstration

2. **📊 Historical Data Tab**
   - Temperature trend visualization
   - Date range selection
   - Statistical summaries
   - Multi-variable weather plots

3. **🎯 Model Performance Tab**
   - Validation and test metrics
   - Model configuration details
   - Training data information
   - Performance benchmarks

4. **📜 Prediction History Tab**
   - Session prediction tracking
   - Input-output comparison
   - Historical analysis

5. **ℹ️ About Tab**
   - Project documentation
   - Technical specifications
   - Use case examples

### Sidebar Controls

- **Model Information**: Current model details and performance
- **Prediction Settings**: Forecast horizon and confidence intervals
- **Data Upload**: Upload custom weather data files

## 🎨 User Interface Features

### Modern Design
- Responsive layout optimized for different screen sizes
- Professional color scheme with custom CSS styling
- Interactive plotly visualizations
- Intuitive navigation with tabbed interface

### Real-time Predictions
- Instant temperature forecasting
- Dynamic confidence interval calculation
- Interactive forecast charts
- Detailed prediction tables

### Data Visualization
- Historical temperature trends
- Multi-variable weather plots
- Statistical distribution charts
- Performance metric displays

## 🔧 Customization

### Theme Configuration
Edit `.streamlit/config.toml` to customize:
- Primary and secondary colors
- Background colors
- Text colors
- Server settings

### Model Integration
The application automatically loads trained models from:
- `models/trained/best_model_*.joblib` - Trained ML model
- `models/trained/feature_columns.joblib` - Feature specifications
- `models/trained/model_metadata.json` - Model information

### Data Sources
- Default: `data/raw/Hanoi-Daily-10-years.csv`
- Upload custom CSV files via the sidebar
- Automatic data validation and processing

## 📈 Performance

### Typical Response Times
- Prediction generation: < 1 second
- Chart rendering: < 2 seconds
- Data loading: < 3 seconds
- Model loading: < 5 seconds (first time)

### Resource Usage
- Memory: ~100-200 MB
- CPU: Low (single-threaded)
- Storage: Models ~10-50 MB

## 🐛 Troubleshooting

### Common Issues

1. **Model Not Found Error**
   ```
   Solution: Run the model training notebook first
   File: 03_model_training_comprehensive.ipynb
   ```

2. **Data Loading Error**
   ```
   Solution: Check data file exists in data/raw/
   Expected: Hanoi-Daily-10-years.csv
   ```

3. **Import Errors**
   ```
   Solution: Install required packages
   Command: pip install -r requirements_streamlit.txt
   ```

4. **Port Already in Use**
   ```
   Solution: Use different port
   Command: streamlit run app_streamlit.py --server.port 8502
   ```

### Debug Mode
Add `--logger.level debug` to the streamlit command for detailed logging:
```bash
streamlit run app_streamlit.py --logger.level debug
```

## 📚 Additional Resources

### Documentation
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Python Documentation](https://plotly.com/python/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

### Model Training
- Complete the Jupyter notebooks in order:
  1. `00_data_exploration_comprehensive.ipynb`
  2. `01_data_processing_comprehensive.ipynb`
  3. `02_feature_engineering_comprehensive.ipynb`
  4. `03_model_training_comprehensive.ipynb`

### Data Requirements
- CSV format with datetime column
- Temperature and weather feature columns
- Minimum 1 year of historical data recommended
- Daily frequency preferred

## 🤝 Contributing

To extend the application:

1. **Add New Features**: Modify `app_streamlit.py`
2. **Improve UI**: Edit CSS styles and layouts
3. **Add Models**: Integrate additional ML models
4. **Enhance Visualizations**: Add new chart types

## 📄 License

This project is part of a Machine Learning course and is intended for educational purposes.

---

**🌡️ Ready to forecast Hanoi's weather? Run the application and start predicting!**