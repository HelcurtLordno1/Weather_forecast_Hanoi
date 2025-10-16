"""
UI Application for Hanoi Temperature Forecasting

This module contains the user interface application using Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
import os
import sys
from typing import Dict, List, Any

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.model_utils import TemperatureForecastingModel
    from src.visualization import create_interactive_temperature_plot, create_prediction_dashboard
    from src.data_utils import load_raw_data
except ImportError:
    # Fallback imports for when running as standalone
    from model_utils import TemperatureForecastingModel
    from visualization import create_interactive_temperature_plot, create_prediction_dashboard
    from data_utils import load_raw_data


class TemperatureForecastingApp:
    """
    Main Streamlit application for temperature forecasting.
    """
    
    def __init__(self):
        self.model = None
        self.data = None
        self.setup_page_config()
    
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Hanoi Temperature Forecasting",
            page_icon="🌡️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def load_model(self, model_path: str):
        """Load trained model."""
        try:
            self.model = joblib.load(model_path)
            return True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def load_data(self, data_path: str):
        """Load data for visualization."""
        try:
            self.data = pd.read_csv(data_path)
            return True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def sidebar_controls(self):
        """Create sidebar controls."""
        st.sidebar.title("🌡️ Temperature Forecasting")
        st.sidebar.markdown("---")
        
        # Model selection
        st.sidebar.subheader("Model Configuration")
        model_options = ["Random Forest", "Gradient Boosting", "Linear Regression"]
        selected_model = st.sidebar.selectbox("Select Model", model_options)
        
        # Prediction settings
        st.sidebar.subheader("Prediction Settings")
        forecast_hours = st.sidebar.slider("Forecast Hours", 1, 168, 24)
        confidence_level = st.sidebar.selectbox("Confidence Level", [90, 95, 99])
        
        # Data upload
        st.sidebar.subheader("Data Upload")
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
        
        return selected_model, forecast_hours, confidence_level, uploaded_file
    
    def main_header(self):
        """Create main header."""
        st.title("🌡️ Hanoi Temperature Forecasting System")
        st.markdown("""
        This application predicts temperature in Hanoi using machine learning models.
        Use the sidebar to configure settings and upload data.
        """)
        st.markdown("---")
    
    def data_overview_section(self):
        """Create data overview section."""
        if self.data is not None:
            st.subheader("📊 Data Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", len(self.data))
            
            with col2:
                if 'temperature' in self.data.columns:
                    avg_temp = self.data['temperature'].mean()
                    st.metric("Average Temperature", f"{avg_temp:.1f}°C")
            
            with col3:
                if 'temperature' in self.data.columns:
                    max_temp = self.data['temperature'].max()
                    st.metric("Maximum Temperature", f"{max_temp:.1f}°C")
            
            with col4:
                if 'temperature' in self.data.columns:
                    min_temp = self.data['temperature'].min()
                    st.metric("Minimum Temperature", f"{min_temp:.1f}°C")
            
            # Data preview
            st.subheader("Data Preview")
            st.dataframe(self.data.head(10))
    
    def temperature_trends_section(self):
        """Create temperature trends visualization."""
        if self.data is not None:
            st.subheader("📈 Temperature Trends")
            
            # Assuming datetime and temperature columns exist
            datetime_col = 'datetime' if 'datetime' in self.data.columns else self.data.columns[0]
            temp_col = 'temperature' if 'temperature' in self.data.columns else self.data.columns[1]
            
            if datetime_col in self.data.columns and temp_col in self.data.columns:
                # Convert datetime if needed
                if not pd.api.types.is_datetime64_any_dtype(self.data[datetime_col]):
                    self.data[datetime_col] = pd.to_datetime(self.data[datetime_col])
                
                # Create interactive plot
                fig = create_interactive_temperature_plot(
                    self.data, datetime_col, temp_col, 
                    "Hanoi Temperature Trends"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def prediction_section(self, forecast_hours: int, confidence_level: int):
        """Create prediction section."""
        st.subheader("🔮 Temperature Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Parameters")
            
            # Manual input controls
            current_temp = st.number_input("Current Temperature (°C)", 
                                         value=25.0, min_value=-10.0, max_value=50.0)
            humidity = st.slider("Humidity (%)", 0, 100, 60)
            pressure = st.number_input("Pressure (hPa)", value=1013.25)
            wind_speed = st.number_input("Wind Speed (m/s)", value=5.0, min_value=0.0)
            
            # Time inputs
            prediction_date = st.date_input("Prediction Date", datetime.now().date())
            prediction_hour = st.slider("Hour", 0, 23, datetime.now().hour)
            
            predict_button = st.button("🔮 Generate Prediction", type="primary")
        
        with col2:
            st.subheader("Prediction Results")
            
            if predict_button:
                # Simulate prediction (replace with actual model prediction)
                predicted_temp = self.simulate_prediction(
                    current_temp, humidity, pressure, wind_speed
                )
                
                # Display prediction
                st.metric("Predicted Temperature", f"{predicted_temp:.1f}°C")
                
                # Generate forecast series
                forecast_data = self.generate_forecast_series(
                    predicted_temp, forecast_hours
                )
                
                # Plot forecast
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=forecast_data['datetime'],
                    y=forecast_data['temperature'],
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='red', width=3)
                ))
                
                fig.update_layout(
                    title=f"{forecast_hours}-Hour Temperature Forecast",
                    xaxis_title="Time",
                    yaxis_title="Temperature (°C)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def simulate_prediction(self, current_temp: float, humidity: float, 
                           pressure: float, wind_speed: float) -> float:
        """
        Simulate temperature prediction (replace with actual model).
        
        Args:
            current_temp: Current temperature
            humidity: Humidity percentage
            pressure: Atmospheric pressure
            wind_speed: Wind speed
            
        Returns:
            Predicted temperature
        """
        # Simple simulation - replace with actual model prediction
        variation = np.random.normal(0, 2)  # Random variation
        humidity_effect = (humidity - 60) * 0.02  # Humidity impact
        pressure_effect = (pressure - 1013.25) * 0.01  # Pressure impact
        wind_effect = wind_speed * 0.1  # Wind cooling effect
        
        predicted_temp = (current_temp + variation + humidity_effect + 
                         pressure_effect - wind_effect)
        
        return predicted_temp
    
    def generate_forecast_series(self, initial_temp: float, hours: int) -> pd.DataFrame:
        """
        Generate forecast time series.
        
        Args:
            initial_temp: Starting temperature
            hours: Number of hours to forecast
            
        Returns:
            DataFrame with forecast data
        """
        timestamps = [datetime.now() + timedelta(hours=i) for i in range(hours + 1)]
        temperatures = [initial_temp]
        
        for i in range(1, hours + 1):
            # Simple trend with random variation
            trend = np.sin(i * np.pi / 12) * 2  # Daily cycle
            variation = np.random.normal(0, 1)
            next_temp = temperatures[-1] + trend + variation
            temperatures.append(next_temp)
        
        return pd.DataFrame({
            'datetime': timestamps,
            'temperature': temperatures
        })
    
    def statistics_section(self):
        """Create statistics section."""
        if self.data is not None:
            st.subheader("📊 Statistical Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Temperature Distribution")
                if 'temperature' in self.data.columns:
                    fig = px.histogram(
                        self.data, x='temperature', bins=30,
                        title="Temperature Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Summary Statistics")
                if 'temperature' in self.data.columns:
                    stats = self.data['temperature'].describe()
                    st.dataframe(stats)
    
    def model_performance_section(self):
        """Create model performance section."""
        st.subheader("🎯 Model Performance")
        
        # Simulated performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("RMSE", "2.34°C", delta="-0.12")
        
        with col2:
            st.metric("MAE", "1.89°C", delta="-0.08")
        
        with col3:
            st.metric("R² Score", "0.923", delta="+0.015")
        
        with col4:
            st.metric("Accuracy", "94.2%", delta="+1.1%")
    
    def run(self):
        """Run the Streamlit application."""
        self.main_header()
        
        # Sidebar controls
        selected_model, forecast_hours, confidence_level, uploaded_file = self.sidebar_controls()
        
        # Handle file upload
        if uploaded_file is not None:
            self.data = pd.read_csv(uploaded_file)
            st.success("Data uploaded successfully!")
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "📈 Visualization", "🔮 Prediction", "🎯 Performance"])
        
        with tab1:
            self.data_overview_section()
        
        with tab2:
            self.temperature_trends_section()
            self.statistics_section()
        
        with tab3:
            self.prediction_section(forecast_hours, confidence_level)
        
        with tab4:
            self.model_performance_section()


def main():
    """Main function to run the Streamlit app."""
    app = TemperatureForecastingApp()
    app.run()


if __name__ == "__main__":
    main()