#!/usr/bin/env python3
"""
Hanoi Hourly Weather Forecasting App Launcher
Step 8 Implementation - Multi-Horizon Forecasting
"""

import streamlit.web.cli as stcli
import sys
import os

def main():
    """Launch the hourly weather forecasting app"""
    
    # Change to app directory
    app_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(app_dir)
    
    # App configuration
    app_file = "streamlit_app_hourly.py"
    
    # Streamlit configuration
    sys.argv = [
        "streamlit",
        "run",
        app_file,
        "--server.port=8502",  # Different port from daily app
        "--server.address=localhost",
        "--server.headless=false",
        "--browser.gatherUsageStats=false",
        "--theme.base=light"
    ]
    
    print("🌤️ LAUNCHING HANOI HOURLY WEATHER FORECASTING APP")
    print("=" * 55)
    print(f"📱 App: {app_file}")
    print(f"🌐 URL: http://localhost:8502")
    print(f"⏰ Features: Multi-horizon forecasting (1h, 6h, 24h, 72h, 168h)")
    print(f"🎯 Targets: Temperature, Humidity, Pressure, Wind, Clouds")
    print("=" * 55)
    print("🚀 Starting Streamlit server...")
    print("   Press Ctrl+C to stop the server")
    
    # Launch Streamlit
    stcli.main()

if __name__ == "__main__":
    main()