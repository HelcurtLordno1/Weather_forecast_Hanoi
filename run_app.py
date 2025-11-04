#!/usr/bin/env python3
"""
🚀 Hanoi Temperature Forecasting App Launcher

Simple launcher script to run the Streamlit application with proper configuration.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Launch the Streamlit application."""
    print("🌡️ Starting Hanoi Temperature Forecasting Application...")
    print("=" * 60)
    
    # Get the current directory
    current_dir = Path(__file__).parent
    app_file = current_dir / "app_streamlit.py"
    
    # Check if the app file exists
    if not app_file.exists():
        print("❌ Error: app_streamlit.py not found!")
        print(f"   Looking for: {app_file}")
        return
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit found")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        print("✅ Streamlit installed")
    
    # Check for other required packages
    required_packages = ['plotly', 'pandas', 'numpy', 'joblib']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} found")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} missing")
    
    # Install missing packages
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
        print("✅ All packages installed")
    
    # Set environment variables
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_SERVER_PORT'] = '8501'
    
    print("\n🚀 Launching Streamlit app...")
    print("📱 The app will open in your default web browser")
    print("🌐 URL: http://localhost:8501")
    print("\n💡 To stop the app, press Ctrl+C in this terminal")
    print("=" * 60)
    
    # Launch the app
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_file),
            "--server.address", "localhost",
            "--server.port", "8501",
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error running application: {e}")


if __name__ == "__main__":
    main()