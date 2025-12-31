#!/usr/bin/env python3
"""
Quick launcher script for the QualiVec Streamlit demo.
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit app."""
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, "app.py")
    
    print("🚀 Starting QualiVec Demo...")
    print("📍 App will be available at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the app")
    print("-" * 50)
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", app_path,
            "--server.headless", "true",
            "--server.address=0.0.0.0",
            "--server.port=8501",
            "--server.enableCORS=false",
            "--server.enableXsrfProtection=false"
        ])
    except KeyboardInterrupt:
        print("\n🛑 App stopped by user")
    except Exception as e:
        print(f"❌ Error starting app: {e}")

if __name__ == "__main__":
    main()
