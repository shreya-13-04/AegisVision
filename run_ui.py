"""
Helper script to run the Streamlit UI application
"""

import subprocess
import sys
import os

def run_streamlit_app():
    """Run the Streamlit application."""
    
    # Get the path to the UI app
    src_dir = os.path.join(os.path.dirname(__file__), "src")
    app_path = os.path.join(src_dir, "ui_app.py")
    
    if not os.path.exists(app_path):
        print(f"❌ Error: ui_app.py not found at {app_path}")
        sys.exit(1)
    
    print("🚀 Starting Vehicle Detection UI...")
    print(f"📱 Open your browser at: http://localhost:8501")
    print("\nPress Ctrl+C to stop the application.\n")
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", app_path,
            "--logger.level=info"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped.")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error running app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_streamlit_app()
