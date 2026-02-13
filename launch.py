"""
Master Launcher for Vehicle Detection UI
Choose between Web (Streamlit) or Desktop (Tkinter) interface
"""

import subprocess
import sys
import os
import json
from pathlib import Path

def show_menu():
    """Display launcher menu."""
    print("\n" + "="*60)
    print("🚗 VEHICLE DETECTION & RISK ANALYSIS SYSTEM")
    print("="*60)
    print("\nChoose UI Interface:\n")
    print("1️⃣  Web UI (Streamlit) - Modern, feature-rich web interface")
    print("2️⃣  Desktop UI (Tkinter) - Local desktop application")
    print("3️⃣  Configuration Manager - Setup and adjust settings")
    print("4️⃣  Test Models - Verify model installation")
    print("0️⃣  Exit\n")

def check_models():
    """Check if models exist and are properly located."""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "model", "best.pt")
    
    if not os.path.exists(model_path):
        print(f"\n❌ ERROR: Model not found at {model_path}")
        print("\nExpected structure:")
        print("VehicleDetection/")
        print("├── model/")
        print("│   └── best.pt  ← MISSING")
        return False
    
    print(f"\n✅ Model found: {model_path}")
    return True

def check_dependencies():
    """Check if required packages are installed."""
    required = [
        'streamlit', 'opencv', 'torch', 'ultralytics', 'PIL'
    ]
    
    print("\nChecking dependencies...")
    missing = []
    
    try:
        import streamlit
        print("✅ streamlit")
    except:
        missing.append('streamlit')
    
    try:
        import cv2
        print("✅ opencv-python")
    except:
        missing.append('opencv-python')
    
    try:
        import torch
        print("✅ torch")
    except:
        missing.append('torch')
    
    try:
        from ultralytics import YOLO
        print("✅ ultralytics")
    except:
        missing.append('ultralytics')
    
    try:
        from PIL import Image
        print("✅ Pillow")
    except:
        missing.append('Pillow')
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print(f"  pip install -r requirements_ui.txt\n")
        return False
    
    print("\n✅ All dependencies installed")
    return True

def run_web_ui():
    """Launch Streamlit web UI."""
    src_dir = os.path.join(os.path.dirname(__file__), "src")
    app_path = os.path.join(src_dir, "ui_app.py")
    
    if not os.path.exists(app_path):
        print(f"❌ Error: ui_app.py not found at {app_path}")
        return
    
    print("\n🚀 Starting Web UI (Streamlit)...")
    print("📱 Your browser will open at: http://localhost:8501")
    print("Press Ctrl+C to stop\n")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", app_path,
            "--logger.level=error"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped.")
    except Exception as e:
        print(f"❌ Error: {e}")

def run_desktop_ui():
    """Launch Tkinter desktop UI."""
    src_dir = os.path.join(os.path.dirname(__file__), "src")
    app_path = os.path.join(src_dir, "desktop_ui.py")
    
    if not os.path.exists(app_path):
        print(f"❌ Error: desktop_ui.py not found at {app_path}")
        return
    
    print("\n🚀 Starting Desktop UI (Tkinter)...")
    print("Press Ctrl+C to stop\n")
    
    try:
        subprocess.run([sys.executable, app_path])
    except KeyboardInterrupt:
        print("\n👋 Application stopped.")
    except Exception as e:
        print(f"❌ Error: {e}")

def config_manager():
    """Manage configuration settings."""
    config_file = os.path.join(os.path.dirname(__file__), "config.json")
    
    print("\n"+"="*60)
    print("⚙️  CONFIGURATION MANAGER")
    print("="*60 + "\n")
    
    # Load current config
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        print("Current Configuration:")
    else:
        config = {
            "confidence_threshold": 0.25,
            "high_risk_threshold": 170,
            "medium_risk_threshold": 130,
            "depth_weight": 0.6,
            "area_weight": 0.0005,
            "process_every_n_frames": 2,
            "frame_resize_width": 960,
            "frame_resize_height": 540
        }
        print("Default Configuration:")
    
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\n1. Edit Confidence Threshold")
    print("2. Edit High Risk Threshold")
    print("3. Edit Medium Risk Threshold")
    print("4. Edit Frame Resolution")
    print("0. Back to Main Menu\n")
    
    choice = input("Select option: ").strip()
    
    if choice == "1":
        try:
            val = float(input("Enter confidence threshold (0.0-1.0): "))
            if 0.0 <= val <= 1.0:
                config["confidence_threshold"] = val
                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=2)
                print("✅ Configuration saved!")
            else:
                print("❌ Invalid value")
        except ValueError:
            print("❌ Invalid input")
    
    elif choice == "2":
        try:
            val = int(input("Enter high risk threshold: "))
            config["high_risk_threshold"] = val
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            print("✅ Configuration saved!")
        except ValueError:
            print("❌ Invalid input")
    
    elif choice == "3":
        try:
            val = int(input("Enter medium risk threshold: "))
            config["medium_risk_threshold"] = val
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            print("✅ Configuration saved!")
        except ValueError:
            print("❌ Invalid input")
    
    elif choice == "4":
        try:
            width = int(input("Enter frame width (default 960): "))
            height = int(input("Enter frame height (default 540): "))
            config["frame_resize_width"] = width
            config["frame_resize_height"] = height
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            print("✅ Configuration saved!")
        except ValueError:
            print("❌ Invalid input")

def test_models():
    """Test if models can be loaded."""
    print("\n"+"="*60)
    print("🧪 MODEL TEST")
    print("="*60 + "\n")
    
    print("Checking models...\n")
    
    if not check_models():
        return
    
    print("\nAttempting to load YOLO model...")
    try:
        from ultralytics import YOLO
        import os
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(BASE_DIR, "model", "best.pt")
        model = YOLO(model_path)
        print("✅ YOLO model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading YOLO: {e}")
        return
    
    print("\nAttempting to load MiDaS model...")
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        midas.to(device)
        print(f"✅ MiDaS model loaded successfully (Device: {device})")
    except Exception as e:
        print(f"❌ Error loading MiDaS: {e}")
        return
    
    print("\n✅ All models loaded successfully!")
    print("\n💡 Tip: Download models can take time on first run")

def main():
    """Main launcher loop."""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(BASE_DIR)
    
    # Check dependencies once
    if not check_dependencies():
        response = input("\nInstall dependencies now? (y/n): ").lower()
        if response == 'y':
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements_ui.txt"])
        else:
            print("Please install dependencies to continue.")
            return
    
    while True:
        show_menu()
        choice = input("Enter your choice (0-4): ").strip()
        
        if choice == "1":
            run_web_ui()
        
        elif choice == "2":
            run_desktop_ui()
        
        elif choice == "3":
            config_manager()
        
        elif choice == "4":
            test_models()
        
        elif choice == "0":
            print("\n👋 Goodbye!\n")
            break
        
        else:
            print("\n❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
