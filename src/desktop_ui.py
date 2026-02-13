"""
Desktop GUI Application for Vehicle Detection & Risk Analysis
Using Tkinter for offline/local deployment
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
import cv2
import torch
import numpy as np
from PIL import Image, ImageTk
import threading
import os
from ultralytics import YOLO
from pathlib import Path
import json
from datetime import datetime

# ============================================================
# CONFIG MANAGEMENT
# ============================================================
CONFIG_FILE = "config.json"

DEFAULT_CONFIG = {
    "confidence_threshold": 0.25,
    "high_risk_threshold": 170,
    "medium_risk_threshold": 130,
    "depth_weight": 0.6,
    "area_weight": 0.0005,
    "process_every_n_frames": 2,
    "frame_resize_width": 960,
    "frame_resize_height": 540
}

def load_config():
    """Load configuration from file or use defaults."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                return {**DEFAULT_CONFIG, **config}
        except:
            return DEFAULT_CONFIG
    return DEFAULT_CONFIG

def save_config(config):
    """Save configuration to file."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

# ============================================================
# MAIN APPLICATION
# ============================================================
class VehicleDetectionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🚗 Vehicle Detection & Risk Analysis System")
        self.root.geometry("1200x700")
        
        # Load config
        self.config = load_config()
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Models
        self.yolo_model = None
        self.midas = None
        self.transform = None
        
        # State
        self.processing = False
        self.current_detections = []
        
        # Setup UI
        self.setup_ui()
        
        # Load models in background
        self.root.after(500, self.load_models)
    
    def setup_ui(self):
        """Create the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left Panel - Controls
        left_frame = ttk.Frame(main_frame, width=250)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5))
        
        # Right Panel - Display
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # ============================================================
        # LEFT PANEL - CONTROLS
        # ============================================================
        ttk.Label(left_frame, text="🎛️ CONTROLS", font=("Arial", 12, "bold")).pack()
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        
        # File input
        ttk.Label(left_frame, text="📁 Input File", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        file_frame = ttk.Frame(left_frame)
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(file_frame, text="📷 Image", command=self.load_image, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(file_frame, text="🎥 Video", command=self.load_video, width=10).pack(side=tk.LEFT, padx=2)
        
        self.file_label = ttk.Label(left_frame, text="No file loaded", foreground="gray")
        self.file_label.pack(fill=tk.X, pady=5)
        
        # Configuration
        ttk.Label(left_frame, text="⚙️ SETTINGS", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 5))
        
        # Confidence
        ttk.Label(left_frame, text="Detection Confidence").pack(anchor=tk.W)
        self.conf_scale = ttk.Scale(left_frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL)
        self.conf_scale.set(self.config["confidence_threshold"])
        self.conf_scale.pack(fill=tk.X, pady=2)
        self.conf_label = ttk.Label(left_frame, text=f"{self.config['confidence_threshold']:.2f}")
        self.conf_label.pack(anchor=tk.E)
        self.conf_scale.config(command=self.update_conf_label)
        
        # High Risk Threshold
        ttk.Label(left_frame, text="High Risk Threshold").pack(anchor=tk.W, pady=(5, 0))
        self.high_risk_var = tk.StringVar(value=str(self.config["high_risk_threshold"]))
        high_risk_entry = ttk.Entry(left_frame, textvariable=self.high_risk_var, width=10)
        high_risk_entry.pack(anchor=tk.W, pady=2)
        
        # Medium Risk Threshold
        ttk.Label(left_frame, text="Medium Risk Threshold").pack(anchor=tk.W, pady=(5, 0))
        self.medium_risk_var = tk.StringVar(value=str(self.config["medium_risk_threshold"]))
        medium_risk_entry = ttk.Entry(left_frame, textvariable=self.medium_risk_var, width=10)
        medium_risk_entry.pack(anchor=tk.W, pady=2)
        
        # Process buttons
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        ttk.Button(left_frame, text="🔍 Process Image", command=self.process_image_btn).pack(fill=tk.X, pady=5)
        ttk.Button(left_frame, text="▶ Process Video", command=self.process_video_btn).pack(fill=tk.X, pady=5)
        
        # Status
        ttk.Label(left_frame, text="📊 STATUS", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 5))
        
        self.status_label = ttk.Label(left_frame, text="Ready", foreground="green")
        self.status_label.pack(fill=tk.X, pady=5)
        
        self.progress = ttk.Progressbar(left_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=5)
        
        # Device info
        ttk.Label(left_frame, text="🔧 DEVICE", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 5))
        ttk.Label(left_frame, text=f"Device: {self.device}").pack(anchor=tk.W)
        ttk.Label(left_frame, text=f"GPU: {'Yes' if torch.cuda.is_available() else 'No'}").pack(anchor=tk.W)
        
        # ============================================================
        # RIGHT PANEL - DISPLAY
        # ============================================================
        # Tabs
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Image Display
        self.image_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.image_tab, text="📸 Image")
        
        image_display_frame = ttk.Frame(self.image_tab)
        image_display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        col1, col2 = image_display_frame, None
        
        # Original image
        ttk.Label(image_display_frame, text="Input Image", font=("Arial", 10, "bold")).pack(side=tk.LEFT, side=tk.TOP)
        self.image_label = ttk.Label(image_display_frame)
        self.image_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Processed image
        ttk.Label(image_display_frame, text="Detected Vehicles", font=("Arial", 10, "bold")).pack(side=tk.RIGHT, side=tk.TOP)
        self.processed_label = ttk.Label(image_display_frame)
        self.processed_label.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # Tab 2: Detection Details
        self.details_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.details_tab, text="📊 Details")
        
        self.details_text = scrolledtext.ScrolledText(self.details_tab, height=20, width=100)
        self.details_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab 3: Statistics
        self.stats_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.stats_tab, text="📈 Statistics")
        
        self.stats_text = scrolledtext.ScrolledText(self.stats_tab, height=20, width=100)
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def update_conf_label(self, value):
        """Update confidence label."""
        self.conf_label.config(text=f"{float(value):.2f}")
    
    def load_models(self):
        """Load YOLO and MiDaS models."""
        self.update_status("Loading models...", "orange")
        self.progress.start()
        
        def load():
            try:
                # Load YOLO
                BASE_DIR = os.path.dirname(os.path.abspath(__file__))
                MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "best.pt")
                
                if not os.path.exists(MODEL_PATH):
                    self.update_status(f"❌ Model not found: {MODEL_PATH}", "red")
                    return
                
                self.yolo_model = YOLO(MODEL_PATH)
                
                # Load MiDaS
                self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
                self.midas.to(self.device)
                self.midas.eval()
                
                transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
                self.transform = transforms.small_transform
                
                self.update_status("✅ Models loaded", "green")
                self.progress.stop()
            except Exception as e:
                self.update_status(f"❌ Error loading models: {str(e)}", "red")
                self.progress.stop()
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
    
    def update_status(self, text, color="black"):
        """Update status label."""
        self.status_label.config(text=text, foreground=color)
        self.root.update()
    
    def load_image(self):
        """Load image file."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if file_path:
            self.file_path = file_path
            self.file_label.config(text=os.path.basename(file_path))
            
            # Display thumbnail
            img = cv2.imread(file_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (250, 250))
            photo = ImageTk.PhotoImage(Image.fromarray(img_resized))
            self.image_label.config(image=photo)
            self.image_label.image = photo
    
    def load_video(self):
        """Load video file."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if file_path:
            self.file_path = file_path
            self.file_label.config(text=os.path.basename(file_path))
    
    def process_image_btn(self):
        """Process image button clicked."""
        if not hasattr(self, 'file_path'):
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        if self.yolo_model is None:
            messagebox.showerror("Error", "Models not loaded yet")
            return
        
        try:
            high_risk = int(self.high_risk_var.get())
            medium_risk = int(self.medium_risk_var.get())
            conf = float(self.conf_scale.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid threshold values")
            return
        
        def process():
            try:
                self.update_status("Processing image...", "orange")
                self.progress.start()
                
                # Read and resize image
                img = cv2.imread(self.file_path)
                img = cv2.resize(img, (
                    self.config["frame_resize_width"],
                    self.config["frame_resize_height"]
                ))
                
                # Get depth map
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                input_batch = self.transform(img_rgb).to(self.device)
                
                with torch.no_grad():
                    prediction = self.midas(input_batch)
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=img_rgb.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()
                
                depth_map = prediction.cpu().numpy()
                depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
                depth_map = depth_map.astype(np.uint8)
                
                # YOLO detection
                results = self.yolo_model(img, conf=conf)
                
                self.current_detections = []
                
                # Process detections
                for r in results:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = map(int, box)
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        
                        cy = min(max(cy, 0), depth_map.shape[0] - 1)
                        cx = min(max(cx, 0), depth_map.shape[1] - 1)
                        
                        depth_value = float(depth_map[cy, cx])
                        box_area = (x2 - x1) * (y2 - y1)
                        risk_score = depth_value * 0.6 + box_area * 0.0005
                        
                        if risk_score > high_risk:
                            risk_level = "HIGH RISK"
                            color = (0, 0, 255)
                        elif risk_score > medium_risk:
                            risk_level = "MEDIUM"
                            color = (0, 165, 255)
                        else:
                            risk_level = "SAFE"
                            color = (0, 255, 0)
                        
                        self.current_detections.append({
                            "id": i + 1,
                            "depth": depth_value,
                            "area": box_area,
                            "risk_score": risk_score,
                            "risk_level": risk_level,
                            "bbox": (x1, y1, x2, y2),
                            "color": color
                        })
                        
                        # Draw
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(img, risk_level, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        cv2.putText(img, f"D:{depth_value:.1f}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Display results
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (400, 300))
                photo = ImageTk.PhotoImage(Image.fromarray(img_resized))
                self.processed_label.config(image=photo)
                self.processed_label.image = photo
                
                # Update details tab
                self.update_details_tab()
                self.update_stats_tab()
                
                self.update_status(f"✅ Processing complete: {len(self.current_detections)} vehicles detected", "green")
                self.progress.stop()
            except Exception as e:
                self.update_status(f"❌ Error: {str(e)}", "red")
                self.progress.stop()
                messagebox.showerror("Error", str(e))
        
        thread = threading.Thread(target=process, daemon=True)
        thread.start()
    
    def process_video_btn(self):
        """Process video button clicked."""
        if not hasattr(self, 'file_path') or not self.file_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            messagebox.showwarning("Warning", "Please load a video file first")
            return
        
        if self.yolo_model is None:
            messagebox.showerror("Error", "Models not loaded yet")
            return
        
        # Similar to image processing but for video
        messagebox.showinfo("Info", "Video processing is being prepared. Please use the Streamlit UI for advanced video features.")
    
    def update_details_tab(self):
        """Update detection details tab."""
        self.details_text.config(state=tk.NORMAL)
        self.details_text.delete(1.0, tk.END)
        
        text = "DETECTION DETAILS\n"
        text += "=" * 80 + "\n\n"
        
        if not self.current_detections:
            text += "No detections\n"
        else:
            text += f"{'ID':<5} {'Risk Level':<12} {'Risk Score':<12} {'Depth':<10} {'Box Area':<12}\n"
            text += "-" * 80 + "\n"
            
            for det in self.current_detections:
                text += f"{det['id']:<5} {det['risk_level']:<12} {det['risk_score']:<12.2f} {det['depth']:<10.1f} {det['area']:<12.0f}\n"
        
        self.details_text.insert(1.0, text)
        self.details_text.config(state=tk.DISABLED)
    
    def update_stats_tab(self):
        """Update statistics tab."""
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        
        text = "STATISTICS & SUMMARY\n"
        text += "=" * 80 + "\n\n"
        
        if self.current_detections:
            total = len(self.current_detections)
            high_risk = sum(1 for d in self.current_detections if d["risk_level"] == "HIGH RISK")
            medium = sum(1 for d in self.current_detections if d["risk_level"] == "MEDIUM")
            safe = sum(1 for d in self.current_detections if d["risk_level"] == "SAFE")
            
            avg_risk = np.mean([d["risk_score"] for d in self.current_detections])
            avg_depth = np.mean([d["depth"] for d in self.current_detections])
            
            text += f"Total Vehicles Detected:  {total}\n"
            text += f"High Risk:               {high_risk}\n"
            text += f"Medium Risk:             {medium}\n"
            text += f"Safe:                    {safe}\n"
            text += f"\nAverage Risk Score:      {avg_risk:.2f}\n"
            text += f"Average Depth:           {avg_depth:.1f}\n"
            text += f"\nProcessed at:            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        else:
            text += "No detections to analyze\n"
        
        self.stats_text.insert(1.0, text)
        self.stats_text.config(state=tk.DISABLED)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = VehicleDetectionUI(root)
    root.mainloop()
