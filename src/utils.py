"""
Utility Functions for Vehicle Detection System
Supports batch processing, data management, and analysis
"""

import cv2
import os
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch
from datetime import datetime

class VehicleDetectionUtils:
    """Utility class for vehicle detection operations."""
    
    def __init__(self, model_path="model/best.pt"):
        """Initialize utilities."""
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_model = None
        self.config = self.load_config()
    
    @staticmethod
    def load_config():
        """Load configuration."""
        if os.path.exists("config.json"):
            with open("config.json", 'r') as f:
                return json.load(f)
        return {
            "confidence_threshold": 0.25,
            "high_risk_threshold": 170,
            "medium_risk_threshold": 130,
        }
    
    def load_model(self):
        """Load YOLO model."""
        if self.yolo_model is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found: {self.model_path}")
            self.yolo_model = YOLO(self.model_path)
            print(f"✅ Model loaded: {self.model_path}")
        return self.yolo_model
    
    def process_batch_images(self, input_dir, output_dir="batch_output", conf=0.25):
        """Process all images in a directory."""
        print(f"\n📁 Batch Processing Images from: {input_dir}")
        
        self.load_model()
        Path(output_dir).mkdir(exist_ok=True)
        
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = [f for f in os.listdir(input_dir) 
                      if f.lower().endswith(supported_formats)]
        
        if not image_files:
            print("❌ No images found")
            return
        
        print(f"Found {len(image_files)} images\n")
        
        results_summary = []
        
        for i, img_file in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] Processing: {img_file}")
            
            img_path = os.path.join(input_dir, img_file)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"  ⚠️  Failed to read image")
                continue
            
            # Detect
            results = self.yolo_model(img, conf=conf)
            num_detections = len(results[0].boxes)
            
            print(f"  ✅ Found {num_detections} vehicles")
            
            # Draw boxes
            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Save
            output_path = os.path.join(output_dir, f"detected_{i}_{img_file}")
            cv2.imwrite(output_path, img)
            
            results_summary.append({
                "filename": img_file,
                "vehicles": num_detections,
                "output": output_path
            })
        
        # Save summary
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\n✅ Batch processing complete!")
        print(f"📊 Results saved to: {output_dir}")
        print(f"   Total images: {len(image_files)}")
        print(f"   Total vehicles: {sum(r['vehicles'] for r in results_summary)}")
        print(f"   Summary: {summary_path}")
    
    def process_batch_videos(self, input_dir, output_dir="batch_output", 
                            conf=0.25, skip_frames=2):
        """Process all videos in a directory."""
        print(f"\n🎥 Batch Processing Videos from: {input_dir}")
        
        self.load_model()
        Path(output_dir).mkdir(exist_ok=True)
        
        video_formats = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
        video_files = [f for f in os.listdir(input_dir) 
                      if f.lower().endswith(video_formats)]
        
        if not video_files:
            print("❌ No videos found")
            return
        
        print(f"Found {len(video_files)} videos\n")
        
        for i, vid_file in enumerate(video_files, 1):
            print(f"[{i}/{len(video_files)}] Processing: {vid_file}")
            
            vid_path = os.path.join(input_dir, vid_file)
            
            # Count detections
            cap = cv2.VideoCapture(vid_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_vehicles = 0
            processed_frames = 0
            
            frame_num = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_num % skip_frames == 0:
                    frame = cv2.resize(frame, (960, 540))
                    results = self.yolo_model(frame, conf=conf)
                    total_vehicles += len(results[0].boxes)
                    processed_frames += 1
                
                frame_num += 1
            
            cap.release()
            
            print(f"  ✅ {total_vehicles} vehicles detected across {processed_frames} frames")
    
    def export_to_csv(self, detections, output_file="detections.csv"):
        """Export detections to CSV."""
        import csv
        
        print(f"\n📊 Exporting to CSV: {output_file}")
        
        if not detections:
            print("❌ No detections to export")
            return
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "ID", "Risk Level", "Risk Score", "Depth", 
                "Box Area", "X1", "Y1", "X2", "Y2"
            ])
            writer.writeheader()
            
            for det in detections:
                writer.writerow({
                    "ID": det.get("id"),
                    "Risk Level": det.get("risk_level"),
                    "Risk Score": f"{det.get('risk_score', 0):.2f}",
                    "Depth": f"{det.get('depth', 0):.1f}",
                    "Box Area": f"{det.get('area', 0):.0f}",
                    "X1": det["bbox"][0],
                    "Y1": det["bbox"][1],
                    "X2": det["bbox"][2],
                    "Y2": det["bbox"][3],
                })
        
        print(f"✅ Exported {len(detections)} detections to {output_file}")
    
    def generate_report(self, detections, output_file="report.txt"):
        """Generate text report."""
        print(f"\n📄 Generating Report: {output_file}")
        
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("VEHICLE DETECTION & RISK ANALYSIS REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Vehicles: {len(detections)}\n")
            
            if detections:
                high_risk = sum(1 for d in detections if d["risk_level"] == "HIGH RISK")
                medium = sum(1 for d in detections if d["risk_level"] == "MEDIUM")
                safe = sum(1 for d in detections if d["risk_level"] == "SAFE")
                
                f.write(f"High Risk: {high_risk}\n")
                f.write(f"Medium Risk: {medium}\n")
                f.write(f"Safe: {safe}\n\n")
                
                avg_risk = np.mean([d["risk_score"] for d in detections])
                f.write(f"Average Risk Score: {avg_risk:.2f}\n\n")
                
                f.write("DETAILED DETECTIONS\n")
                f.write("-"*80 + "\n")
                
                for det in detections:
                    f.write(f"\nVehicle #{det['id']}\n")
                    f.write(f"  Risk Level: {det['risk_level']}\n")
                    f.write(f"  Risk Score: {det['risk_score']:.2f}\n")
                    f.write(f"  Depth: {det['depth']:.1f} cm\n")
                    f.write(f"  Box Area: {det['area']:.0f} pixels\n")
                    f.write(f"  Position: ({det['bbox'][0]}, {det['bbox'][1]})\n")
        
        print(f"✅ Report saved to {output_file}")
    
    def validate_installation(self):
        """Validate system setup."""
        print("\n" + "="*80)
        print("🧪 SYSTEM VALIDATION")
        print("="*80 + "\n")
        
        # Check Python
        import sys
        print(f"✅ Python: {sys.version.split()[0]}")
        
        # Check PyTorch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ GPU Available: {'Yes' if torch.cuda.is_available() else 'No'}")
        
        # Check OpenCV
        print(f"✅ OpenCV: {cv2.__version__}")
        
        # Check model
        if os.path.exists(self.model_path):
            size_mb = os.path.getsize(self.model_path) / (1024*1024)
            print(f"✅ Model: Found ({size_mb:.1f} MB)")
        else:
            print(f"❌ Model: Not found at {self.model_path}")
        
        # Try load model
        try:
            self.load_model()
            print("✅ Model Loading: Success")
        except Exception as e:
            print(f"❌ Model Loading: {e}")
        
        print("\n" + "="*80)

def main():
    """Main utility menu."""
    print("\n" + "="*80)
    print("🚗 UTILITY FUNCTIONS - VEHICLE DETECTION SYSTEM")
    print("="*80 + "\n")
    
    utils = VehicleDetectionUtils()
    
    while True:
        print("\nOptions:")
        print("1. Batch Process Images")
        print("2. Batch Process Videos")
        print("3. Export to CSV")
        print("4. Generate Report")
        print("5. Validate Installation")
        print("0. Exit\n")
        
        choice = input("Select option: ").strip()
        
        if choice == "1":
            input_dir = input("Enter image directory path: ").strip()
            if os.path.isdir(input_dir):
                utils.process_batch_images(input_dir)
            else:
                print("❌ Directory not found")
        
        elif choice == "2":
            input_dir = input("Enter video directory path: ").strip()
            if os.path.isdir(input_dir):
                utils.process_batch_videos(input_dir)
            else:
                print("❌ Directory not found")
        
        elif choice == "3":
            output_file = input("Enter output CSV filename (default: detections.csv): ").strip()
            if not output_file:
                output_file = "detections.csv"
            # Placeholder detections for demo
            detections = [
                {"id": 1, "risk_level": "HIGH RISK", "risk_score": 185.4, 
                 "depth": 120.5, "area": 12000, "bbox": (156, 234, 400, 500)},
            ]
            utils.export_to_csv(detections, output_file)
        
        elif choice == "4":
            output_file = input("Enter output report filename (default: report.txt): ").strip()
            if not output_file:
                output_file = "report.txt"
            # Placeholder detections for demo
            detections = [
                {"id": 1, "risk_level": "HIGH RISK", "risk_score": 185.4, 
                 "depth": 120.5, "area": 12000, "bbox": (156, 234, 400, 500)},
            ]
            utils.generate_report(detections, output_file)
        
        elif choice == "5":
            utils.validate_installation()
        
        elif choice == "0":
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()
