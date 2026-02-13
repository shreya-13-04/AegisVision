# 🚗 Vehicle Detection & Risk Analysis UI - QUICK START GUIDE

## 🚀 Getting Started in 5 Minutes

### Step 1: Extract and Navigate
```bash
cd VehicleDetection
```

### Step 2: Install Dependencies (First Time Only)
```bash
pip install -r requirements_ui.txt
```

### Step 3: Launch the Application
```bash
python launch.py
```

The launcher menu will appear - choose your interface:
- **Option 1**: Web UI (Streamlit) - Modern browser-based interface
- **Option 2**: Desktop UI (Tkinter) - Local desktop application

---

## 📋 System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|------------|
| RAM | 4GB | 8GB+ |
| Storage | 500MB | 1GB |
| GPU | Not required | NVIDIA (3GB+ VRAM) |
| Python | 3.8+ | 3.10+ |

---

## 🎯 Key Features

### ✅ Image Processing
- Upload JPG, PNG, BMP images
- Real-time vehicle detection
- Depth map visualization
- Risk analysis with color-coded alerts
- Detailed detection statistics

### ✅ Video Processing (Web UI)
- Support for MP4, AVI, MOV, MKV
- Frame-by-frame analysis
- Real-time processing progress
- Risk distribution analytics
- Video-level statistics

### ✅ Risk Analysis
Shows for each vehicle:
- **Risk Level**: HIGH RISK 🔴 | MEDIUM 🟠 | SAFE 🟢
- **Risk Score**: 0-300+ (calculated)
- **Depth**: Distance estimation
- **Box Area**: Vehicle size in pixels
- **Position**: Bounding box coordinates

---

## 🎨 UI Options Comparison

### Web UI (Streamlit) - Recommended
```
✅ Modern web interface
✅ Video processing with detailed analytics  
✅ Real-time depth visualization
✅ Interactive charts and graphs
✅ Statistics dashboard
✅ Responsive mobile-friendly
⚠️  Requires browser
```

**Usage**: `python launch.py` → Option 1 → Browser opens at http://localhost:8501

### Desktop UI (Tkinter)
```
✅ Offline, no browser needed
✅ Quick image processing
✅ Configuration management
✅ Single-window interface
✅ Works on all platforms
⚠️  Limited video features
```

**Usage**: `python launch.py` → Option 2

---

## 📊 Understanding Risk Scores

### Formula
```
Risk Score = (Depth Value × 0.6) + (Box Area × 0.0005)
```

### Scale
| Score | Level | Status | Action |
|-------|-------|--------|--------|
| > 170 | HIGH RISK 🔴 | Collision likely | Immediate action |
| 131-170 | MEDIUM 🟠 | Caution zone | Monitor vehicle |
| < 130 | SAFE 🟢 | No threat | No action |

### Parameters Explained
- **Depth Value**: Distance from camera (lower = closer = higher risk)
- **Box Area**: How large the vehicle appears (bigger = higher risk)
- **Both weighted**: Depth has 60% influence, size has 40%

---

## 🛠️ Configuration

### Quick Settings (Sidebar in Web UI)
- Detection Confidence: 0.1-1.0 (lower = more detections)
- High Risk Threshold: Customize risk score boundaries
- Medium Risk Threshold: Customize risk score boundaries

### Persistent Configuration (config.json)
```bash
python launch.py
Select: Option 3 (Configuration Manager)
```

**Editable settings**:
- `confidence_threshold`: Detection confidence (0.0-1.0)
- `high_risk_threshold`: High risk score cutoff
- `medium_risk_threshold`: Medium risk score cutoff
- `frame_resize_width`: Input resolution width
- `frame_resize_height`: Input resolution height

---

## 📁 File Structure
```
VehicleDetection/
├── launch.py                 ← START HERE
├── run_ui.py                 (Alternative launcher)
├── requirements_ui.txt       (Dependencies)
├── config.json              (Settings - auto-created)
│
├── src/
│   ├── ui_app.py            (Web UI - Streamlit)
│   ├── desktop_ui.py        (Desktop UI - Tkinter)
│   ├── collision_system.py  (Core detection engine)
│   ├── depth_model.py       (Depth estimation)
│   └── realtime_collision.py (Video processing)
│
├── model/
│   └── best.pt              (YOLO model - required)
│
├── test/                    (Sample test images/videos)
├── dataset/                 (Training data)
└── SETUP_UI.md             (Detailed setup guide)
```

---

## 🚨 Troubleshooting

### "Model not found"
```
❌ Model not found at ../model/best.pt
```
**Solution**: Ensure `model/best.pt` exists in the project root

### "Models not loaded yet"
**Solution**: Wait 10-15 seconds for first-time model loading (it downloads models)

### "CUDA out of memory"
**Solution**: 
- Lower frame resolution: Edit config.json
- Process fewer frames: Set "Process every N frames" > 1
- Use CPU mode: Usually happens automatically, but restart if needed

### "Port 8501 already in use"
```bash
streamlit run src/ui_app.py --server.port 8502
```

### Slow Processing
**Optimization tips**:
1. Use GPU (automatic if available)
2. Lower input resolution (960x540)
3. Reduce confidence threshold (faster but more FPs)
4. Process every 2-3 frames for video

---

## 💻 Command Reference

### Launch Main Menu
```bash
python launch.py
```

### Launch Web UI Directly
```bash
python run_ui.py
```

### Launch Web UI with Custom Port
```bash
streamlit run src/ui_app.py --server.port 8502
```

### Launch Desktop UI Directly
```bash
python src/desktop_ui.py
```

### Test Models
```bash
python launch.py
Select: Option 4 (Test Models)
```

### View/Edit Configuration
```bash
python launch.py
Select: Option 3 (Configuration Manager)
```

---

## 📈 Performance Tips

| Tip | Impact | Difficulty |
|-----|--------|-----------|
| Use GPU | 40-50x faster | Auto-detected |
| Lower resolution (720p) | 2-3x faster | Easy |
| Process every 2nd frame | 2x faster | Easy |
| Lower confidence threshold | Slight speedup | Easy |
| Close other applications | 10-20% faster | Easy |

---

## 🎓 What the System Detects

### Detection Model (YOLO)
- Identifies vehicle bounding boxes
- Confidence scores for each detection
- Supports: cars, trucks, buses, motorcycles

### Depth Estimation (MiDaS)
- Estimates distance to each vehicle
- Monocular depth from single image
- No stereo/multi-camera setup needed

### Risk Scoring
- Combines depth and size information
- Provides actionable risk levels
- Color-coded for quick assessment

---

## 🔧 Advanced Usage

### Batch Processing Images
Create a Python script:
```python
import cv2
from src.ui_app import process_frame
from ultralytics import YOLO
import torch

# Load models
yolo = YOLO("model/best.pt")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")

# Process each image
for image_path in image_list:
    frame, detections, depth = process_frame(...)
    # Save results
```

### Custom Risk Thresholds
Edit `config.json`:
```json
{
  "high_risk_threshold": 150,
  "medium_risk_threshold": 100
}
```

### Adjust Model Weights
Edit `src/collision_system.py`:
```python
# Current: 60% depth, 40% size
risk_score = depth_value * 0.6 + box_area * 0.0005

# Custom: 70% depth, 30% size
risk_score = depth_value * 0.7 + box_area * 0.0003
```

---

## 📞 Support & Resources

### Models Used
- **YOLO**: https://github.com/ultralytics/ultralytics
- **MiDaS**: https://github.com/isl-org/MiDaS
- **OpenCV**: https://opencv.org/

### Documentation
- YOLO Docs: https://docs.ultralytics.com/
- Streamlit Docs: https://docs.streamlit.io/
- MiDaS GitHub: https://github.com/isl-org/MiDaS

### Keyboard Shortcuts
| Shortcut | Action |
|----------|--------|
| Ctrl+C | Stop application |
| R (in browser) | Reload Streamlit |
| Ctrl+R | Clear cache (Streamlit) |

---

## 🎯 Next Steps

### After First Run
1. ✅ Test with sample image
2. ✅ Adjust confidence threshold
3. ✅ Try different risk thresholds
4. ✅ Test with your own images/videos

### Customization
- [ ] Fine-tune risk thresholds for your use case
- [ ] Adjust frame resolution for better accuracy/speed
- [ ] Modify risk formula for different requirements
- [ ] Export annotated videos/reports

### Enhancement Ideas
- Add webcam real-time detection
- Export results to CSV/PDF
- Set up alerts/notifications
- Train custom model on your data

---

## 📝 Sample Output

### Detection Report
```
DETECTED VEHICLES: 3

Vehicle #1
├─ Risk Level: HIGH RISK (🔴)
├─ Risk Score: 185.4
├─ Depth: 120.5 cm
├─ Box Area: 12000 px²
└─ Position: (156, 234)

Vehicle #2
├─ Risk Level: MEDIUM (🟠)
├─ Risk Score: 145.2
├─ Depth: 180.3 cm
├─ Box Area: 8500 px²
└─ Position: (420, 320)

Vehicle #3
├─ Risk Level: SAFE (🟢)
├─ Risk Score: 95.1
├─ Depth: 240.1 cm
├─ Box Area: 4200 px²
└─ Position: (650, 180)
```

---

## ⚠️ Important Notes

1. **First Load**: Models download automatically (5-10 min, one-time)
2. **GPU Required**: First time setup takes longer on CPU
3. **Privacy**: All processing happens locally, no data collection
4. **Browser**: Web UI works with Chrome, Firefox, Edge, Safari

---

**Happy Vehicle Detection! 🚗**

For detailed technical documentation, see [SETUP_UI.md](SETUP_UI.md)
