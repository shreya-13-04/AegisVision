# Vehicle Detection UI - Setup Guide

## Overview
A comprehensive web-based UI for vehicle detection and risk analysis using YOLO and MiDaS depth estimation.

## Features ✨

### 📸 Image Processing
- Upload and process single images
- Real-time vehicle detection
- Depth map visualization
- Risk analysis with detailed metrics
- Interactive detection table with all parameters

### 🎥 Video Processing
- Upload and process video files
- Frame-by-frame analysis
- Real-time progress tracking
- Video-level statistics and analytics
- Risk distribution across frames

### 📊 Analytics Dashboard
- System configuration display
- Risk scoring algorithm explanation
- Usage tips and best practices

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- (Optional) CUDA-capable GPU with NVIDIA drivers for faster processing

### Step 1: Install Dependencies

```bash
# Navigate to project directory
cd VehicleDetection

# Install UI dependencies
pip install -r requirements_ui.txt
```

### Step 2: Verify Model
Ensure your YOLO model is at: `model/best.pt`

## Running the UI

### Method 1: Using the Helper Script
```bash
python run_ui.py
```

### Method 2: Direct Streamlit Command
```bash
streamlit run src/ui_app.py
```

The application will open automatically at `http://localhost:8501`

## Usage Guide

### Image Processing
1. Navigate to the **"📸 Image Processing"** tab
2. Upload an image (JPG, PNG, etc.)
3. View detected vehicles with risk levels
4. Analyze depth map and risk metrics

### Video Processing
1. Navigate to the **"🎥 Video Processing"** tab
2. Upload a video file (MP4, AVI, MOV, etc.)
3. Set frame processing interval (optional)
4. Click "Start Processing"
5. View real-time processing with results
6. Analyze overall video statistics

### Configuration
Use the sidebar to adjust:
- **Detection Confidence**: Threshold for vehicle detection (0.1-1.0)
- **Risk Score Thresholds**: Customize HIGH RISK and MEDIUM risk levels
- Device automatically selects GPU if available

## Risk Scoring Algorithm

```
Risk Score = (Depth Value × 0.6) + (Bounding Box Area × 0.0005)
```

### Risk Levels
- 🔴 **HIGH RISK** (>170): Immediate collision threat
- 🟠 **MEDIUM** (131-170): Approach with caution
- 🟢 **SAFE** (≤130): No immediate threat

### Parameters
- **Depth Value**: Distance to vehicle (from MiDaS depth estimation model)
- **Bounding Box Area**: Size of detected vehicle in pixels

## Output Parameters

Each detected vehicle shows:
- **ID**: Vehicle identifier
- **Risk Level**: Classification (HIGH RISK/MEDIUM/SAFE)
- **Risk Score**: Calculated numerical risk value
- **Depth**: Estimated distance in normalized units
- **Box Area**: Size of detection bounding box
- **Position**: X, Y coordinates of vehicle center

## System Requirements

### Minimum
- RAM: 4GB
- Storage: 500MB for models
- GPU: Not required (CPU mode available)

### Recommended
- RAM: 8GB+
- Storage: 1GB free space
- GPU: NVIDIA with CUDA support (3GB+ VRAM)

## Troubleshooting

### Model Not Found
```
❌ Error: Model not found at ../model/best.pt
```
**Solution**: Ensure `model/best.pt` exists in the project root

### CUDA Out of Memory
**Solution**: 
- Process video frames at lower resolution (adjust in code)
- Reduce confidence threshold
- Process every Nth frame for video

### Slow Processing
**Solution**:
- Use GPU instead of CPU (install CUDA)
- Process every 2-3 frames for video instead of every frame
- Reduce image resolution

### Streamlit Port Already in Use
```bash
streamlit run src/ui_app.py --server.port 8502
```

## Advanced Configuration

### Modify Risk Thresholds
Edit `ui_app.py` and change default values:
```python
high_risk_threshold = 170  # Line ~50
medium_risk_threshold = 130  # Line ~55
```

### Change Input Resolution
Adjust in Tab 1 (Image) and Tab 2 (Video):
```python
# Change from (960, 540) to (1280, 720) for higher quality
image_resized = cv2.resize(image_array, (1280, 720))
```

## File Structure
```
VehicleDetection/
├── src/
│   ├── ui_app.py              # Main Streamlit application
│   ├── collision_system.py    # Detection engine
│   ├── depth_model.py         # Depth estimation
│   ├── realtime_collision.py  # Video processing
│   └── ...
├── model/
│   └── best.pt               # YOLO model
├── requirements_ui.txt       # UI dependencies
├── run_ui.py                # Launcher script
└── SETUP_UI.md             # This file
```

## API/Function Reference

### Core Functions

#### `process_frame(frame, yolo_model, midas, transform, conf, high_threshold, medium_threshold)`
Process a single frame and return detections.

**Returns**:
- `processed_frame`: Annotated image with bounding boxes
- `detections_data`: List of detection dictionaries
- `depth_map`: Normalized depth map

#### `calculate_risk_score(depth_value, box_area)`
Calculate risk score for a detection.

**Returns**: Float risk score

#### `get_risk_level(risk_score, high_threshold, medium_threshold)`
Classify risk level based on score.

**Returns**: Tuple of (risk_level_string, color_bgr, risk_index)

## Performance Tips

1. **GPU Acceleration**: Risk ~40-50x speed improvement
2. **Frame Skipping**: Process every 2-3 frames for real-time video
3. **Resolution**: Lower input resolution for faster processing
4. **Batch Processing**: Process multiple frames in parallel (future enhancement)

## Keyboard Shortcuts in Streamlit
- **Ctrl+C**: Stop the application
- **R**: Reload the app
- **Ctrl+R**: Clear cache and reload

## Support & Documentation

### Models Used
- **YOLO**: https://github.com/ultralytics/ultralytics
- **MiDaS**: https://github.com/isl-org/MiDaS

### Dependencies
- Streamlit: https://streamlit.io/
- OpenCV: https://opencv.org/
- PyTorch: https://pytorch.org/

## Future Enhancements
- [ ] Export annotated videos
- [ ] Multiple video processing
- [ ] Real-time webcam input
- [ ] Alert notifications for high-risk scenarios
- [ ] Historical analysis and reports
- [ ] Custom model training integration
