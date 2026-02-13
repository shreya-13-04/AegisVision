import streamlit as st
import cv2
import torch
import numpy as np
import os
from ultralytics import YOLO
from pathlib import Path
import tempfile
from PIL import Image
import time

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Vehicle Detection & Risk Analysis",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM STYLING
# ============================================================
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .high-risk {
        background-color: #ff4444;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-weight: bold;
    }
    .medium-risk {
        background-color: #ffa500;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-weight: bold;
    }
    .safe {
        background-color: #44aa44;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR CONFIGURATION
# ============================================================
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")

# Model confidence
confidence_threshold = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.1,
    max_value=1.0,
    value=0.25,
    step=0.05
)

# Risk score thresholds
st.sidebar.subheader("Risk Score Thresholds")
col1, col2 = st.sidebar.columns(2)
with col1:
    high_risk_threshold = st.number_input(
        "High Risk (>)",
        value=170,
        min_value=100,
        max_value=300
    )
with col2:
    medium_risk_threshold = st.number_input(
        "Medium Risk (>)",
        value=130,
        min_value=50,
        max_value=200
    )

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.info(f"🔧 Device: {device}")

# ============================================================
# MODEL LOADING (Cached)
# ============================================================
@st.cache_resource
def load_yolo_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "best.pt")
    
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}")
        return None
    
    return YOLO(MODEL_PATH)

@st.cache_resource
def load_depth_model():
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.to(device)
    midas.eval()
    
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.small_transform
    
    return midas, transform

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_depth_map(image_rgb, midas, transform):
    """Generate depth map for an image."""
    input_batch = transform(image_rgb).to(device)
    
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    depth_map = prediction.cpu().numpy()
    
    # Normalize for visualization
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map = depth_map.astype(np.uint8)
    
    return depth_map, prediction.cpu().numpy()

def calculate_risk_score(depth_value, box_area):
    """Calculate risk score based on depth and bounding box area."""
    risk_score = depth_value * 0.6 + box_area * 0.0005
    return risk_score

def get_risk_level(risk_score, high_threshold, medium_threshold):
    """Determine risk level based on score."""
    if risk_score > high_threshold:
        return "HIGH RISK", (0, 0, 255), 2
    elif risk_score > medium_threshold:
        return "MEDIUM", (0, 165, 255), 1
    else:
        return "SAFE", (0, 255, 0), 0

def process_frame(frame, yolo_model, midas, transform, conf, high_threshold, medium_threshold):
    """Process a single frame for detection and risk analysis."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Get depth map
    depth_map, depth_raw = get_depth_map(frame_rgb, midas, transform)
    
    # YOLO detection
    results = yolo_model(frame, conf=conf)
    
    detections_data = []
    
    # Process detections
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            
            # Center of bounding box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            # Ensure coordinates are within bounds
            cy = min(max(cy, 0), depth_map.shape[0] - 1)
            cx = min(max(cx, 0), depth_map.shape[1] - 1)
            
            depth_value = float(depth_map[cy, cx])
            box_area = (x2 - x1) * (y2 - y1)
            
            # Calculate risk
            risk_score = calculate_risk_score(depth_value, box_area)
            risk_level, color, risk_idx = get_risk_level(risk_score, high_threshold, medium_threshold)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            cv2.putText(
                frame,
                f"{risk_level}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
            
            # Draw depth value
            cv2.putText(
                frame,
                f"D:{depth_value:.1f}",
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
            
            # Store detection data
            detections_data.append({
                "id": i + 1,
                "bbox": (x1, y1, x2, y2),
                "depth": depth_value,
                "area": box_area,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "color": color
            })
    
    return frame, detections_data, depth_map

# ============================================================
# MAIN APP
# ============================================================
def main():
    st.title("🚗 Vehicle Detection & Risk Analysis System")
    st.markdown("---")
    
    # Load models
    with st.spinner("Loading models..."):
        yolo_model = load_yolo_model()
        midas, transform = load_depth_model()
    
    if yolo_model is None:
        st.error("Failed to load models. Please check the model paths.")
        return
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📸 Image Processing", "🎥 Video Processing", "📊 Analytics"])
    
    # ============================================================
    # TAB 1: IMAGE PROCESSING
    # ============================================================
    with tab1:
        st.header("Image Processing")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload Image")
            uploaded_image = st.file_uploader(
                "Choose an image file (JPG, PNG, etc.)",
                type=["jpg", "jpeg", "png", "bmp"],
                key="image_uploader"
            )
        
        if uploaded_image is not None:
            # Read image
            image_array = np.array(Image.open(uploaded_image))
            
            # Resize for processing
            height = min(image_array.shape[0], 1080)
            aspect_ratio = image_array.shape[1] / image_array.shape[0]
            width = int(height * aspect_ratio)
            image_resized = cv2.resize(image_array, (width, height))
            
            # Convert to BGR for OpenCV
            image_bgr = cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR)
            
            # Process image
            with st.spinner("Processing image..."):
                processed_frame, detections, depth_map = process_frame(
                    image_bgr,
                    yolo_model,
                    midas,
                    transform,
                    confidence_threshold,
                    high_risk_threshold,
                    medium_risk_threshold
                )
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Detected Vehicles")
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                st.image(processed_frame_rgb, use_column_width=True)
            
            with col2:
                st.subheader("Depth Map")
                st.image(depth_map, use_column_width=True, clamp=True)
            
            # ============================================================
            # DETECTION STATISTICS
            # ============================================================
            st.markdown("---")
            st.subheader("📊 Detection Statistics")
            
            if len(detections) > 0:
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Vehicles", len(detections))
                
                high_risk_count = sum(1 for d in detections if d["risk_level"] == "HIGH RISK")
                with col2:
                    st.metric("High Risk", high_risk_count, delta=None)
                
                medium_risk_count = sum(1 for d in detections if d["risk_level"] == "MEDIUM")
                with col3:
                    st.metric("Medium Risk", medium_risk_count, delta=None)
                
                safe_count = sum(1 for d in detections if d["risk_level"] == "SAFE")
                with col4:
                    st.metric("Safe", safe_count, delta=None)
                
                # Detailed detections table
                st.subheader("Detailed Detection Report")
                
                detection_data = []
                for det in detections:
                    detection_data.append({
                        "ID": det["id"],
                        "Risk Level": det["risk_level"],
                        "Risk Score": f"{det['risk_score']:.2f}",
                        "Depth (cm)": f"{det['depth']:.1f}",
                        "Box Area (px²)": f"{det['area']:.0f}",
                        "Position": f"({det['bbox'][0]}, {det['bbox'][1]})"
                    })
                
                st.dataframe(detection_data, use_container_width=True)
                
                # Risk distribution chart
                import plotly.express as px
                risk_counts = {}
                for det in detections:
                    risk = det["risk_level"]
                    risk_counts[risk] = risk_counts.get(risk, 0) + 1
                
                fig = px.pie(
                    values=list(risk_counts.values()),
                    names=list(risk_counts.keys()),
                    title="Risk Distribution",
                    color_discrete_map={"HIGH RISK": "#ff4444", "MEDIUM": "#ffa500", "SAFE": "#44aa44"}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No vehicles detected in the image.")
    
    # ============================================================
    # TAB 2: VIDEO PROCESSING
    # ============================================================
    with tab2:
        st.header("Video Processing")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Upload Video")
            uploaded_video = st.file_uploader(
                "Choose a video file (MP4, AVI, MOV, etc.)",
                type=["mp4", "avi", "mov", "mkv"],
                key="video_uploader"
            )
        
        with col2:
            st.subheader("Processing Settings")
            process_every_n_frames = st.number_input(
                "Process every N frames",
                value=2,
                min_value=1,
                max_value=10
            )
        
        if uploaded_video is not None:
            # Save uploaded video
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_video.read())
                video_path = tmp_file.name
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            st.info(f"Video: {total_frames} frames @ {fps:.1f} FPS")
            
            # Create placeholders
            progress_bar = st.progress(0)
            status_text = st.empty()
            col1, col2 = st.columns(2)
            frame_placeholder = col1.empty()
            depth_placeholder = col2.empty()
            
            # Statistics
            all_detections = []
            frame_count = 0
            processed_frames = 0
            
            # Process video
            process_button = st.button("Start Processing", key="process_video")
            
            if process_button:
                start_time = time.time()
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    
                    if not ret:
                        break
                    
                    if frame_count % process_every_n_frames == 0:
                        # Resize frame
                        frame = cv2.resize(frame, (960, 540))
                        
                        # Process frame
                        processed_frame, detections, depth_map = process_frame(
                            frame,
                            yolo_model,
                            midas,
                            transform,
                            confidence_threshold,
                            high_risk_threshold,
                            medium_risk_threshold
                        )
                        
                        # Store detections
                        all_detections.extend(detections)
                        processed_frames += 1
                        
                        # Display
                        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame_rgb, use_column_width=True)
                        depth_placeholder.image(depth_map, use_column_width=True, clamp=True)
                        
                        # Update progress
                        progress = frame_count / total_frames
                        progress_bar.progress(progress)
                        status_text.text(f"Processing: Frame {frame_count}/{total_frames}")
                    
                    frame_count += 1
                
                cap.release()
                elapsed_time = time.time() - start_time
                
                # Final results
                st.success(f"✅ Processing complete in {elapsed_time:.2f} seconds")
                
                # Video statistics
                st.markdown("---")
                st.subheader("📊 Video Analysis Results")
                
                if len(all_detections) > 0:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Vehicles Detected", len(all_detections))
                    
                    high_risk_count = sum(1 for d in all_detections if d["risk_level"] == "HIGH RISK")
                    with col2:
                        st.metric("High Risk Detections", high_risk_count)
                    
                    medium_risk_count = sum(1 for d in all_detections if d["risk_level"] == "MEDIUM")
                    with col3:
                        st.metric("Medium Risk Detections", medium_risk_count)
                    
                    safe_count = sum(1 for d in all_detections if d["risk_level"] == "SAFE")
                    with col4:
                        st.metric("Safe Detections", safe_count)
                    
                    # Risk score distribution
                    risk_scores = [d["risk_score"] for d in all_detections]
                    avg_risk = np.mean(risk_scores)
                    max_risk = np.max(risk_scores)
                    min_risk = np.min(risk_scores)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Avg Risk Score", f"{avg_risk:.2f}")
                    with col2:
                        st.metric("Max Risk Score", f"{max_risk:.2f}")
                    with col3:
                        st.metric("Min Risk Score", f"{min_risk:.2f}")
                    
                    # Charts
                    import plotly.graph_objects as go
                    
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=risk_scores,
                        nbinsx=20,
                        name="Risk Score Distribution"
                    ))
                    fig.update_layout(
                        title="Risk Score Distribution Across Video",
                        xaxis_title="Risk Score",
                        yaxis_title="Frequency"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No vehicles detected in the video.")
            
            # Cleanup: ensure video handle is released and attempt to remove temp file
            try:
                # Ensure capture is released
                if 'cap' in locals() and cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass

                # Close any OpenCV windows and force garbage collection
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass

                import gc
                time.sleep(0.05)
                gc.collect()

                # Try removing the temp file
                try:
                    os.unlink(video_path)
                except PermissionError:
                    st.warning(f"Could not delete temporary video file (in use): {video_path}. It will be removed when no longer locked.")
                except Exception as e:
                    st.warning(f"Cleanup error deleting temp file: {e}")
            except Exception as e:
                st.warning(f"Cleanup error: {e}")
    
    # ============================================================
    # TAB 3: ANALYTICS & INSIGHTS
    # ============================================================
    with tab3:
        st.header("📊 System Information & Insights")
        
        # System info
        st.subheader("System Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**GPU Available:** {'Yes' if torch.cuda.is_available() else 'No'}")
            st.write(f"**Device:** {device}")
            st.write(f"**CUDA Version:** {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
        
        with col2:
            st.write(f"**Confidence Threshold:** {confidence_threshold}")
            st.write(f"**High Risk Threshold:** {high_risk_threshold}")
            st.write(f"**Medium Risk Threshold:** {medium_risk_threshold}")
        
        # Risk scoring explanation
        st.markdown("---")
        st.subheader("🎯 Risk Scoring Algorithm")
        st.markdown("""
        **Risk Score Formula:**
        ```
        Risk Score = (Depth Value × 0.6) + (Bounding Box Area × 0.0005)
        ```
        
        **Risk Level Classification:**
        - 🔴 **HIGH RISK**: Risk Score > {} (Immediate collision threat)
        - 🟠 **MEDIUM**: {} < Risk Score ≤ {} (Approach with caution)
        - 🟢 **SAFE**: Risk Score ≤ {} (No immediate threat)
        
        **Parameters:**
        - **Depth Value**: Distance to vehicle (from MiDaS depth estimation)
        - **Bounding Box Area**: Size of detected vehicle
        
        **Models Used:**
        - **Detection**: YOLO (Ultralytics)
        - **Depth Estimation**: MiDaS Small (Intel)
        """.format(high_risk_threshold, medium_risk_threshold, high_risk_threshold, medium_risk_threshold))
        
        # Tips
        st.markdown("---")
        st.subheader("💡 Tips for Better Results")
        st.markdown("""
        1. **Image/Video Quality**: Use clear, well-lit images/videos for better detection
        2. **Confidence Threshold**: Lower values detect more vehicles (higher false positives)
        3. **Frame Rate**: For videos, processing every frame is slower but more accurate
        4. **GPU**: Use GPU processing for faster analysis (requires CUDA-capable GPU)
        5. **Risk Thresholds**: Adjust thresholds based on your safety requirements
        """)

if __name__ == "__main__":
    main()
