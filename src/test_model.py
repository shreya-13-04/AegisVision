from ultralytics import YOLO
import cv2
import os

# Get current file directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build absolute paths (VERY IMPORTANT)
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "best.pt")
IMAGE_PATH = os.path.join(BASE_DIR, "..", "test_images", "test1.jpg")

# Load model
model = YOLO(MODEL_PATH)

# Run prediction
results = model.predict(
    source=IMAGE_PATH,
    conf=0.25,
    save=True
)

print("✅ Detection completed!")
