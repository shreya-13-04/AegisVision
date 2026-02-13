from ultralytics import YOLO
import cv2
import torch
import os

########################################
# PATH SETUP
########################################

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "best.pt")

image_name = input("Enter image name (example: test1.jpg): ")

IMAGE_PATH = os.path.join(BASE_DIR, "..", "test", image_name)

if not os.path.exists(IMAGE_PATH):
    print("❌ Image not found.")
    exit()

########################################
# LOAD MODELS ONCE
########################################

print("Loading YOLO...")
yolo = YOLO(MODEL_PATH)

print("Loading MiDaS depth model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(device)
midas.eval()

transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform

print("✅ Models loaded.\n")

########################################
# READ IMAGE
########################################

frame = cv2.imread(IMAGE_PATH)

frame = cv2.resize(frame, (960,540))

########################################
# YOLO DETECTION
########################################

results = yolo(frame, conf=0.25)

########################################
# DEPTH ESTIMATION
########################################

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
input_batch = transform(rgb).to(device)

with torch.no_grad():

    prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

depth_map = prediction.cpu().numpy()

########################################
# SMART COLLISION SCORING
########################################

for r in results:

    boxes = r.boxes.xyxy.cpu().numpy()

    for box in boxes:

        x1, y1, x2, y2 = map(int, box)

        cx = int((x1+x2)/2)
        cy = int((y1+y2)/2)

        depth_value = depth_map[cy, cx]
        box_area = (x2-x1)*(y2-y1)

        risk_score = depth_value*0.6 + box_area*0.0005

        if risk_score > 170:
            color = (0,0,255)
            label = "HIGH RISK"
        elif risk_score > 130:
            color = (0,165,255)
            label = "MEDIUM"
        else:
            color = (0,255,0)
            label = "SAFE"

        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

        cv2.putText(
            frame,
            f"{label}",
            (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

########################################
# SAVE OUTPUT (VERY IMPORTANT)
########################################

output_path = os.path.join(BASE_DIR, "..", "test", "output_"+image_name)

cv2.imwrite(output_path, frame)

print(f"✅ Output saved at: {output_path}")

cv2.imshow("Collision Result", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
