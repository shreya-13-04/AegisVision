from ultralytics import YOLO
import cv2
import os
from depth_model import get_depth_map
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "best.pt")
IMAGE_PATH = os.path.join(BASE_DIR, "..", "test_images", "test1.jpg")

model = YOLO(MODEL_PATH)

results = model.predict(source=IMAGE_PATH, conf=0.25)

img = cv2.imread(IMAGE_PATH)

depth_map = get_depth_map(IMAGE_PATH)

for r in results:

    boxes = r.boxes.xyxy.cpu().numpy()

    for box in boxes:

        x1, y1, x2, y2 = map(int, box)

        # center of bounding box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        depth_value = depth_map[cy, cx]

        print("Depth:", depth_value)

        # Collision heuristic
        if depth_value > 180:
            color = (0,0,255)  # RED
            label = "WARNING"
        else:
            color = (0,255,0)

        cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
        cv2.putText(img,str(depth_value),(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

cv2.imshow("Collision System", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
