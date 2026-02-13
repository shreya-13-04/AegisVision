import torch
import cv2
import numpy as np

# Load MiDaS
model_type = "MiDaS_small"  # fastest

midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

# transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

transform = midas_transforms.small_transform


def get_depth_map(image_path):

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # normalize for visualization
    depth_map = cv2.normalize(
        depth_map,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    depth_map = depth_map.astype(np.uint8)

    return depth_map
