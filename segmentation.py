import cv2
import numpy as np
import os
from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")

input_dir  = r"C:\Users\Ashwin Hari\Desktop\AI based Triage\dataset_4\images"
output_dir = r"C:\Users\Ashwin Hari\Desktop\AI based Triage\humans_silhouettes"

os.makedirs(output_dir, exist_ok=True)

for img_name in os.listdir(input_dir):

    img_path = os.path.join(input_dir, img_name)
    image = cv2.imread(img_path)

    if image is None:
        continue

    results = model(image)

    # Check if mask exists
    if results[0].masks is None:
        continue

    mask = results[0].masks.data[0].cpu().numpy()
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    mask = (mask > 0.5).astype(np.uint8)

    output = image.copy()
    output[mask == 1] = [0, 0, 0]
    output[mask == 0] = [255, 255, 255]
    save_path = os.path.join(output_dir, img_name)
    cv2.imwrite(save_path, output)

print("Silhouette images saved.")