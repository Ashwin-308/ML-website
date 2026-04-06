import os
import cv2
from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolov8n.pt")

INPUT_DIR = r"C:\Users\Ashwin Hari\Desktop\AI based Triage\dataset_4\images"
OUTPUT_DIR = r"C:\Users\Ashwin Hari\Desktop\AI based Triage\humans"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for img_name in os.listdir(INPUT_DIR):

    img_path = os.path.join(INPUT_DIR, img_name)
    image = cv2.imread(img_path)

    results = model(image)

    for result in results:
        boxes = result.boxes

        for i in range(len(boxes)):
            cls = int(boxes.cls[i])

            if cls == 0:  # 0 = person in COCO
                x1, y1, x2, y2 = map(int, boxes.xyxy[i])

                crop = image[y1:y2, x1:x2]

                save_path = os.path.join(
                    OUTPUT_DIR,
                    f"{img_name.split('.')[0]}_{i}.png"
                )

                cv2.imwrite(save_path, crop)

print("Cropping completed.")