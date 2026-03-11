import cv2
from matplotlib import image
import numpy as np
import os
from ultralytics import YOLO
from call_resnet import classify
from PIL import Image
model = YOLO("yolov8n-seg.pt")


def callyolo(img):
        results = model(img)
        if results[0].masks is None:
                return "No human detected"
        mask = results[0].masks.data[0].cpu().numpy()
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        mask = (mask > 0.5).astype(np.uint8)
        output = img.copy()
        output[mask == 1] = [0, 0, 0]
        output[mask == 0] = [255, 255, 255]
        output = Image.fromarray(output)
        return classify(output)
        
        

