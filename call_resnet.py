import os
import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader,random_split
from torch.utils.data import WeightedRandomSampler
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image



all_preds = []
all_labels = []
all_probs = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])




model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features,4)
model.load_state_dict(torch.load("resnet_posture.pth", map_location=device))
model.to(device)
model.eval()


def classify(img):
    image = img.convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)

    return predicted.item()




  