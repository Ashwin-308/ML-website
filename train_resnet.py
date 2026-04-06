import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader,random_split
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

data_dir = r"C:\Users\Ashwin Hari\Desktop\AI based Triage\silhouettes_dataset"
batch_size = 32
epochs = 10
LR = 0.0001

all_preds = []
all_labels = []
all_probs = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset =  datasets.ImageFolder(data_dir,transform = transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset,val_dataset = random_split(dataset,[train_size,val_size])


train_targets = [train_dataset[i][1] for i in range(len(train_dataset))]
class_counts = np.bincount(train_targets)
class_weights = 1.0 / class_counts
sample_weights = [class_weights[label] for label in train_targets]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False)

model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = True
model.fc = nn.Linear(model.fc.in_features,4)
model = model.to(device)
model.eval()


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(model.layer4.parameters()) + list(model.fc.parameters()), lr=1e-5)

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for images,labels in train_loader:
        images,labels = images.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    model.eval()
    correct = 0
    total = 0
   
    with torch.no_grad():
        for images,labels in val_loader:
            images,labels = images.to(device),labels.to(device)
            outputs = model(images)
            _,predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {train_loss:.4f} Val Accuracy: {accuracy:.2f}%")
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)

        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
all_probs = np.array(all_probs)
all_labels = np.array(all_labels)
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)    
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, digits=4))
torch.save(model.state_dict(),"resnet_posture.pth")
print("Model saved")