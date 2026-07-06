import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

import json
import cv2
import numpy as np


# -----------------------
# DEVICE
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# -----------------------
# DATASET
# -----------------------
class KeypointsDataset(Dataset):
    def __init__(self, img_dir, data_file):
        self.img_dir = img_dir

        with open(data_file, "r") as f:
            self.data = json.load(f)

        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        img_path = f"{self.img_dir}/{item['id']}.png"
        img = cv2.imread(img_path)

        if img is None:
            raise FileNotFoundError(img_path)

        h, w = img.shape[:2]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transforms(img)

        # FIXED BUG: item (not items)
        kps = np.array(item['kps'], dtype=np.float32).reshape(-1)

        # scale to resized image
        kps[::2] *= 224.0 / w
        kps[1::2] *= 224.0 / h

        return img, torch.tensor(kps, dtype=torch.float32)


# -----------------------
# DATA PATHS (YOUR CASE)
# -----------------------
img_dir = "/content/data/data/images"
train_json = "/content/data/data/data_train.json"
val_json = "/content/data/data/data_val.json"


# -----------------------
# DATALOADERS
# -----------------------
train_dataset = KeypointsDataset(img_dir, train_json)
val_dataset = KeypointsDataset(img_dir, val_json)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)


# -----------------------
# MODEL
# -----------------------
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, 14 * 2)
model = model.to(device)


# -----------------------
# LOSS + OPTIMIZER
# -----------------------
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# -----------------------
# TRAINING LOOP
# -----------------------
epochs = 20

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for i, (imgs, kps) in enumerate(train_loader):
        imgs = imgs.to(device)
        kps = kps.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)

        loss = criterion(outputs, kps)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 10 == 0:
            print(f"Epoch {epoch} | Iter {i} | Loss {loss.item():.4f}")

    print(f"Epoch {epoch} Avg Loss: {total_loss / len(train_loader):.4f}")


# -----------------------
# SAVE MODEL (FIXED)
# -----------------------
torch.save(model.state_dict(), "keypoints_model.pth")
print("Model saved!")