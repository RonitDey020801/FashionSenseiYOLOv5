# train_attr_model.py
import pandas as pd
import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm

# -----------------------------
# MODIFY attribute_names here
# -----------------------------
colors_list  = ["black", "white", "red", "blue", "green", "yellow", "purple", "brown"]
article_list = ["tshirt", "shirt", "jeans", "shorts", "dress", "jacket", "sweatshirt", "kurta", "top"]
attribute_names = colors_list + article_list

NUM_ATTRIBUTES = len(attribute_names)

# -----------------------------

class FashionAttrDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        # Exclude only the filename column
        self.attr_cols = self.df.columns.tolist()[1:]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['image_name'])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(row[self.attr_cols].values.astype('float32'))
        return image, labels

# Training configs
BATCH_SIZE    = 8
NUM_EPOCHS    = 15
LEARNING_RATE = 1e-4

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = FashionAttrDataset(
    csv_file='labels.csv',
    image_dir='images',
    transform=transform
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = timm.create_model('efficientnet_b0', pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features, NUM_ATTRIBUTES)

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss = {total_loss / len(loader):.4f}")

torch.save(model.state_dict(), 'attr_model_weights.pth')
print("Saved model as attr_model_weights.pth")