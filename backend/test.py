import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from model import build_model

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ---------------- DATA ----------------
test_data   = datasets.ImageFolder("data/test", transform=transform)
test_loader = DataLoader(test_data, batch_size=32)

class_names = test_data.classes
print("Classes:", class_names)

# ---------------- LOAD MODEL ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_model()
model.load_state_dict(torch.load("model.pth", map_location=device))
model = model.to(device)
model.eval()

# ---------------- EVALUATE ----------------
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# ---------------- REPORT ----------------
print("\n📊 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

print("🔢 Confusion Matrix:")
cm = confusion_matrix(all_labels, all_preds)
print(cm)

# Per-class accuracy
print("\n✅ Per-class accuracy:")
for i, cls in enumerate(class_names):
    correct = cm[i][i]
    total = cm[i].sum()
    print(f"  {cls}: {correct}/{total} ({100*correct/total:.1f}%)")