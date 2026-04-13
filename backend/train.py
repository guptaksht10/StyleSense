import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

from model import build_model

# ✅ Confirm folders exist
print("data/train exists:", os.path.exists("data/train"))
print("data/test exists:", os.path.exists("data/test"))

# ---------------- SETTINGS ----------------
BATCH_SIZE = 32
EPOCHS = 15
LR = 0.001

# ---------------- TRANSFORMS ----------------
# ✅ Added normalization (ImageNet stats) + augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),                              # ← add
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),        # ← add
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ---------------- DATA ----------------
train_data = datasets.ImageFolder("data/train", transform=train_transform)
test_data  = datasets.ImageFolder("data/test",  transform=test_transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE)

# ✅ Confirm class order — must match app.py class_names
print("Class order:", train_data.classes)
# Expected: ['dress', 'jeans', 'shirt', 'shoes']

# ---------------- MODEL ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = build_model().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=LR)  # ✅ only train classifier

# ---------------- TRAINING LOOP ----------------
best_acc = 0.0

for epoch in range(EPOCHS):
    # --- Train ---
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # --- Validate ---
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {total_loss:.4f} — Val Accuracy: {acc:.1f}%")

    # ✅ Save best model only
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "model.pth")
        print(f"  ✅ Saved best model (acc: {acc:.1f}%)")

print(f"\nTraining complete. Best accuracy: {best_acc:.1f}%")

# ✅ paste this at the very bottom of train.py, after your existing loop

print("\n🔥 Phase 2: Fine-tuning full model...")

# Unfreeze everything
for param in model.parameters():
    param.requires_grad = True

# Lower LR to avoid destroying pretrained weights
optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(5):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            _, predicted = torch.max(model(images), 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f"Fine-tune {epoch+1}/5 — Loss: {total_loss:.4f} — Acc: {acc:.1f}%")
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "model.pth")
        print(f"  ✅ Saved ({acc:.1f}%)")

print(f"\n🏆 Final best accuracy: {best_acc:.1f}%")