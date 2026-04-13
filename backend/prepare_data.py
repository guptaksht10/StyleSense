import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------- LOAD CSV ----------------
df = pd.read_csv("images.csv")
df["label"] = df["label"].str.strip()

# ---------------- LABEL MAPPING ----------------
label_map = {
    "T-Shirt": "shirt", "Shirt": "shirt", "Polo": "shirt",
    "Longsleeve": "shirt", "Undershirt": "shirt", "Top": "shirt",
    "Blouse": "shirt", "Hoodie": "shirt", "Jacket": "shirt", "Sweater": "shirt",
    "Pants": "jeans", "Shorts": "jeans", "Jeans": "jeans",
    "Dress": "dress", "Skirt": "dress",
    "Shoes": "shoes", "Sneakers": "shoes", "Sandals": "shoes",
}

df["label"] = df["label"].map(label_map)
df = df.dropna(subset=["label"])

print(f"Rows after mapping: {len(df)}")
print(df["label"].value_counts())

# ---------------- BALANCE DATA ----------------
min_count = df["label"].value_counts().min()
df = df.groupby("label").sample(n=min_count, random_state=42)

print("\nBalanced dataset:")
print(df["label"].value_counts())

# ---------------- TRAIN / TEST SPLIT ----------------
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)

# ---------------- IMAGE FOLDER ----------------
image_folder = "images_original"
if not os.path.exists(image_folder):
    image_folder = "images_compressed"

print(f"\nUsing image folder: {image_folder}")

# ---------------- COPY FILES ----------------
copied = 0
missing = 0

for split, split_df in [("train", train_df), ("test", test_df)]:
    for _, row in split_df.iterrows():
        label = row["label"]
        filename = str(row["image"]).strip()

        src = None
        for ext in ["", ".jpg", ".jpeg", ".png", ".webp"]:
            candidate = os.path.join(image_folder, filename + ext)
            if os.path.exists(candidate):
                src = candidate
                break

        dst_dir = os.path.join("data", split, label)
        os.makedirs(dst_dir, exist_ok=True)

        if src:
            try:
                shutil.copy(src, dst_dir)
                copied += 1
            except Exception as e:
                print(f"Copy error: {filename} → {e}")
        else:
            missing += 1

print(f"\nDone! Copied: {copied} | Missing: {missing}")

# ---------------- VERIFY ----------------
print("\nFinal folder counts:")
for split in ["train", "test"]:
    for cls in ["shirt", "jeans", "dress", "shoes"]:
        path = os.path.join("data", split, cls)
        count = len(os.listdir(path)) if os.path.exists(path) else 0
        print(f"  {split}/{cls}: {count} images")