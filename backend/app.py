from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import io
import numpy as np
import cv2
import os

from model import build_model

app = Flask(__name__)

# ✅ Single clean CORS setup — no @after_request conflict
CORS(app, origins=["https://style-sense-olive.vercel.app"], supports_credentials=False)

# ---------------- CLASS NAMES ----------------
class_names = ["dress", "jeans", "shirt", "shoes"]

# ---------------- LOAD MODEL ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_model()
model.load_state_dict(torch.load("model.pth", map_location=device))
model = model.to(device)
model.eval()

print("✅ Model loaded successfully")

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ---------------- COLOR DETECTION ----------------
def detect_color(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    pixels = np.float32(img.reshape(-1, 3))

    _, labels, centers = cv2.kmeans(
        pixels, 3, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10, cv2.KMEANS_RANDOM_CENTERS
    )

    counts = np.bincount(labels.flatten())
    dominant = centers[np.argmax(counts)]
    b, g, r = dominant

    if r > 200 and g > 200 and b > 200:
        return "white"
    elif r < 50 and g < 50 and b < 50:
        return "black"
    elif abs(int(r) - int(g)) < 25 and abs(int(g) - int(b)) < 25 and 50 < r < 200:
        return "gray"
    elif r > g and r > b and r > 100:
        return "red"
    elif g > r and g > b and g > 100:
        return "green"
    elif b > r and b > g and b > 100:
        return "blue"
    elif r > 150 and g > 120 and b < 80:
        return "yellow"
    elif r > 120 and g < 80 and b > 120:
        return "purple"
    else:
        return "mixed"

# ---------------- ROUTES ----------------
@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze():
    # ✅ Handle preflight request
    if request.method == "OPTIONS":
        return jsonify({}), 200

    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")

        # Category prediction
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probs, 0)

        category = class_names[predicted.item()]
        color = detect_color(image)

        return jsonify({
            "category": category,
            "confidence": round(float(confidence.item()), 3),
            "color": color
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- HEALTH CHECK ----------------
@app.route("/")
def home():
    return "✅ Backend is running"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)