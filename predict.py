import os
import sys
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

# Reduce TensorFlow messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = r"models\best_xception.keras"
IMG_SIZE = (299, 299)

# ==========================================
# CHECK ARGUMENTS
# ==========================================

if len(sys.argv) < 2:
    print("\nERROR: No image path provided.")
    print("\nUsage:")
    print(r'  python predict.py "path\to\image.jpg"')
    sys.exit(1)

IMAGE_PATH = sys.argv[1]

if not os.path.isfile(IMAGE_PATH):
    print("\nERROR: Image not found.")
    print("Path:", IMAGE_PATH)
    sys.exit(1)

if not os.path.isfile(MODEL_PATH):
    print("\nERROR: Model not found.")
    print("Expected model:", MODEL_PATH)
    sys.exit(1)

# ==========================================
# LOAD MODEL & PREDICT
# ==========================================

print("\nLoading Universal AI & Deepfake Detector...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# Load Image
pil_img = Image.open(IMAGE_PATH).convert("RGB")
img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

# Face Detection
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

if len(faces) > 0:
    faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces_sorted[0]
    pad_w = int(w * 0.15)
    pad_h = int(h * 0.15)
    img_h, img_w = img_cv.shape[:2]
    x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
    x2, y2 = min(img_w, x + w + pad_w), min(img_h, y + h + pad_h)
    eval_img = pil_img.crop((x1, y1, x2, y2))
    region_info = f"Detected Face cropped ({x2-x1}x{y2-y1} px)"
else:
    eval_img = pil_img
    region_info = f"Full Image ({pil_img.size[0]}x{pil_img.size[1]} px)"

print(f"Region analyzed: {region_info}")

# Preprocess for Xception
img_resized = eval_img.resize(IMG_SIZE)
image_array = np.expand_dims(np.array(img_resized, dtype=np.float32), axis=0)
image_array = tf.keras.applications.xception.preprocess_input(image_array)

# Prediction
prediction = model.predict(image_array, verbose=0)
real_probability = float(prediction[0][0])
fake_probability = 1.0 - real_probability

result = "REAL" if real_probability >= 0.5 else "FAKE"

# Reconfigure stdout for UTF-8 on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ==========================================
# DISPLAY RESULT
# ==========================================

print("\n" + "=" * 50)
print("       UNIVERSAL AI & DEEPFAKE PREDICTION")
print("=" * 50)
print(f"Image            : {IMAGE_PATH}")
print(f"Region           : {region_info}")
print(f"Real probability : {real_probability * 100:.2f}%")
print(f"Fake probability : {fake_probability * 100:.2f}%")
print(f"Prediction       : {'[+] REAL (Authentic)' if result == 'REAL' else '[-] FAKE (AI Generated / Deepfake)'}")
print("=" * 50 + "\n")