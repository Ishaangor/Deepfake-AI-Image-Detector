import os
import glob
import random
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# Reduce TF logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 60)
print("TRAINING UNIVERSAL AI & DEEPFAKE DETECTOR")
print("=" * 60)

# Paths
AI_FACES_DIR = r"data\ai_faces\AI-face-detection-Dataset\AI"
AI_REAL_DIR = r"data\ai_faces\AI-face-detection-Dataset\real"
RVF_FAKE_DIR = r"data\deepfake_dataset\train\fake"
RVF_REAL_DIR = r"data\deepfake_dataset\train\real"

# Collect file paths
ai_fake_files = glob.glob(os.path.join(AI_FACES_DIR, "*.jpg"))
rvf_fake_files = glob.glob(os.path.join(RVF_FAKE_DIR, "*.jpg"))

ai_real_files = glob.glob(os.path.join(AI_REAL_DIR, "*.jpg"))
rvf_real_files = glob.glob(os.path.join(RVF_REAL_DIR, "*.jpg"))

print(f"Collected modern AI fake images: {len(ai_fake_files)}")
print(f"Collected StyleGAN fake images: {len(rvf_fake_files)}")
print(f"Collected modern real images: {len(ai_real_files)}")
print(f"Collected classic real images: {len(rvf_real_files)}")

# Balance the dataset:
# Fake = 0, Real = 1 (matching project standard)
# We take all 1,001 AI fakes + 1,500 RVF fakes = 2,501 fakes
# We take 1,250 modern real + 1,250 RVF real = 2,500 real
random.shuffle(ai_fake_files)
random.shuffle(rvf_fake_files)
random.shuffle(ai_real_files)
random.shuffle(rvf_real_files)

selected_fakes = ai_fake_files + rvf_fake_files[:1500]
selected_reals = ai_real_files[:1250] + rvf_real_files[:1250]

all_paths = selected_fakes + selected_reals
all_labels = [0] * len(selected_fakes) + [1] * len(selected_reals)

# Split train / val / test
train_paths, test_paths, train_labels, test_labels = train_test_split(
    all_paths, all_labels, test_size=0.15, random_state=SEED, stratify=all_labels
)
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_paths, train_labels, test_size=0.15, random_state=SEED, stratify=train_labels
)

print(f"\nDataset splits:")
print(f"  Train: {len(train_paths)} (Fakes: {train_labels.count(0)}, Reals: {train_labels.count(1)})")
print(f"  Val:   {len(val_paths)} (Fakes: {val_labels.count(0)}, Reals: {val_labels.count(1)})")
print(f"  Test:  {len(test_paths)} (Fakes: {test_labels.count(0)}, Reals: {test_labels.count(1)})")

# Feature Extractor using Xception backbone
base_model = tf.keras.applications.Xception(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(299, 299, 3)
)
base_model.trainable = False

def extract_features_batch(paths, batch_size=32):
    features = []
    total = len(paths)
    for i in range(0, total, batch_size):
        batch_paths = paths[i:i+batch_size]
        batch_imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB").resize((299, 299))
            arr = np.array(img, dtype=np.float32)
            arr = tf.keras.applications.xception.preprocess_input(arr)
            batch_imgs.append(arr)
        batch_arr = np.array(batch_imgs)
        feats = base_model.predict(batch_arr, verbose=0)
        features.append(feats)
        if (i // batch_size) % 20 == 0 or i + batch_size >= total:
            print(f"  Progress: {min(i+batch_size, total)}/{total} images processed")
    return np.vstack(features)

print("\nExtracting Xception features for training set...")
X_train = extract_features_batch(train_paths)
y_train = np.array(train_labels, dtype=np.float32)

print("\nExtracting Xception features for validation set...")
X_val = extract_features_batch(val_paths)
y_val = np.array(val_labels, dtype=np.float32)

print("\nExtracting Xception features for test set...")
X_test = extract_features_batch(test_paths)
y_test = np.array(test_labels, dtype=np.float32)

# Build Classification Head
classifier = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2048,)),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

classifier.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        verbose=1
    )
]

print("\nTraining Deep Classifier Head...")
history = classifier.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=35,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

# Test evaluation
y_pred_probs = classifier.predict(X_test, verbose=0).ravel()
y_pred = (y_pred_probs >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_probs)

print("\n" + "=" * 60)
print(f"TEST RESULTS: Accuracy = {acc*100:.2f}%, ROC-AUC = {auc:.4f}")
print("=" * 60)
print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))

# Combine base_model + classifier into a single end-to-end Keras model!
inputs = tf.keras.Input(shape=(299, 299, 3))
x = base_model(inputs)
outputs = classifier(x)
full_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="universal_xception_detector")

os.makedirs("models", exist_ok=True)
full_model_path = r"models\best_universal_detector.keras"
full_model.save(full_model_path)
print(f"\nFull end-to-end model saved to: {full_model_path}")

# Verify on sample images
print("\n" + "=" * 60)
print("VERIFICATION ON CHALLENGE IMAGES:")
print("=" * 60)

test_suite = [
    r"C:\Users\DELL\Desktop\Fakers.jpg",
    r"C:\Users\DELL\Desktop\fake_20.png",
    r"C:\Users\DELL\Desktop\real_20.png",
    r"test_images\face_real.jpg",
    r"test_images\face_fake.jpg",
]

for p in test_suite:
    if os.path.exists(p):
        img = Image.open(p).convert("RGB").resize((299, 299))
        arr = np.expand_dims(np.array(img, dtype=np.float32), 0)
        arr = tf.keras.applications.xception.preprocess_input(arr)
        pred = float(full_model.predict(arr, verbose=0)[0][0])
        label = "REAL" if pred >= 0.5 else "FAKE"
        print(f"  {os.path.basename(p):<25} -> Real: {pred*100:5.1f}% | Fake: {(1-pred)*100:5.1f}% | Pred: {label}")
