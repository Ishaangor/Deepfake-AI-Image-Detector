import os
import sys

# Reduce TensorFlow messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import tensorflow as tf
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# SETTINGS
# ==========================================

DATASET_PATH = r"data\deepfake_dataset"
MODEL_PATH = r"models\best_xception.keras"

IMG_SIZE = (299, 299)
BATCH_SIZE = 16

# ==========================================
# LOAD TEST DATASET
# ==========================================

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

class_names = test_ds.class_names
print("\nClass names:", class_names)

# ==========================================
# XCEPTION PREPROCESSING
# ==========================================

preprocess_input = tf.keras.applications.xception.preprocess_input

def preprocess_images(images, labels):
    images = preprocess_input(images)
    return images, labels

test_ds = test_ds.map(
    preprocess_images,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# ==========================================
# LOAD BEST MODEL
# ==========================================

print("\nLoading best XceptionNet model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# ==========================================
# GET PREDICTIONS
# ==========================================

print("\nGenerating predictions...")

probabilities = model.predict(
    test_ds,
    verbose=1
)

# Convert predictions to 1D
probabilities = probabilities.ravel()

# ==========================================
# GET TRUE LABELS
# ==========================================

true_labels = np.concatenate([
    labels.numpy().ravel()
    for _, labels in test_ds
])

# ==========================================
# CONVERT PROBABILITY TO CLASS
# ==========================================

predicted_labels = (
    probabilities >= 0.5
).astype(int)

# ==========================================
# CALCULATE METRICS
# ==========================================

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

precision = precision_score(
    true_labels,
    predicted_labels
)

recall = recall_score(
    true_labels,
    predicted_labels
)

f1 = f1_score(
    true_labels,
    predicted_labels
)

auc = roc_auc_score(
    true_labels,
    probabilities
)

# ==========================================
# PRINT RESULTS
# ==========================================

print("\n")
print("==========================================")
print("       XCEPTION DEEPFAKE TEST RESULTS")
print("==========================================")

print(f"Test Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Test Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Test Recall   : {recall:.4f} ({recall*100:.2f}%)")
print(f"Test F1 Score : {f1:.4f} ({f1*100:.2f}%)")
print(f"Test ROC-AUC  : {auc:.4f}")

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    true_labels,
    predicted_labels
)

print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================")

print(cm)

print("\n")
print("              Predicted")
print("              Fake   Real")
print(f"Actual Fake   {cm[0][0]:5d} {cm[0][1]:6d}")
print(f"Actual Real   {cm[1][0]:5d} {cm[1][1]:6d}")

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names
    )
)

print("\n==========================================")
print("EVALUATION COMPLETED")
print("==========================================")