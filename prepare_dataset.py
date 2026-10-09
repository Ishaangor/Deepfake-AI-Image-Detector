import os

# Reduce TensorFlow messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

# ==========================================
# SETTINGS
# ==========================================

DATASET_PATH = r"data\deepfake_dataset"

IMG_SIZE = (299, 299)
BATCH_SIZE = 16
SEED = 42

# ==========================================
# LOAD TRAIN DATA
# ==========================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=SEED
)

# ==========================================
# LOAD VALIDATION DATA
# ==========================================

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "validation"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

# ==========================================
# LOAD TEST DATA
# ==========================================

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

# ==========================================
# CHECK CLASS NAMES
# ==========================================

print("\nClass names:")
print(train_ds.class_names)

# ==========================================
# XCEPTION PREPROCESSING
# ==========================================

preprocess_input = tf.keras.applications.xception.preprocess_input

def preprocess_images(images, labels):
    images = preprocess_input(images)
    return images, labels

train_ds = train_ds.map(
    preprocess_images,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.map(
    preprocess_images,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = test_ds.map(
    preprocess_images,
    num_parallel_calls=tf.data.AUTOTUNE
)

# ==========================================
# PREFETCH
# ==========================================

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# ==========================================
# CHECK ONE BATCH
# ==========================================

images, labels = next(iter(train_ds))

print("\n====================================")
print("DATASET CHECK")
print("====================================")

print("Image batch shape :", images.shape)
print("Label batch shape :", labels.shape)
print("Image data type   :", images.dtype)
print("Image minimum     :", tf.reduce_min(images).numpy())
print("Image maximum     :", tf.reduce_max(images).numpy())

print("\nDataset pipeline is ready!")