import os

# Reduce TensorFlow messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout
)
from tensorflow.keras.models import Model

# ==========================================
# SETTINGS
# ==========================================

DATASET_PATH = r"data\deepfake_dataset"

IMG_SIZE = (299, 299)
BATCH_SIZE = 16
SEED = 42

# ==========================================
# LOAD DATASETS
# ==========================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "validation"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

print("\nClasses:", train_ds.class_names)

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
# XCEPTION BACKBONE
# ==========================================

base_model = Xception(
    weights="imagenet",
    include_top=False,
    input_shape=(299, 299, 3)
)

# Freeze Xception
base_model.trainable = False

print("\nXceptionNet loaded.")
print("Xception layers:", len(base_model.layers))
print("Xception frozen:", not base_model.trainable)

# ==========================================
# CLASSIFICATION HEAD
# ==========================================

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(
    256,
    activation="relu"
)(x)

x = Dropout(0.5)(x)

output = Dense(
    1,
    activation="sigmoid"
)(x)

model = Model(
    inputs=base_model.input,
    outputs=output
)

# ==========================================
# COMPILE
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc")
    ]
)

# ==========================================
# MODEL SUMMARY
# ==========================================

# ==========================================
# MODEL SUMMARY
# ==========================================

model.summary()


# ==========================================
# CALLBACKS
# ==========================================

os.makedirs("models", exist_ok=True)

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        filepath="models/best_xception.keras",
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=2,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=1,
        verbose=1
    )
]


# ==========================================
# TRAIN MODEL
# ==========================================

EPOCHS = 5

print("\n====================================")
print("STARTING XCEPTION TRAINING")
print("====================================")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


# ==========================================
# SAVE FINAL MODEL
# ==========================================

model.save(
    "models/xception_deepfake_final.keras"
)

print("\n====================================")
print("TRAINING COMPLETED")
print("====================================")