import os
import shutil
import random

# ==========================================
# PATHS
# ==========================================

SOURCE = r"data\rvf10k"
OUTPUT = r"data\deepfake_dataset"

# ==========================================
# SETTINGS
# ==========================================

SEED = 42
random.seed(SEED)

# ==========================================
# CREATE FOLDERS
# ==========================================

splits = ["train", "validation", "test"]
classes = ["real", "fake"]

for split in splits:
    for cls in classes:
        os.makedirs(
            os.path.join(OUTPUT, split, cls),
            exist_ok=True
        )

# ==========================================
# 1. COPY EXISTING TRAIN → FINAL TRAIN
# ==========================================

for cls in classes:

    source_folder = os.path.join(
        SOURCE, "train", cls
    )

    destination_folder = os.path.join(
        OUTPUT, "train", cls
    )

    images = [
        f for f in os.listdir(source_folder)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        )
    ]

    print(f"\n{cls.upper()} TRAIN IMAGES: {len(images)}")

    for image in images:

        shutil.copy2(
            os.path.join(source_folder, image),
            os.path.join(destination_folder, image)
        )

# ==========================================
# 2. SPLIT EXISTING VALID INTO
#    VALIDATION + TEST
# ==========================================

for cls in classes:

    source_folder = os.path.join(
        SOURCE, "valid", cls
    )

    images = [
        f for f in os.listdir(source_folder)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        )
    ]

    random.shuffle(images)

    midpoint = len(images) // 2

    validation_images = images[:midpoint]
    test_images = images[midpoint:]

    print(f"\n{cls.upper()}")
    print("Total:", len(images))
    print("Validation:", len(validation_images))
    print("Test:", len(test_images))

    # --------------------------------------
    # VALIDATION
    # --------------------------------------

    validation_folder = os.path.join(
        OUTPUT, "validation", cls
    )

    for image in validation_images:

        shutil.copy2(
            os.path.join(source_folder, image),
            os.path.join(validation_folder, image)
        )

    # --------------------------------------
    # TEST
    # --------------------------------------

    test_folder = os.path.join(
        OUTPUT, "test", cls
    )

    for image in test_images:

        shutil.copy2(
            os.path.join(source_folder, image),
            os.path.join(test_folder, image)
        )

# ==========================================
# DONE
# ==========================================

print("\n====================================")
print("DATASET CREATION COMPLETED")
print("====================================")

print("\nDataset location:")
print(OUTPUT)