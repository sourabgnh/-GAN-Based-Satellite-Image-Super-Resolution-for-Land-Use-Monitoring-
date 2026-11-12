import os
import random
import shutil
from torchvision.datasets import EuroSAT

# -----------------------------
# Settings
# -----------------------------
SOURCE = "dataset/original"
OUTPUT = "dataset"

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

SEED = 42

random.seed(SEED)

# -----------------------------
# Load EuroSAT
# -----------------------------
dataset = EuroSAT(
    root=SOURCE,
    download=False
)

print("Total images:", len(dataset))
print("Classes:", dataset.classes)

# -----------------------------
# Create folders
# -----------------------------
for split in ["train", "validation", "test"]:
    for class_name in dataset.classes:
        os.makedirs(
            os.path.join(OUTPUT, split, class_name),
            exist_ok=True
        )

# -----------------------------
# Group images by class
# -----------------------------
class_images = {}

for index in range(len(dataset)):
    _, label = dataset[index]

    class_name = dataset.classes[label]

    if class_name not in class_images:
        class_images[class_name] = []

    class_images[class_name].append(index)

# -----------------------------
# Split each class
# -----------------------------
for class_name, indices in class_images.items():

    random.shuffle(indices)

    total = len(indices)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    print(
        f"{class_name}: "
        f"Train={len(train_indices)}, "
        f"Validation={len(val_indices)}, "
        f"Test={len(test_indices)}"
    )

    # -------------------------
    # Copy images
    # -------------------------
    for split, split_indices in [
        ("train", train_indices),
        ("validation", val_indices),
        ("test", test_indices)
    ]:

        for index in split_indices:

            image_path, _ = dataset.samples[index]

            filename = os.path.basename(image_path)

            destination = os.path.join(
                OUTPUT,
                split,
                class_name,
                filename
            )

            shutil.copy2(image_path, destination)

print("\nDataset splitting completed successfully!")