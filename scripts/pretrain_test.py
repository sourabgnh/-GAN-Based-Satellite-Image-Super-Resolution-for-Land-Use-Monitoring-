import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from scripts.sr_dataset import SatelliteSRDataset
from models.esrgan.rrdbnet import RRDBNet


# =========================
# SETTINGS
# =========================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 2
EPOCHS = 3
TRAIN_IMAGES = 100
LR = 1e-4

WEIGHTS_DIR = "weights"

os.makedirs(WEIGHTS_DIR, exist_ok=True)


# =========================
# DEVICE
# =========================

print("Device:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# =========================
# DATASET
# =========================

full_dataset = SatelliteSRDataset(
    "dataset/train/AnnualCrop"
)

train_dataset = Subset(
    full_dataset,
    range(TRAIN_IMAGES)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

print(
    "Training images:",
    len(train_dataset)
)


# =========================
# GENERATOR
# =========================

generator = RRDBNet(
    num_blocks=6
).to(DEVICE)


# =========================
# LOSS
# =========================

criterion = nn.L1Loss()


# =========================
# OPTIMIZER
# =========================

optimizer = torch.optim.Adam(
    generator.parameters(),
    lr=LR
)


# =========================
# TRAINING
# =========================

for epoch in range(EPOCHS):

    generator.train()

    total_loss = 0.0

    print(
        "\nEpoch:",
        epoch + 1,
        "/",
        EPOCHS
    )

    for batch_index, (lr_images, hr_images) in enumerate(train_loader):

        lr_images = lr_images.to(DEVICE)
        hr_images = hr_images.to(DEVICE)

        # Generate SR image
        sr_images = generator(
            lr_images
        )

        # Reconstruction loss
        loss = criterion(
            sr_images,
            hr_images
        )

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        print(
            "Batch",
            batch_index + 1,
            "/",
            len(train_loader),
            "| Loss:",
            round(loss.item(), 4)
        )

    average_loss = (
        total_loss /
        len(train_loader)
    )

    print(
        "Average Loss:",
        round(average_loss, 4)
    )


# =========================
# SAVE GENERATOR
# =========================

output_path = os.path.join(
    WEIGHTS_DIR,
    "generator_pretrained_test.pth"
)

torch.save(
    generator.state_dict(),
    output_path
)

print(
    "\nPretraining completed!"
)

print(
    "Generator saved:",
    output_path
)