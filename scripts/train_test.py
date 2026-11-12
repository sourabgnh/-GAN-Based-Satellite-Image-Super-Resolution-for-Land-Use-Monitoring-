import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from scripts.sr_dataset import SatelliteSRDataset
from models.esrgan.rrdbnet import RRDBNet
from models.esrgan.discriminator import Discriminator


# =========================
# SETTINGS
# =========================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 2
EPOCHS = 1
TRAIN_IMAGES = 100
VAL_IMAGES = 20

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

val_dataset = Subset(
    full_dataset,
    range(TRAIN_IMAGES, TRAIN_IMAGES + VAL_IMAGES)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# =========================
# MODELS
# =========================

generator = RRDBNet(
    num_blocks=6
).to(DEVICE)

discriminator = Discriminator().to(DEVICE)


# =========================
# LOSSES
# =========================

pixel_loss = nn.L1Loss()

adversarial_loss = nn.BCEWithLogitsLoss()


# =========================
# OPTIMIZERS
# =========================

g_optimizer = torch.optim.Adam(
    generator.parameters(),
    lr=LR
)

d_optimizer = torch.optim.Adam(
    discriminator.parameters(),
    lr=LR
)


# =========================
# TRAINING
# =========================

for epoch in range(EPOCHS):

    generator.train()
    discriminator.train()

    total_g_loss = 0.0
    total_d_loss = 0.0

    print("\nEpoch:", epoch + 1)

    for batch_index, (lr_images, hr_images) in enumerate(train_loader):

        lr_images = lr_images.to(DEVICE)
        hr_images = hr_images.to(DEVICE)

        # -------------------------
        # GENERATOR
        # -------------------------

        sr_images = generator(lr_images)

        g_pixel = pixel_loss(
            sr_images,
            hr_images
        )

        real_labels = torch.ones(
            hr_images.size(0),
            1,
            device=DEVICE
        )

        fake_predictions = discriminator(
            sr_images
        )

        g_adv = adversarial_loss(
            fake_predictions,
            real_labels
        )

        g_loss = (
            g_pixel +
            0.001 * g_adv
        )

        g_optimizer.zero_grad()

        g_loss.backward()

        g_optimizer.step()

        # -------------------------
        # DISCRIMINATOR
        # -------------------------

        real_predictions = discriminator(
            hr_images
        )

        fake_predictions = discriminator(
            sr_images.detach()
        )

        fake_labels = torch.zeros(
            hr_images.size(0),
            1,
            device=DEVICE
        )

        d_real_loss = adversarial_loss(
            real_predictions,
            real_labels
        )

        d_fake_loss = adversarial_loss(
            fake_predictions,
            fake_labels
        )

        d_loss = (
            d_real_loss +
            d_fake_loss
        ) * 0.5

        d_optimizer.zero_grad()

        d_loss.backward()

        d_optimizer.step()

        total_g_loss += g_loss.item()
        total_d_loss += d_loss.item()

        print(
            "Batch",
            batch_index + 1,
            "/",
            len(train_loader),
            "| G Loss:",
            round(g_loss.item(), 4),
            "| D Loss:",
            round(d_loss.item(), 4)
        )

    average_g_loss = (
        total_g_loss / len(train_loader)
    )

    average_d_loss = (
        total_d_loss / len(train_loader)
    )

    print("\nEpoch completed")

    print(
        "Average Generator Loss:",
        round(average_g_loss, 4)
    )

    print(
        "Average Discriminator Loss:",
        round(average_d_loss, 4)
    )


# =========================
# VALIDATION
# =========================

generator.eval()

validation_loss = 0.0

with torch.no_grad():

    for lr_images, hr_images in val_loader:

        lr_images = lr_images.to(DEVICE)
        hr_images = hr_images.to(DEVICE)

        sr_images = generator(lr_images)

        loss = pixel_loss(
            sr_images,
            hr_images
        )

        validation_loss += loss.item()


validation_loss /= len(val_loader)

print(
    "\nValidation Pixel Loss:",
    round(validation_loss, 4)
)


# =========================
# SAVE MODELS
# =========================

torch.save(
    generator.state_dict(),
    os.path.join(
        WEIGHTS_DIR,
        "generator_test.pth"
    )
)

torch.save(
    discriminator.state_dict(),
    os.path.join(
        WEIGHTS_DIR,
        "discriminator_test.pth"
    )
)

print("\nModels saved successfully!")

print(
    "Generator:",
    "weights/generator_test.pth"
)

print(
    "Discriminator:",
    "weights/discriminator_test.pth"
)