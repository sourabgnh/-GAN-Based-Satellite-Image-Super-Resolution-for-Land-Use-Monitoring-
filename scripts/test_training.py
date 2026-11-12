import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scripts.sr_dataset import SatelliteSRDataset
from models.esrgan.rrdbnet import RRDBNet
from models.esrgan.discriminator import Discriminator


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# Dataset
dataset = SatelliteSRDataset(
    "dataset/train/AnnualCrop"
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0
)


# Models
generator = RRDBNet(
    num_blocks=6
).to(device)

discriminator = Discriminator().to(device)


# Loss functions
pixel_loss = nn.L1Loss()

adversarial_loss = nn.BCEWithLogitsLoss()


# Optimizers
g_optimizer = torch.optim.Adam(
    generator.parameters(),
    lr=1e-4
)

d_optimizer = torch.optim.Adam(
    discriminator.parameters(),
    lr=1e-4
)


# Get one batch
lr_images, hr_images = next(iter(loader))

lr_images = lr_images.to(device)
hr_images = hr_images.to(device)


print("LR batch:", lr_images.shape)
print("HR batch:", hr_images.shape)


# -------------------------
# Generator forward pass
# -------------------------

sr_images = generator(lr_images)

print("SR batch:", sr_images.shape)


# -------------------------
# Generator loss
# -------------------------

g_pixel = pixel_loss(
    sr_images,
    hr_images
)

real_labels = torch.ones(
    hr_images.size(0),
    1,
    device=device
)

fake_predictions = discriminator(
    sr_images
)

g_adv = adversarial_loss(
    fake_predictions,
    real_labels
)

g_loss = g_pixel + 0.001 * g_adv


# -------------------------
# Generator update
# -------------------------

g_optimizer.zero_grad()

g_loss.backward()

g_optimizer.step()


# -------------------------
# Discriminator update
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
    device=device
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


print("\nTraining test completed successfully!")

print("Generator pixel loss:",
      g_pixel.item())

print("Generator total loss:",
      g_loss.item())

print("Discriminator loss:",
      d_loss.item())

if torch.cuda.is_available():
    print(
        "GPU memory allocated:",
        round(
            torch.cuda.memory_allocated() / 1024**2,
            2
        ),
        "MB"
    )