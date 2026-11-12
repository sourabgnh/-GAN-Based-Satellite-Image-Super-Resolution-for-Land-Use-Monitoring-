import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from models.esrgan.rrdbnet import RRDBNet
from scripts.sr_dataset import SatelliteSRDataset

def main():
    torch.manual_seed(42)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# Load the training dataset
train_dataset = SatelliteSRDataset(
    root_dir="dataset/train/AnnualCrop"
)

if len(train_dataset) < 500:
    raise ValueError(
        f"Expected at least 500 images, found {len(train_dataset)}"
    )

# Select 500 images reproducibly
rng = torch.Generator().manual_seed(42)
indices = torch.randperm(
    len(train_dataset), generator=rng
)[:500].tolist()

train_subset = Subset(train_dataset, indices)

train_loader = DataLoader(
    train_subset,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

print("Training images:", len(train_subset))
print("Batches per epoch:", len(train_loader))

# Initialize generator
model = RRDBNet().to(device)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

epochs = 10
os.makedirs("weights", exist_ok=True)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for batch_index, (lr_images, hr_images) in enumerate(
        train_loader, start=1
    ):
        lr_images = lr_images.to(device)
        hr_images = hr_images.to(device)

        optimizer.zero_grad(set_to_none=True)

        sr_images = model(lr_images)
        loss = criterion(sr_images, hr_images)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_index % 50 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Batch [{batch_index}/{len(train_loader)}] "
                f"Loss: {loss.item():.5f}"
            )

    average_loss = running_loss / len(train_loader)

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Average L1 Loss: {average_loss:.5f}"
    )

    torch.save(
        model.state_dict(),
        "weights/generator_500_latest.pth"
    )

torch.save(
    model.state_dict(),
    "weights/generator_pretrained_500.pth"
)

print("Pretraining complete.")
print("Final model saved to weights/generator_pretrained_500.pth")


if __name__ == "main":
    main()
