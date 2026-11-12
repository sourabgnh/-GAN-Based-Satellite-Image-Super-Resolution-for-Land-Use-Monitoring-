import os
import torch
from PIL import Image
import torchvision.transforms.functional as TF

from models.esrgan.rrdbnet import RRDBNet


# =========================
# SETTINGS
# =========================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

INPUT_IMAGE = "dataset/train/AnnualCrop"

OUTPUT_DIR = "results/esrgan"

WEIGHTS = "weights/generator_test.pth"


os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# LOAD MODEL
# =========================

generator = RRDBNet(
    num_blocks=6
).to(DEVICE)

generator.load_state_dict(
    torch.load(
        WEIGHTS,
        map_location=DEVICE,
        weights_only=True
    )
)

generator.eval()


print("Model loaded successfully.")
print("Device:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# =========================
# SELECT IMAGE
# =========================

files = [
    f for f in os.listdir(INPUT_IMAGE)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
]

filename = files[0]

input_path = os.path.join(
    INPUT_IMAGE,
    filename
)


print("Input image:", filename)


# =========================
# LOAD HR IMAGE
# =========================

hr_image = Image.open(
    input_path
).convert("RGB")

print(
    "Original HR size:",
    hr_image.size
)


# =========================
# CREATE LR IMAGE
# =========================

lr_image = hr_image.resize(
    (16, 16),
    Image.Resampling.BICUBIC
)

print(
    "LR size:",
    lr_image.size
)


# =========================
# CONVERT TO TENSOR
# =========================

lr_tensor = TF.to_tensor(
    lr_image
).unsqueeze(0).to(DEVICE)


# =========================
# GENERATE SR IMAGE
# =========================

with torch.no_grad():

    sr_tensor = generator(
        lr_tensor
    )

    sr_tensor = torch.clamp(
        sr_tensor,
        0,
        1
    )


# =========================
# CONVERT SR TO IMAGE
# =========================

sr_image = TF.to_pil_image(
    sr_tensor.squeeze(0).cpu()
)


print(
    "SR size:",
    sr_image.size
)


# =========================
# SAVE IMAGES
# =========================

base_name = os.path.splitext(
    filename
)[0]


lr_path = os.path.join(
    OUTPUT_DIR,
    "lr_" + base_name + ".png"
)

sr_path = os.path.join(
    OUTPUT_DIR,
    "sr_" + base_name + ".png"
)

hr_path = os.path.join(
    OUTPUT_DIR,
    "hr_" + base_name + ".png"
)


lr_image.save(lr_path)

sr_image.save(sr_path)

hr_image.save(hr_path)


print("\nImages saved successfully!")

print("LR:", lr_path)
print("SR:", sr_path)
print("HR:", hr_path)