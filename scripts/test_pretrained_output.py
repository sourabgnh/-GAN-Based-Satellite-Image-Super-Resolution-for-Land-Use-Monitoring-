import torch
from PIL import Image
import torchvision.transforms.functional as TF

from models.esrgan.rrdbnet import RRDBNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_PATH = "dataset/train/AnnualCrop/AnnualCrop_1.jpg"

LR_PATH = "results/esrgan/lr_pretrained.png"
SR_PATH = "results/esrgan/sr_pretrained.png"
HR_PATH = "results/esrgan/hr_pretrained.png"

# Load generator
generator = RRDBNet(num_blocks=6).to(DEVICE)

generator.load_state_dict(
    torch.load(
        "weights/generator_pretrained_test.pth",
        map_location=DEVICE
    )
)

generator.eval()

print("Model loaded")
print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# Load HR image
hr_image = Image.open(INPUT_PATH).convert("RGB")

# Create LR image
lr_image = hr_image.resize(
    (16, 16),
    Image.Resampling.BICUBIC
)

# Convert LR to tensor
lr_tensor = TF.to_tensor(lr_image).unsqueeze(0).to(DEVICE)

# Generate SR
with torch.no_grad():
    sr_tensor = generator(lr_tensor)

# Convert SR tensor to image
sr_tensor = torch.clamp(sr_tensor, 0, 1)

sr_image = TF.to_pil_image(
    sr_tensor.squeeze(0).cpu()
)

# Save images
hr_image.save(HR_PATH)
lr_image.save(LR_PATH)
sr_image.save(SR_PATH)

print("\nImages saved:")
print("LR:", LR_PATH)
print("SR:", SR_PATH)
print("HR:", HR_PATH)

print("\nImage sizes:")
print("LR:", lr_image.size)
print("SR:", sr_image.size)
print("HR:", hr_image.size)