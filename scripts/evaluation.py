import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

HR_PATH = "results/esrgan/hr_pretrained.png"
SR_PATH = "results/esrgan/sr_pretrained.png"
LR_PATH = "results/esrgan/lr_pretrained.png"

hr = np.array(
    Image.open(HR_PATH).convert("RGB")
)

sr = np.array(
    Image.open(SR_PATH).convert("RGB")
)

lr = Image.open(LR_PATH).convert("RGB")

# Bicubic reconstruction: 16x16 -> 64x64
bicubic = np.array(
    lr.resize(
        (64, 64),
        Image.Resampling.BICUBIC
    )
)

# PSNR
bicubic_psnr = peak_signal_noise_ratio(
    hr,
    bicubic,
    data_range=255
)

sr_psnr = peak_signal_noise_ratio(
    hr,
    sr,
    data_range=255
)

# SSIM
bicubic_ssim = structural_similarity(
    hr,
    bicubic,
    channel_axis=2,
    data_range=255
)

sr_ssim = structural_similarity(
    hr,
    sr,
    channel_axis=2,
    data_range=255
)

print("\n===== PRETRAINED GENERATOR EVALUATION =====")

print("\nBicubic:")
print("PSNR:", round(bicubic_psnr, 4), "dB")
print("SSIM:", round(bicubic_ssim, 4))

print("\nPretrained Generator:")
print("PSNR:", round(sr_psnr, 4), "dB")
print("SSIM:", round(sr_ssim, 4))

print("\n===== COMPARISON =====")

print(
    "PSNR difference:",
    round(sr_psnr - bicubic_psnr, 4),
    "dB"
)

print(
    "SSIM difference:",
    round(sr_ssim - bicubic_ssim, 4)
)

if sr_psnr > bicubic_psnr:
    print("\nPretrained Generator has higher PSNR than Bicubic.")
else:
    print("\nBicubic has higher PSNR than Pretrained Generator.")

if sr_ssim > bicubic_ssim:
    print("Pretrained Generator has higher SSIM than Bicubic.")
else:
    print("Bicubic has higher SSIM than Pretrained Generator.")