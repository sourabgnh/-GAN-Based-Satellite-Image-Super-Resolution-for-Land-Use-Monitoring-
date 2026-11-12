from PIL import Image
import os

HR_SIZE = 64
LR_SIZE = 16

input_folder = "dataset/train/AnnualCrop"
output_folder = "results/bicubic"

os.makedirs(output_folder, exist_ok=True)

files = [
    f for f in os.listdir(input_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

test_files = files[:10]

processed_count = 0

for filename in test_files:

    input_path = os.path.join(input_folder, filename)

    image = Image.open(input_path).convert("RGB")

    # HR image: 64 x 64
    hr_image = image.resize(
        (HR_SIZE, HR_SIZE),
        Image.Resampling.BICUBIC
    )

    # LR image: 16 x 16
    lr_image = hr_image.resize(
        (LR_SIZE, LR_SIZE),
        Image.Resampling.BICUBIC
    )

    # Bicubic reconstruction: 16 x 16 -> 64 x 64
    bicubic_image = lr_image.resize(
        (HR_SIZE, HR_SIZE),
        Image.Resampling.BICUBIC
    )

    output_name = "bicubic_" + filename
    output_path = os.path.join(output_folder, output_name)

    bicubic_image.save(output_path)

    processed_count += 1

    print("Processed:", filename)

print("\nBicubic test completed!")
print("Images created:", processed_count)