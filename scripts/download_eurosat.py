from torchvision.datasets import EuroSAT

print("Downloading EuroSAT RGB dataset...")
print("Please wait...")

dataset = EuroSAT(
    root="dataset/original",
    download=True
)

print("\nDownload completed successfully!")
print("Total images:", len(dataset))
print("Number of classes:", len(dataset.classes))
print("Classes:")

for i, class_name in enumerate(dataset.classes):
    print(i, "-", class_name)