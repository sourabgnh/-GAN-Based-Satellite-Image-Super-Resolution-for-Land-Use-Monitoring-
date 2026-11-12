from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class SatelliteSRDataset(Dataset):
    """Load HR satellite images and create paired LR inputs on the fly.

    Expected directory layout:
        dataset/train/
            AnnualCrop/*.jpg
            Forest/*.jpg
            ...
    
    A single class directory is also supported for backwards compatibility.
    By default the dataset returns (lr, hr), matching the original training
    scripts. Set return_label=True to return (lr, hr, label) for later
    land-use-aware experiments.
    """

    def __init__(
        self,
        root_dir,
        scale=4,
        return_label=False,
        hr_size=None,
    ):
        self.root_dir = Path(root_dir)
        self.scale = int(scale)
        self.return_label = return_label
        self.hr_size = hr_size

        if self.scale < 1:
            raise ValueError("scale must be a positive integer.")
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")
        if not self.root_dir.is_dir():
            raise NotADirectoryError(f"Expected a directory: {self.root_dir}")

        image_paths = sorted(
            path for path in self.root_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise RuntimeError(f"No supported images found under: {self.root_dir}")

        # The first directory below root_dir is the class name. If images are
        # stored directly in root_dir, treat the folder itself as one class.
        class_names = sorted({
            path.relative_to(self.root_dir).parts[0]
            if len(path.relative_to(self.root_dir).parts) > 1
            else self.root_dir.name
            for path in image_paths
        })
        self.classes = class_names
        self.class_to_idx = {name: index for index, name in enumerate(class_names)}

        self.samples = []
        for path in image_paths:
            relative_parts = path.relative_to(self.root_dir).parts
            class_name = relative_parts[0] if len(relative_parts) > 1 else self.root_dir.name
            self.samples.append((path, self.class_to_idx[class_name]))

        self.files = [path.name for path, _ in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        with Image.open(image_path) as image:
            hr_image = image.convert("RGB")

        if self.hr_size is not None:
            if isinstance(self.hr_size, int):
                target_size = (self.hr_size, self.hr_size)
            else:
                target_size = tuple(self.hr_size)
            hr_image = hr_image.resize(target_size, Image.Resampling.BICUBIC)

        width, height = hr_image.size
        if width % self.scale != 0 or height % self.scale != 0:
            raise ValueError(
                f"Image {image_path} has size {width}x{height}, which is not "
                f"divisible by scale={self.scale}. Set hr_size or fix the image."
            )

        lr_size = (width // self.scale, height // self.scale)
        lr_image = hr_image.resize(lr_size, Image.Resampling.BICUBIC)

        hr_tensor = TF.to_tensor(hr_image)
        lr_tensor = TF.to_tensor(lr_image)

        if self.return_label:
            return lr_tensor, hr_tensor, label
        return lr_tensor, hr_tensor
