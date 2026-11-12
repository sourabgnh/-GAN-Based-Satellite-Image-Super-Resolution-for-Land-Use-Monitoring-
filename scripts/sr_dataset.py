from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import os


class SatelliteSRDataset(Dataset):

    def __init__(self, root_dir, scale=4):

        self.root_dir = root_dir
        self.scale = scale

        self.files = [
            f for f in os.listdir(root_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        filename = self.files[index]

        path = os.path.join(
            self.root_dir,
            filename
        )

        hr = Image.open(path).convert("RGB")

        hr = TF.to_tensor(hr)

        lr_size = (
            hr.shape[1] // self.scale,
            hr.shape[2] // self.scale
        )

        lr = TF.resize(
            hr,
            lr_size,
            interpolation=TF.InterpolationMode.BICUBIC
        )

        return lr, hr