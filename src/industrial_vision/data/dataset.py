import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

DEFAULT_IMAGE_SIZE = (640, 224)


class KSDD2Dataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        manifest_path: Path,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.image_size = image_size

        with Path(manifest_path).open(newline="") as file:
            self.samples = list(csv.DictReader(file))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image_path = self.dataset_root / sample["image"]
        mask_path = self.dataset_root / sample["mask"]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = TF.resize(
                image,
                self.image_size,
                interpolation=InterpolationMode.BILINEAR,
            )
            image = TF.to_tensor(image)

        with Image.open(mask_path) as mask:
            mask = mask.convert("L")
            mask = TF.resize(
                mask,
                self.image_size,
                interpolation=InterpolationMode.NEAREST,
            )
            mask = (TF.to_tensor(mask) > 0).float()

        label = int(sample["label"])
        return image, mask, label
