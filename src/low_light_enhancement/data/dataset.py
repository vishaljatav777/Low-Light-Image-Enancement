"""Image-folder dataset for unpaired low-light training."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class LowLightDataset(Dataset):
    def __init__(self, folder: str | Path, image_size: int = 256) -> None:
        self.folder = Path(folder)
        if not self.folder.is_dir():
            raise FileNotFoundError(f"Image folder does not exist: {self.folder}")
        self.paths = sorted(p for p in self.folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if not self.paths:
            raise ValueError(f"No supported images found in {self.folder}")
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            array = np.asarray(image, dtype=np.float32) / 255.0
            return torch.from_numpy(array).permute(2, 0, 1)
