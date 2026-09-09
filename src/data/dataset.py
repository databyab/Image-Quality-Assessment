from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
METADATA_FILE = PROJECT_ROOT / "data" / "metadata" / "splits.csv"

IMAGE_SIZE = 224


class PhotoDataset(Dataset):
    def __init__(
        self,
        split: str,
        transform=None,
    ):
        self.split = split

        self.metadata = pd.read_csv(METADATA_FILE)
        self.metadata = self.metadata[
            self.metadata["split"] == split
        ].reset_index(drop=True)

        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
            ]
        )

        if len(self.metadata) == 0:
            raise ValueError(
                f"No images found for split: {split}"
            )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        row = self.metadata.iloc[index]

        image_path = RAW_DIR / row["filename"]

        image = Image.open(image_path).convert("RGB")

        image = self.transform(image)

        return {
            "image": image,
            "image_id": row["image_id"],
            "filename": row["filename"],
        }


def create_datasets():
    train_dataset = PhotoDataset("train")
    val_dataset = PhotoDataset("val")
    test_dataset = PhotoDataset("test")

    return train_dataset, val_dataset, test_dataset