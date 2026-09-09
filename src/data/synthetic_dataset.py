from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

LABEL_COLUMNS = [
    "blur",
    "noise",
    "underexposure",
    "overexposure",
    "compression",
]


class SyntheticPhotoDataset(Dataset):
    def __init__(
        self,
        split: str,
        transform=None,
    ):
        self.split = split

        if split == "train":
            image_dir = PROCESSED_DIR / "synthetic"
            labels_file = (
                METADATA_DIR / "synthetic_labels.csv"
            )
        elif split == "val":
            image_dir = PROCESSED_DIR / "synthetic_val"
            labels_file = (
                METADATA_DIR / "synthetic_val_labels.csv"
            )
        elif split == "test":
            image_dir = PROCESSED_DIR / "synthetic_test"
            labels_file = (
                METADATA_DIR / "synthetic_test_labels.csv"
            )
        else:
            raise ValueError(
                "Split must be train, val, or test."
            )

        self.image_dir = image_dir
        self.metadata = pd.read_csv(labels_file)

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        row = self.metadata.iloc[index]

        image_path = (
            self.image_dir / row["filename"]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        image = self.transform(image)

        labels = torch.tensor(
            row[LABEL_COLUMNS].values.astype("float32")
        )

        return {
            "image": image,
            "labels": labels,
            "image_id": row["synthetic_id"],
            "source_id": row["source_id"],
        }


def create_datasets():
    train_dataset = SyntheticPhotoDataset("train")
    val_dataset = SyntheticPhotoDataset("val")
    test_dataset = SyntheticPhotoDataset("test")

    return (
        train_dataset,
        val_dataset,
        test_dataset,
    )