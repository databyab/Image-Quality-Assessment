from pathlib import Path
import random

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
OUTPUT_FILE = METADATA_DIR / "splits.csv"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


def get_images() -> list[Path]:
    """Find supported images in the raw dataset directory."""

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw dataset directory does not exist: {RAW_DIR}"
        )

    return sorted(
        path
        for path in RAW_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def split_images(
    images: list[Path],
) -> tuple[list[Path], list[Path], list[Path]]:
    """Split images into train, validation and test sets."""

    if not images:
        raise ValueError("No images were found.")

    if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-6:
        raise ValueError(
            "TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0"
        )

    rng = random.Random(SEED)

    shuffled_images = images.copy()
    rng.shuffle(shuffled_images)

    total_images = len(shuffled_images)

    train_end = int(total_images * TRAIN_RATIO)
    val_end = train_end + int(total_images * VAL_RATIO)

    return (
        shuffled_images[:train_end],
        shuffled_images[train_end:val_end],
        shuffled_images[val_end:],
    )


def create_metadata(
    train_images: list[Path],
    val_images: list[Path],
    test_images: list[Path],
) -> pd.DataFrame:
    """Create metadata for the dataset split."""

    records = []

    split_groups = {
        "train": train_images,
        "val": val_images,
        "test": test_images,
    }

    for split, images in split_groups.items():
        for image_path in images:
            records.append(
                {
                    "image_id": image_path.stem,
                    "filename": image_path.name,
                    "split": split,
                    "source_dataset": "COCO-2017-Val",
                }
            )

    return pd.DataFrame(records)


def save_metadata(metadata: pd.DataFrame) -> None:
    """Save metadata to CSV."""

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(OUTPUT_FILE, index=False)


def main() -> None:
    print("PhotoIQ Dataset Splitter")
    print("-" * 30)

    images = get_images()

    print(f"Images found: {len(images)}")

    if not images:
        raise ValueError(f"No supported images found in {RAW_DIR}")

    train_images, val_images, test_images = split_images(images)

    print(f"Train: {len(train_images)}")
    print(f"Validation: {len(val_images)}")
    print(f"Test: {len(test_images)}")

    metadata = create_metadata(
        train_images,
        val_images,
        test_images,
    )

    save_metadata(metadata)

    print(f"\nMetadata saved to: {OUTPUT_FILE}")
    print("\nSplit distribution:")
    print(metadata["split"].value_counts())


if __name__ == "__main__":
    main()
