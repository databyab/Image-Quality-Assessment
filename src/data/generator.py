from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image

from src.defects.pipeline import apply_defects


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "synthetic"

SPLITS_FILE = METADATA_DIR / "splits.csv"
LABELS_FILE = METADATA_DIR / "synthetic_labels.csv"

SEED = 42
VARIANTS_PER_IMAGE = 5

DEFECTS = [
    "blur",
    "noise",
    "underexposure",
    "overexposure",
    "compression",
]


def load_training_images() -> pd.DataFrame:
    metadata = pd.read_csv(SPLITS_FILE)

    train = metadata[
        metadata["split"] == "train"
    ].reset_index(drop=True)

    if train.empty:
        raise ValueError("No training images found.")

    return train


def sample_defects(rng: random.Random, variant: int) -> dict:
    labels = {
        "blur": 0,
        "noise": 0,
        "underexposure": 0,
        "overexposure": 0,
        "compression": 0,
    }

    severities = {
        "blur_severity": 0,
        "noise_severity": 0,
        "underexposure_severity": 0,
        "overexposure_severity": 0,
        "compression_severity": 0,
    }

    if variant == 0:
        return {**labels, **severities}

    num_defects = rng.randint(1, 3)

    selected = rng.sample(
        DEFECTS,
        num_defects,
    )

    if "underexposure" in selected and "overexposure" in selected:
        selected.remove("overexposure")

    for defect in selected:
        labels[defect] = 1
        severities[f"{defect}_severity"] = rng.randint(1, 3)

    return {**labels, **severities}


def generate_variant(
    image: np.ndarray,
    defect_config: dict,
    rng: random.Random,
) -> np.ndarray:

    blur_type = rng.choice(
        ["gaussian", "motion", "defocus"]
    )

    noise_type = rng.choice(
        ["gaussian", "poisson", "salt_pepper"]
    )

    return apply_defects(
        image,
        blur=(
            defect_config["blur_severity"]
            if defect_config["blur"]
            else None
        ),
        noise=(
            defect_config["noise_severity"]
            if defect_config["noise"]
            else None
        ),
        underexposure=(
            defect_config["underexposure_severity"]
            if defect_config["underexposure"]
            else None
        ),
        overexposure=(
            defect_config["overexposure_severity"]
            if defect_config["overexposure"]
            else None
        ),
        compression=(
            defect_config["compression_severity"]
            if defect_config["compression"]
            else None
        ),
        blur_type=blur_type,
        noise_type=noise_type,
    )


def generate_dataset() -> None:
    rng = random.Random(SEED)
    np.random.seed(SEED)

    metadata = load_training_images()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    total = len(metadata) * VARIANTS_PER_IMAGE
    generated = 0

    print(f"Training images: {len(metadata)}")
    print(f"Variants per image: {VARIANTS_PER_IMAGE}")
    print(f"Total synthetic images: {total}")
    print()

    for _, row in metadata.iterrows():

        source_id = row["image_id"]
        source_filename = row["filename"]

        image_path = RAW_DIR / source_filename

        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        for variant in range(VARIANTS_PER_IMAGE):

            defect_config = sample_defects(
                rng,
                variant,
            )

            synthetic_image = generate_variant(
                image,
                defect_config,
                rng,
            )

            synthetic_id = (
                f"{source_id}_{variant:02d}"
            )

            filename = (
                f"{synthetic_id}.jpg"
            )

            output_path = OUTPUT_DIR / filename

            Image.fromarray(
                synthetic_image
            ).save(
                output_path,
                quality=95,
            )

            records.append(
                {
                    "synthetic_id": synthetic_id,
                    "source_id": source_id,
                    "source_filename": source_filename,
                    "filename": filename,
                    "split": "train",
                    **defect_config,
                }
            )

            generated += 1

        if generated % 500 == 0:
            print(
                f"Generated {generated}/{total}"
            )

    labels = pd.DataFrame(records)

    labels.to_csv(
        LABELS_FILE,
        index=False,
    )

    print()
    print("Synthetic dataset generated.")
    print(f"Images: {generated}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Labels: {LABELS_FILE}")


if __name__ == "__main__":
    generate_dataset()