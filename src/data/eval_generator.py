from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image

from src.defects.pipeline import apply_defects


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SPLITS_FILE = METADATA_DIR / "splits.csv"

SEED = 42
VARIANTS_PER_IMAGE = 5

DEFECTS = [
    "blur",
    "noise",
    "underexposure",
    "overexposure",
    "compression",
]


def load_split(split: str) -> pd.DataFrame:
    metadata = pd.read_csv(SPLITS_FILE)

    data = metadata[
        metadata["split"] == split
    ].reset_index(drop=True)

    if data.empty:
        raise ValueError(
            f"No images found for split: {split}"
        )

    return data


def sample_defects(
    rng: random.Random,
    variant: int,
) -> dict:
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

    if (
        "underexposure" in selected
        and "overexposure" in selected
    ):
        selected.remove("overexposure")

    for defect in selected:
        labels[defect] = 1
        severities[
            f"{defect}_severity"
        ] = rng.randint(1, 3)

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


def generate_eval_split(split: str) -> None:
    rng = random.Random(SEED)
    np.random.seed(SEED)

    metadata = load_split(split)

    output_dir = (
        PROCESSED_DIR / f"synthetic_{split}"
    )

    labels_file = (
        METADATA_DIR / f"synthetic_{split}_labels.csv"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    total = (
        len(metadata) * VARIANTS_PER_IMAGE
    )

    generated = 0

    print(f"Split: {split}")
    print(f"Source images: {len(metadata)}")
    print(
        f"Variants per image: {VARIANTS_PER_IMAGE}"
    )
    print(f"Total images: {total}")
    print()

    for _, row in metadata.iterrows():

        source_id = row["image_id"]
        source_filename = row["filename"]

        image_path = RAW_DIR / source_filename

        image = Image.open(
            image_path
        ).convert("RGB")

        image = np.array(image)

        for variant in range(
            VARIANTS_PER_IMAGE
        ):
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

            output_path = (
                output_dir / filename
            )

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
                    "split": split,
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
        labels_file,
        index=False,
    )

    print()
    print(
        f"{split} evaluation dataset generated."
    )
    print(f"Images: {generated}")
    print(f"Output: {output_dir}")
    print(f"Labels: {labels_file}")


def main() -> None:
    generate_eval_split("val")
    print()
    generate_eval_split("test")


if __name__ == "__main__":
    main()