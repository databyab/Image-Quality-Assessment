"""Copy a folder of photos into PhotoIQ defect-label folders."""
import argparse
import csv
import json
import shutil
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.inference import LABEL_COLUMNS, PhotoQualityPredictor

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
REVIEW_FOLDER = "Needs review"
NO_DEFECT_FOLDER = "No detected defects"


def _unique_destination(folder: Path, filename: str) -> Path:
    candidate = folder / filename
    stem, suffix = candidate.stem, candidate.suffix
    counter = 2
    while candidate.exists():
        candidate = folder / f"{stem} ({counter}){suffix}"
        counter += 1
    return candidate


def sort_folder(input_dir: Path, output_dir: Path, checkpoint_path: Path,
                uncertainty_band: float = 0.10) -> dict:
    """Copy images into every matching defect folder; borderline scores go to review."""
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    checkpoint_path = checkpoint_path.expanduser().resolve()

    if not input_dir.is_dir():
        raise ValueError(f"Input folder does not exist: {input_dir}")
    if input_dir == output_dir or input_dir in output_dir.parents:
        raise ValueError("Output folder must be outside the input folder.")
    if not checkpoint_path.is_file():
        raise ValueError(f"Model checkpoint does not exist: {checkpoint_path}")
    if not 0 <= uncertainty_band <= 0.5:
        raise ValueError("Uncertainty band must be between 0 and 0.5.")

    images = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError("No JPEG or PNG images were found in the input folder.")

    predictor = PhotoQualityPredictor(str(checkpoint_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {label: 0 for label in LABEL_COLUMNS}
    counts[NO_DEFECT_FOLDER] = 0
    counts[REVIEW_FOLDER] = 0
    rows = []
    skipped = 0

    for image_path in images:
        relative_path = image_path.relative_to(input_dir)
        try:
            with Image.open(image_path) as image:
                image.verify()
            results = predictor.predict(str(image_path))
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as exc:
            skipped += 1
            rows.append({"source": str(relative_path), "folders": "",
                         "status": f"skipped: {exc}", "probabilities": "{}"})
            continue

        detected = [label for label, result in results.items() if result["detected"]]
        uncertain = any(abs(result["probability"] - 0.5) <= uncertainty_band
                        for result in results.values())
        destinations = detected.copy()
        if uncertain:
            destinations.append(REVIEW_FOLDER)
        elif not destinations:
            destinations.append(NO_DEFECT_FOLDER)

        for label in destinations:
            folder = output_dir / label
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, _unique_destination(folder, image_path.name))
            counts[label] += 1

        rows.append({
            "source": str(relative_path),
            "folders": "; ".join(destinations),
            "status": "review" if uncertain else "sorted",
            "probabilities": json.dumps(
                {key: round(value["probability"], 4) for key, value in results.items()},
                sort_keys=True,
            ),
        })

    manifest = output_dir / "photoiq_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["source", "folders", "status", "probabilities"])
        writer.writeheader()
        writer.writerows(rows)

    return {"processed": len(images) - skipped, "skipped": skipped,
            "counts": counts, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description="Sort photos into PhotoIQ defect folders.")
    parser.add_argument("input_folder", type=Path, help="Folder containing photos")
    parser.add_argument("output_folder", type=Path, help="Folder for labeled copies")
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).resolve().parents[1]
                        / "experiments" / "baseline" / "checkpoints" / "best_model.pt")
    parser.add_argument("--uncertainty-band", type=float, default=0.10,
                        help="Scores within this distance of 0.5 are also copied to Needs review")
    args = parser.parse_args()

    try:
        summary = sort_folder(args.input_folder, args.output_folder, args.checkpoint,
                              args.uncertainty_band)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))

    print(f"Processed: {summary['processed']}; skipped: {summary['skipped']}")
    for label, count in summary["counts"].items():
        print(f"{label}: {count}")
    print(f"Manifest: {summary['manifest']}")


if __name__ == "__main__":
    main()

