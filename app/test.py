from pathlib import Path

from app.inference import PhotoQualityPredictor


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "baseline"
    / "checkpoints"
    / "best_model.pt"
)

IMAGE_PATH = PROJECT_ROOT / "data" / "processed" / "synthetic" /"139_04.jpg"


predictor = PhotoQualityPredictor(str(MODEL_PATH))

results = predictor.predict(str(IMAGE_PATH))

for defect, result in results.items():
    print(
        f"{defect:15s} "
        f"{result['probability']:.3f} "
        f"{'YES' if result['detected'] else 'NO'}"
    )
