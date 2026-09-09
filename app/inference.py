from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.models.cnn import PhotoQualityCNN


LABEL_COLUMNS = [
    "blur",
    "noise",
    "underexposure",
    "overexposure",
    "compression",
]


class PhotoQualityPredictor:
    def __init__(self, checkpoint_path: str):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = PhotoQualityCNN(num_classes=5)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def predict(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = torch.sigmoid(logits)[0]

        results = {}

        for label, probability in zip(
            LABEL_COLUMNS,
            probabilities.cpu().tolist(),
        ):
            results[label] = {
                "probability": probability,
                "detected": probability >= 0.5,
            }

        return results