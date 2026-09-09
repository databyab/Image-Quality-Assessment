from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.synthetic_dataset import SyntheticPhotoDataset
from src.models.cnn import PhotoQualityCNN


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "baseline"
    / "checkpoints"
)

METRICS_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "baseline"
    / "metrics"
)

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 20

NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def create_dataloaders():
    train_dataset = SyntheticPhotoDataset("train")
    val_dataset = SyntheticPhotoDataset("val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
):
    model.train()

    total_loss = 0.0

    for batch in loader:
        images = batch["image"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()
        optimizer.step()

        total_loss += (
            loss.item() * images.size(0)
        )

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
):
    model.eval()

    total_loss = 0.0

    for batch in loader:
        images = batch["image"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        total_loss += (
            loss.item() * images.size(0)
        )

    return total_loss / len(loader.dataset)


def save_checkpoint(
    model,
    optimizer,
    epoch,
    val_loss,
):
    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
    }

    path = (
        CHECKPOINT_DIR
        / "best_model.pt"
    )

    torch.save(
        checkpoint,
        path,
    )


def save_metrics(history):
    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        METRICS_DIR
        / "training_history.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            history,
            file,
            indent=2,
        )


def main():
    print("PhotoIQ CNN Training")
    print("-" * 30)
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")
    print()

    train_loader, val_loader = (
        create_dataloaders()
    )

    model = PhotoQualityCNN().to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        val_loss = validate(
            model,
            val_loader,
            criterion,
        )

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
            )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

    save_metrics(history)

    print()
    print("Training complete.")
    print(
        f"Best validation loss: "
        f"{best_val_loss:.4f}"
    )
    print(
        "Checkpoint: "
        f"{CHECKPOINT_DIR / 'best_model.pt'}"
    )
    print(
        "Metrics: "
        f"{METRICS_DIR / 'training_history.json'}"
    )


if __name__ == "__main__":
    main()