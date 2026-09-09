# PhotoIQ

PhotoIQ is a Python project for detecting common technical photo-quality defects with a convolutional neural network. The current model predicts five labels independently:

- blur
- noise
- underexposure
- overexposure
- compression

The project includes synthetic data generation, model training, evaluation artifacts, and a Streamlit demonstration app.

## Project layout

- `app/`: Streamlit demo and checkpoint-based inference.
- `configs/`: experiment configuration files.
- `data/metadata/`: dataset split and synthetic-label CSV files.
- `data/raw/`: source images used by the data pipeline.
- `data/processed/`: generated and processed image data.
- `experiments/baseline/`: baseline checkpoint, metrics, plots, and experiment notes.
- `src/data/`: dataset loaders and synthetic-data generation.
- `src/defects/`: image defect transformations.
- `src/models/`: the `PhotoQualityCNN` model.
- `src/training/`: training code.
- `src/evaluation/`: evaluation modules.
- `src/visualization/`: feature-map and Grad-CAM utilities.

## Model and preprocessing

The model is defined in `src/models/cnn.py`. It contains four convolutional feature blocks, batch normalization, ReLU activations, max pooling in the first three blocks, adaptive average pooling, and a linear classifier with five outputs.

Images are converted to RGB, resized to 224 by 224 pixels, and converted to tensors. Inference applies a sigmoid to each output and reports a defect as detected when its probability is at least 0.5.

## Data

The synthetic dataset uses five defect labels. The generator creates five variants per training image with a fixed random seed of 42. Each non-original variant can contain one to three selected defects, with underexposure and overexposure prevented from being selected together. Defect severity values range from 1 to 3.

The recorded metadata contains:

- 17,500 training images in `data/metadata/synthetic_labels.csv`.
- 3,750 validation images in `data/metadata/synthetic_val_labels.csv`.
- 3,750 test images in `data/metadata/synthetic_test_labels.csv`.

## Installation

Install the pinned dependencies from the repository root:

```powershell
python -m pip install -r requirements.txt
```

## Training

The baseline training entry point is:

```powershell
python -m src.training.train
```

The training module uses a batch size of 32, a learning rate of `0.001`, 20 epochs, the Adam optimizer, and `BCEWithLogitsLoss`. It uses CUDA when available and otherwise uses the CPU.

## Demo application

The Streamlit application loads `experiments/baseline/checkpoints/best_model.pt` and accepts JPEG and PNG uploads:

```powershell
streamlit run app/demo.py
```

Uploads are limited to 10 MB and 25,000,000 pixels before inference. Temporary inference files are removed after prediction.

## Baseline results

The recorded baseline metrics are documented in [experiments/baseline/README.md](experiments/baseline/README.md).
