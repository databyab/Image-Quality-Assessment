# Baseline Experiment

This directory contains the recorded baseline experiment artifacts for PhotoIQ.

## Training setup

The training implementation is `src/training/train.py`. It trains `PhotoQualityCNN` on the synthetic training split and validates on the synthetic validation split.

Recorded training settings:

- Batch size: 32
- Learning rate: `0.001`
- Epochs: 20
- Optimizer: Adam
- Loss: `BCEWithLogitsLoss`
- DataLoader workers: 0
- Device: CUDA when available, otherwise CPU

The model predicts the following labels independently: blur, noise, underexposure, overexposure, and compression.

## Dataset records

The synthetic metadata files contain:

- Training: 17,500 images
- Validation: 3,750 images
- Test: 3,750 images

The synthetic generator uses random seed 42 and creates five variants per source training image. The generated transformations cover blur, noise, underexposure, overexposure, and JPEG compression.

## Recorded overall metrics

The values below are recorded in `metrics/overall_metrics.csv`.

| Metric | Value |
| --- | ---: |
| Best epoch | 19 |
| Best validation loss | 0.1214138066 |
| Macro F1 | 0.9208054351 |
| Micro F1 | 0.9260665221 |
| Mean PR AUC | 0.9707379206 |

## Recorded test metrics

The values below are recorded in `metrics/test_metrics.csv`.

| Defect | Precision | Recall | F1 | PR AUC |
| --- | ---: | ---: | ---: | ---: |
| Blur | 0.9389121339 | 0.9532710280 | 0.9460370995 | 0.9868303678 |
| Noise | 0.9709864603 | 0.8290668869 | 0.8944320713 | 0.9643748234 |
| Underexposure | 0.9369442199 | 0.9982773471 | 0.9666388657 | 0.9804972917 |
| Overexposure | 0.8912466844 | 0.8245398773 | 0.8565965583 | 0.9371436595 |
| Compression | 0.9261318507 | 0.9549549550 | 0.9403225806 | 0.9848434604 |

## Artifacts

- `checkpoints/best_model.pt`: saved model checkpoint used by the demo application.
- `metrics/overall_metrics.csv`: aggregate baseline metrics.
- `metrics/test_metrics.csv`: per-defect test metrics.
- `plots/`: experiment plots, when generated.

The recorded metrics describe this baseline run on the repository's synthetic dataset. They do not establish performance on an external or real-world dataset.
