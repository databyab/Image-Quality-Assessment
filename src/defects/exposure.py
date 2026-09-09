import numpy as np


def underexposure(
    image: np.ndarray,
    severity: int = 1,
) -> np.ndarray:
    factors = {
        1: 0.75,
        2: 0.50,
        3: 0.30,
    }

    if severity not in factors:
        raise ValueError("Severity must be 1, 2, or 3.")

    result = image.astype(np.float32) * factors[severity]

    return np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)


def overexposure(
    image: np.ndarray,
    severity: int = 1,
) -> np.ndarray:
    factors = {
        1: 1.25,
        2: 1.60,
        3: 2.00,
    }

    if severity not in factors:
        raise ValueError("Severity must be 1, 2, or 3.")

    result = image.astype(np.float32) * factors[severity]

    return np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)


def apply_exposure(
    image: np.ndarray,
    severity: int = 1,
    exposure_type: str = "under",
) -> np.ndarray:
    if exposure_type == "under":
        return underexposure(image, severity)

    if exposure_type == "over":
        return overexposure(image, severity)

    raise ValueError(
        f"Unknown exposure type: {exposure_type}"
    )