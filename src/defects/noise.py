import numpy as np


def gaussian_noise(
    image: np.ndarray,
    severity: int = 1,
) -> np.ndarray:
    stds = {
        1: 8.0,
        2: 18.0,
        3: 30.0,
    }

    if severity not in stds:
        raise ValueError("Severity must be 1, 2, or 3.")

    noise = np.random.normal(
        0,
        stds[severity],
        image.shape,
    )

    result = image.astype(np.float32) + noise

    return np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)


def poisson_noise(
    image: np.ndarray,
    severity: int = 1,
) -> np.ndarray:
    scales = {
        1: 30.0,
        2: 15.0,
        3: 7.0,
    }

    if severity not in scales:
        raise ValueError("Severity must be 1, 2, or 3.")

    image_float = image.astype(np.float32) / 255.0
    scale = scales[severity]

    noisy = np.random.poisson(
        image_float * scale
    ) / scale

    noisy = np.clip(
        noisy,
        0,
        1,
    )

    return (noisy * 255).astype(np.uint8)


def salt_and_pepper_noise(
    image: np.ndarray,
    severity: int = 1,
) -> np.ndarray:
    amounts = {
        1: 0.001,
        2: 0.005,
        3: 0.01,
    }

    if severity not in amounts:
        raise ValueError("Severity must be 1, 2, or 3.")

    result = image.copy()

    amount = amounts[severity]

    random_values = np.random.random(
        image.shape[:2]
    )

    result[
        random_values < amount / 2
    ] = 0

    result[
        random_values > 1 - amount / 2
    ] = 255

    return result


def apply_noise(
    image: np.ndarray,
    severity: int = 1,
    noise_type: str = "gaussian",
) -> np.ndarray:
    if noise_type == "gaussian":
        return gaussian_noise(image, severity)

    if noise_type == "poisson":
        return poisson_noise(image, severity)

    if noise_type == "salt_pepper":
        return salt_and_pepper_noise(image, severity)

    raise ValueError(
        f"Unknown noise type: {noise_type}"
    )