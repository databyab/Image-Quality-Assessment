import cv2
import numpy as np


def gaussian_blur(image: np.ndarray, severity: int = 1) -> np.ndarray:
    kernels = {
        1: 5,
        2: 11,
        3: 21,
    }

    if severity not in kernels:
        raise ValueError("Severity must be 1, 2, or 3.")

    kernel_size = kernels[severity]

    return cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        0,
    )


def motion_blur(
    image: np.ndarray,
    severity: int = 1,
    angle: float = 0.0,
) -> np.ndarray:
    sizes = {
        1: 7,
        2: 15,
        3: 25,
    }

    if severity not in sizes:
        raise ValueError("Severity must be 1, 2, or 3.")

    size = sizes[severity]

    kernel = np.zeros(
        (size, size),
        dtype=np.float32,
    )

    kernel[size // 2, :] = 1.0 / size

    center = (size // 2, size // 2)

    rotation_matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0,
    )

    kernel = cv2.warpAffine(
        kernel,
        rotation_matrix,
        (size, size),
    )

    kernel_sum = kernel.sum()

    if kernel_sum > 0:
        kernel /= kernel_sum

    return cv2.filter2D(
        image,
        -1,
        kernel,
    )


def defocus_blur(
    image: np.ndarray,
    severity: int = 1,
) -> np.ndarray:
    radii = {
        1: 3,
        2: 7,
        3: 11,
    }

    if severity not in radii:
        raise ValueError("Severity must be 1, 2, or 3.")

    radius = radii[severity]

    y, x = np.ogrid[
        -radius:radius + 1,
        -radius:radius + 1,
    ]

    kernel = (
        x**2 + y**2 <= radius**2
    ).astype(np.float32)

    kernel /= kernel.sum()

    return cv2.filter2D(
        image,
        -1,
        kernel,
    )


def apply_blur(
    image: np.ndarray,
    severity: int = 1,
    blur_type: str = "gaussian",
) -> np.ndarray:
    if blur_type == "gaussian":
        return gaussian_blur(image, severity)

    if blur_type == "motion":
        return motion_blur(image, severity)

    if blur_type == "defocus":
        return defocus_blur(image, severity)

    raise ValueError(
        f"Unknown blur type: {blur_type}"
    )