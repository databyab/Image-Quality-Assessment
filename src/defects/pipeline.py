from typing import Optional

import numpy as np

from .blur import apply_blur
from .compression import jpeg_compression
from .exposure import apply_exposure
from .noise import apply_noise


def apply_defects(
    image: np.ndarray,
    blur: Optional[int] = None,
    noise: Optional[int] = None,
    underexposure: Optional[int] = None,
    overexposure: Optional[int] = None,
    compression: Optional[int] = None,
    blur_type: str = "gaussian",
    noise_type: str = "gaussian",
) -> np.ndarray:
    result = image.copy()

    if blur is not None:
        result = apply_blur(
            result,
            severity=blur,
            blur_type=blur_type,
        )

    if noise is not None:
        result = apply_noise(
            result,
            severity=noise,
            noise_type=noise_type,
        )

    if underexposure is not None:
        result = apply_exposure(
            result,
            severity=underexposure,
            exposure_type="under",
        )

    if overexposure is not None:
        result = apply_exposure(
            result,
            severity=overexposure,
            exposure_type="over",
        )

    if compression is not None:
        result = jpeg_compression(
            result,
            severity=compression,
        )

    return result