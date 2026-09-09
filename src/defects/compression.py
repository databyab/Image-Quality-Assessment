import cv2
import numpy as np


def jpeg_compression(
    image: np.ndarray,
    severity: int = 1,
) -> np.ndarray:
    qualities = {
        1: 60,
        2: 30,
        3: 10,
    }

    if severity not in qualities:
        raise ValueError("Severity must be 1, 2, or 3.")

    image_bgr = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR,
    )

    success, encoded = cv2.imencode(
        ".jpg",
        image_bgr,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            qualities[severity],
        ],
    )

    if not success:
        raise RuntimeError("JPEG encoding failed.")

    decoded = cv2.imdecode(
        encoded,
        cv2.IMREAD_COLOR,
    )

    if decoded is None:
        raise RuntimeError("JPEG decoding failed.")

    return cv2.cvtColor(
        decoded,
        cv2.COLOR_BGR2RGB,
    )