from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError


MIN_DIMENSION = 96
MIN_SHARPNESS = 18.0
MIN_CONTRAST = 8.0
MIN_COLOR_STD = 4.0
MIN_BRIGHTNESS = 18.0
MAX_BRIGHTNESS = 238.0


def analyze_image_quality(path: Path) -> dict[str, Any]:
    reasons: list[str] = []
    try:
        image = Image.open(path).convert("RGB")
    except (OSError, UnidentifiedImageError):
        return _result(path, 0, 0, 0.0, 0.0, 0.0, 0.0, False, ["unreadable"])

    width, height = image.size
    if width < MIN_DIMENSION or height < MIN_DIMENSION:
        reasons.append("too_small")

    small = image.resize((min(width, 320), min(height, 320)))
    rgb = np.asarray(small, dtype=np.float32)
    gray = np.asarray(small.convert("L"), dtype=np.float32)

    brightness = float(gray.mean())
    contrast = float(gray.std())
    color_std = float(rgb.std(axis=(0, 1)).mean())
    sharpness = _laplacian_variance(gray)

    if brightness < MIN_BRIGHTNESS:
        reasons.append("too_dark")
    if brightness > MAX_BRIGHTNESS:
        reasons.append("too_bright")
    if contrast < MIN_CONTRAST:
        reasons.append("low_contrast")
    if color_std < MIN_COLOR_STD:
        reasons.append("near_solid_color")
    if sharpness < MIN_SHARPNESS:
        reasons.append("blurry")

    score = _quality_score(sharpness, contrast, color_std, brightness)
    return _result(path, width, height, sharpness, brightness, contrast, color_std, not reasons, reasons, score)


def image_similarity(path_a: Path, path_b: Path) -> float:
    try:
        a = _fingerprint(path_a)
        b = _fingerprint(path_b)
    except (OSError, UnidentifiedImageError):
        return 0.0
    return max(0.0, min(1.0, 1.0 - float(np.mean(np.abs(a - b)))))


def _fingerprint(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L").resize((16, 16))
    values = np.asarray(image, dtype=np.float32) / 255.0
    return values.flatten()


def _laplacian_variance(gray: np.ndarray) -> float:
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    lap = (
        -4 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(lap.var())


def _quality_score(sharpness: float, contrast: float, color_std: float, brightness: float) -> float:
    sharpness_score = min(1.0, sharpness / 120.0)
    contrast_score = min(1.0, contrast / 55.0)
    color_score = min(1.0, color_std / 45.0)
    brightness_score = max(0.0, 1.0 - abs(brightness - 128.0) / 128.0)
    score = 0.38 * sharpness_score + 0.26 * contrast_score + 0.18 * color_score + 0.18 * brightness_score
    return round(score, 4)


def _result(
    path: Path,
    width: int,
    height: int,
    sharpness: float,
    brightness: float,
    contrast: float,
    color_std: float,
    accepted: bool,
    reject_reasons: list[str],
    score: float = 0.0,
) -> dict[str, Any]:
    return {
        "keyframe_path": str(path),
        "width": width,
        "height": height,
        "sharpness": round(sharpness, 4),
        "brightness": round(brightness, 4),
        "contrast": round(contrast, 4),
        "color_std": round(color_std, 4),
        "quality_score": score,
        "accepted": accepted,
        "reject_reasons": reject_reasons,
    }
