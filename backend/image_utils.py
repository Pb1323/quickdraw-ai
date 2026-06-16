import base64
import binascii
import io
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import BLANK_PIXEL_RATIO_THRESHOLD, MAX_IMAGE_BYTES


def decode_data_url(data_url: str) -> Image.Image:
    if not isinstance(data_url, str) or "," not in data_url:
        raise ValueError("Invalid image data URL.")

    header, encoded = data_url.split(",", 1)
    if not header.startswith("data:image"):
        raise ValueError("Only image data URLs are accepted.")

    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Image is not valid base64 data.") from exc

    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Image payload is too large.")

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Decoded payload is not a readable image.") from exc

    return image


def _content_bbox(gray: np.ndarray, threshold: int = 245) -> Optional[Tuple[int, int, int, int]]:
    mask = gray < threshold
    if not np.any(mask):
        return None

    coords = np.argwhere(mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return int(x_min), int(y_min), int(x_max), int(y_max)


def preprocess_canvas_image(image: Image.Image, output_size: int = 28) -> Tuple[np.ndarray, bool]:
    gray = image.convert("L")
    arr = np.asarray(gray, dtype=np.uint8)

    ink_ratio = float(np.mean(arr < 245))
    if ink_ratio < BLANK_PIXEL_RATIO_THRESHOLD:
        blank = np.zeros((1, 1, output_size, output_size), dtype=np.float32)
        return blank, True

    bbox = _content_bbox(arr)
    if bbox is None:
        blank = np.zeros((1, 1, output_size, output_size), dtype=np.float32)
        return blank, True

    x_min, y_min, x_max, y_max = bbox
    cropped = gray.crop((x_min, y_min, x_max + 1, y_max + 1))

    pad = 8
    padded = ImageOps.expand(cropped, border=pad, fill=255)

    w, h = padded.size
    side = max(w, h)
    square = Image.new("L", (side, side), color=255)
    square.paste(padded, ((side - w) // 2, (side - h) // 2))

    resized = square.resize((output_size, output_size), Image.Resampling.LANCZOS)
    inverted = ImageOps.invert(resized)

    norm = np.asarray(inverted, dtype=np.float32) / 255.0
    tensor = norm[np.newaxis, np.newaxis, :, :]
    return tensor, False
