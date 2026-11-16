"""OpenCV helpers shared by the FastAPI backend and experiments."""
import cv2
import numpy as np

IMG_SIZE = 224


def load_image(path_or_bytes, size=IMG_SIZE):
    """Return an RGB uint8 image resized to (size, size). Accepts a path or raw bytes."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        img = cv2.imdecode(np.frombuffer(path_or_bytes, np.uint8), cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(path_or_bytes), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def leaf_mask(rgb):
    """Rough leaf segmentation in HSV (green + yellow/brown diseased tissue)."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    healthy = cv2.inRange(hsv, (25, 40, 40), (95, 255, 255))
    diseased = cv2.inRange(hsv, (8, 40, 30), (25, 255, 255))
    mask = cv2.bitwise_or(healthy, diseased)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def leaf_ratio(rgb):
    """Fraction of pixels that look like leaf tissue (used to reject non-leaf photos)."""
    return float(np.count_nonzero(leaf_mask(rgb))) / (rgb.shape[0] * rgb.shape[1])


def to_model_input(rgb):
    """The model has its own rescaling layer, so we feed raw 0-255 float32."""
    return np.expand_dims(rgb.astype(np.float32), 0)
