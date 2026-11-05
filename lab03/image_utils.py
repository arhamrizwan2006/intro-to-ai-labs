"""
Lab 03 - image helper
Introduction to Artificial Intelligence (AI-101L)

image_to_features() turns any image file into a fixed-length feature vector:
load -> force RGB -> resize -> grayscale -> flatten.
"""

import numpy as np
from PIL import Image


def image_to_features(path, size=(64, 64)):
    """Return a 1-D uint8 feature vector for the image stored at `path`.

    Parameters
    ----------
    path : str
        Path to an image file (png, jpg, ...).
    size : tuple[int, int]
        Target (width, height). Every image is forced to this size, so every
        returned vector has length width * height. That guarantee is what lets
        many different images be stacked into one matrix for a model.

    Returns
    -------
    numpy.ndarray of shape (width * height,), dtype uint8
    """
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("size must contain two positive integers")

    with Image.open(path) as picture:
        rgb = picture.convert("RGB")            # same channel layout for every file
        resized = rgb.resize((width, height))   # same pixel grid for every file
        gray = resized.convert("L")             # three channels -> one

    return np.asarray(gray, dtype=np.uint8).ravel()
