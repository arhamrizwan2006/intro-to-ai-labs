"""
Lab 03 - home assignment data generator

Creates a small labelled image dataset of 20 pictures in two classes:
    data/class_a/  -> a filled circle   (label 0)
    data/class_b/  -> a filled square   (label 1)

The images are synthetic (drawn with Pillow), have different sizes, colours and
positions, and carry a little pixel noise, so resizing and flattening actually
have something to do. Run:  python make_data.py
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SEED = 7
PER_CLASS = 10
OUT = Path("data")


def draw_picture(rng, shape):
    """Draw one noisy picture containing a circle or a square."""
    width = int(rng.integers(80, 141))
    height = int(rng.integers(80, 141))
    background = tuple(int(v) for v in rng.integers(170, 256, size=3))
    colour = tuple(int(v) for v in rng.integers(0, 140, size=3))

    picture = Image.new("RGB", (width, height), background)
    pen = ImageDraw.Draw(picture)

    half = int(min(width, height) * rng.uniform(0.2, 0.35))
    cx = int(rng.integers(half + 2, width - half - 1))
    cy = int(rng.integers(half + 2, height - half - 1))
    box = [cx - half, cy - half, cx + half, cy + half]
    if shape == "circle":
        pen.ellipse(box, fill=colour)
    else:
        pen.rectangle(box, fill=colour)

    pixels = np.asarray(picture, dtype=np.int16)
    pixels = pixels + rng.normal(0, 6, size=pixels.shape).astype(np.int16)
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), "RGB")


def main():
    rng = np.random.default_rng(SEED)
    for folder, shape in (("class_a", "circle"), ("class_b", "square")):
        target = OUT / folder
        target.mkdir(parents=True, exist_ok=True)
        for i in range(1, PER_CLASS + 1):
            draw_picture(rng, shape).save(target / f"{shape}_{i:02d}.png")
    print("Created", 2 * PER_CLASS, "images under", OUT)


if __name__ == "__main__":
    main()
