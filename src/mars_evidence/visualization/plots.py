"""Small, reusable image exploration plots."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def plot_image_grid(
    paths: Sequence[str | Path],
    titles: Sequence[str] | None = None,
    columns: int = 4,
    figsize: tuple[int, int] | None = None,
) -> None:
    """Display a grid of images, skipping unreadable files with a visible label."""
    if not paths:
        print("No hay imágenes para mostrar.")
        return
    rows = (len(paths) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=figsize or (4 * columns, 4 * rows), squeeze=False)
    for position, path in enumerate(paths):
        axis = axes.flat[position]
        try:
            with Image.open(path) as image:
                axis.imshow(image.convert("RGB"))
            axis.set_title(titles[position] if titles else Path(path).name, fontsize=8)
        except (OSError, ValueError) as exc:
            axis.text(0.5, 0.5, f"No legible\n{exc}", ha="center", va="center", fontsize=8)
        axis.axis("off")
    for axis in axes.flat[len(paths) :]:
        axis.axis("off")
    figure.tight_layout()
