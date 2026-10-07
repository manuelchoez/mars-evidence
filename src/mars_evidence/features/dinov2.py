"""DINOv2 embedding extraction with lazy model loading."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from mars_evidence.config import get_device


class DINOv2Embedder:
    """Batch image embedder; model weights load only when instantiated."""

    def __init__(self, model_name: str = "facebook/dinov2-small", device: torch.device | None = None) -> None:
        self.device = device or get_device()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def transform(self, images: Sequence[Image.Image]) -> dict[str, torch.Tensor]:
        """Prepare a batch of RGB PIL images."""
        return self.processor(images=list(images), return_tensors="pt")

    def embed_images(self, paths: Sequence[str | Path], batch_size: int = 8) -> np.ndarray:
        """Return L2-normalized CLS embeddings for image paths."""
        if not paths:
            return np.empty((0, 0), dtype=np.float32)
        vectors: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(paths), batch_size):
                batch_paths = paths[start : start + batch_size]
                images = [Image.open(path).convert("RGB") for path in batch_paths]
                try:
                    inputs = {key: value.to(self.device) for key, value in self.transform(images).items()}
                    outputs = self.model(**inputs)
                    batch = outputs.last_hidden_state[:, 0]
                    batch = torch.nn.functional.normalize(batch, p=2, dim=1)
                    vectors.append(batch.cpu().numpy())
                finally:
                    for image in images:
                        image.close()
        return np.concatenate(vectors).astype(np.float32)


def save_embeddings(embeddings: np.ndarray, image_ids: Sequence[str], path: str | Path) -> None:
    """Save embeddings and aligned image identifiers to NPZ."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, embeddings=embeddings, image_ids=np.asarray(image_ids))


def load_embeddings(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load embeddings and image identifiers from NPZ."""
    with np.load(Path(path), allow_pickle=False) as data:
        return data["embeddings"], data["image_ids"]
