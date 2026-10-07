"""Exploratory visual anomaly scoring."""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def visual_anomaly_scores(embeddings: np.ndarray, neighbors: int = 5) -> np.ndarray:
    """Score each image by normalized mean cosine distance to k nearest neighbors."""
    if len(embeddings) < 2:
        return np.zeros(len(embeddings), dtype=np.float32)
    k = min(max(1, neighbors), len(embeddings) - 1)
    search = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(embeddings)
    distances, _ = search.kneighbors(embeddings)
    raw = distances[:, 1:].mean(axis=1)
    spread = raw.max() - raw.min()
    return ((raw - raw.min()) / spread if spread > 0 else np.zeros_like(raw)).astype(np.float32)
