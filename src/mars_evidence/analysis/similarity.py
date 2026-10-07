"""Cosine similarity search using scikit-learn."""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def find_similar(embeddings: np.ndarray, manifest: pd.DataFrame, index: int, k: int = 5) -> pd.DataFrame:
    """Return the k nearest other images for an embedding row."""
    if len(embeddings) != len(manifest):
        raise ValueError("Embeddings y manifiesto deben tener la misma longitud.")
    if not 0 <= index < len(embeddings):
        raise IndexError(f"Índice fuera de rango: {index}")
    scores = cosine_similarity(embeddings[index : index + 1], embeddings)[0]
    order = np.argsort(-scores)
    order = [int(i) for i in order if i != index][:k]
    rows = manifest.iloc[order].copy()
    rows.insert(0, "similarity_score", scores[order])
    return rows.reset_index(drop=True)
