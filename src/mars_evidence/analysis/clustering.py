"""UMAP and HDBSCAN helpers."""

import numpy as np


def reduce_umap(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    """Reduce embeddings with UMAP."""
    import umap

    if len(embeddings) < 3:
        raise ValueError("UMAP requiere al menos 3 imágenes.")
    neighbors = min(n_neighbors, len(embeddings) - 1)
    reducer = umap.UMAP(
        n_neighbors=neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)


def cluster_hdbscan(embeddings: np.ndarray, min_cluster_size: int = 5, min_samples: int = 3) -> np.ndarray:
    """Assign HDBSCAN labels, where -1 denotes noise."""
    import hdbscan

    if len(embeddings) < min_cluster_size:
        raise ValueError("HDBSCAN requiere al menos min_cluster_size imágenes.")
    return hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit_predict(embeddings)
