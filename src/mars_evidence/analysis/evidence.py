"""Utilities for building a human-reviewable pilot evidence table."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

EVIDENCE_COLUMNS = [
    "image_id",
    "filename",
    "relative_path",
    "source_url",
    "sol",
    "camera",
    "acquisition_date",
    "width",
    "height",
    "sha256",
    "cluster_id",
    "is_cluster_noise",
    "anomaly_score",
    "anomaly_rank",
    "nearest_image_1",
    "nearest_similarity_1",
    "nearest_image_2",
    "nearest_similarity_2",
    "nearest_image_3",
    "nearest_similarity_3",
    "human_review_status",
    "human_explanation",
    "possible_artifact",
    "review_notes",
]

REVIEW_COLUMNS = [
    "reviewed",
    "expected_terrain",
    "rover_artifact",
    "image_quality_issue",
    "requires_further_review",
]


def deterministic_image_order(manifest: pd.DataFrame) -> pd.DataFrame:
    """Return a stable manifest order based on the image identifier."""
    if "image_id" not in manifest.columns:
        raise ValueError("El manifiesto debe incluir la columna 'image_id'.")
    if manifest["image_id"].duplicated().any():
        raise ValueError("El manifiesto contiene image_id duplicados.")
    return manifest.sort_values("image_id", kind="stable").reset_index(drop=True)


def validate_embedding_alignment(
    embedding_ids: Sequence[str], image_ids: Sequence[str]
) -> None:
    """Validate that embeddings and image identifiers have the same order."""
    expected = [str(image_id) for image_id in image_ids]
    actual = [str(image_id) for image_id in embedding_ids]
    if actual != expected:
        raise ValueError(
            "El orden de embeddings no coincide con el orden de imágenes "
            "del manifiesto."
        )


def save_embedding_artifacts(
    embeddings: np.ndarray,
    image_ids: Sequence[str],
    embedding_path: Path,
    order_path: Path,
) -> None:
    """Save embeddings and their exact image order as NumPy files."""
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embedding_path, np.asarray(embeddings, dtype=np.float32))
    np.save(order_path, np.asarray([str(image_id) for image_id in image_ids]))


def load_embedding_artifact(
    embedding_path: Path,
    order_path: Path,
    legacy_npz_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Load current artifacts or a compatible legacy DINOv2 NPZ file."""
    if embedding_path.exists() and order_path.exists():
        embeddings = np.load(embedding_path)
        image_ids = np.load(order_path).astype(str)
        return np.asarray(embeddings, dtype=np.float32), image_ids, True

    if legacy_npz_path is None or not legacy_npz_path.exists():
        return np.empty((0, 0), dtype=np.float32), np.asarray([], dtype=str), False

    with np.load(legacy_npz_path, allow_pickle=True) as data:
        if "embeddings" not in data or "image_ids" not in data:
            raise ValueError("El archivo legado no contiene embeddings e image_ids.")
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        image_ids = np.asarray(data["image_ids"], dtype=str)
    return embeddings, image_ids, True


def nearest_neighbors_table(
    embeddings: np.ndarray,
    image_ids: Sequence[str],
    k: int = 3,
) -> pd.DataFrame:
    """Return cosine-nearest neighbors, excluding each query image itself."""
    matrix = np.asarray(embeddings, dtype=np.float32)
    ids = np.asarray([str(image_id) for image_id in image_ids], dtype=str)
    if matrix.ndim != 2 or matrix.shape[0] != len(ids):
        raise ValueError("Embeddings e image_ids deben tener el mismo número de filas.")
    if k < 1 or k >= len(ids):
        raise ValueError("k debe ser mayor que cero y menor que el número de imágenes.")

    similarities = cosine_similarity(matrix)
    rows: list[dict[str, object]] = []
    for index, image_id in enumerate(ids):
        candidate_indices = np.argsort(-similarities[index], kind="stable")
        candidate_indices = [candidate for candidate in candidate_indices if candidate != index]
        row: dict[str, object] = {"image_id": image_id}
        for rank, neighbor_index in enumerate(candidate_indices[:k], start=1):
            row[f"nearest_image_{rank}"] = ids[neighbor_index]
            row[f"nearest_similarity_{rank}"] = float(similarities[index, neighbor_index])
        rows.append(row)
    return pd.DataFrame(rows)


def rank_anomalies(scores: Sequence[float]) -> np.ndarray:
    """Return one-based anomaly ranks, with larger scores ranked first."""
    values = np.asarray(scores, dtype=float)
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=int)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def build_pilot_evidence_table(
    manifest: pd.DataFrame,
    cluster_labels: Sequence[int],
    anomaly_scores: Sequence[float],
    neighbors: pd.DataFrame,
) -> pd.DataFrame:
    """Combine verified metadata and computed metrics for human review."""
    ordered = deterministic_image_order(manifest).copy()
    labels = np.asarray(cluster_labels)
    scores = np.asarray(anomaly_scores, dtype=float)
    if len(ordered) != len(labels) or len(ordered) != len(scores):
        raise ValueError("Manifiesto y resultados calculados tienen longitudes distintas.")

    def metadata_column(name: str, fallback: str = "") -> pd.Series:
        if name in ordered.columns:
            return ordered[name].fillna(fallback)
        return pd.Series([fallback] * len(ordered), index=ordered.index)

    table = pd.DataFrame(
        {
            "image_id": metadata_column("image_id"),
            "filename": metadata_column("file_name"),
            "relative_path": metadata_column("relative_path"),
            "source_url": metadata_column("source_url"),
            "sol": metadata_column("sol"),
            "camera": metadata_column("camera"),
            "acquisition_date": metadata_column("earth_date"),
            "width": metadata_column("width"),
            "height": metadata_column("height"),
            "sha256": metadata_column("sha256"),
            "cluster_id": labels,
            "is_cluster_noise": labels == -1,
            "anomaly_score": scores,
            "anomaly_rank": rank_anomalies(scores),
            "human_review_status": "pending",
            "human_explanation": "",
            "possible_artifact": "",
            "review_notes": "",
        }
    )
    table = table.merge(neighbors, on="image_id", how="left", validate="one_to_one")
    for column in EVIDENCE_COLUMNS:
        if column not in table.columns:
            table[column] = ""
    return table[EVIDENCE_COLUMNS].sort_values("anomaly_rank").reset_index(drop=True)


def export_evidence_table(table: pd.DataFrame, output_path: Path) -> None:
    """Write an evidence table to CSV, creating its parent directory."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
