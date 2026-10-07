"""Tests for pilot evidence review utilities."""

import numpy as np
import pandas as pd
import pytest

from mars_evidence.analysis.evidence import (
    build_pilot_evidence_table,
    deterministic_image_order,
    export_evidence_table,
    nearest_neighbors_table,
    rank_anomalies,
    validate_embedding_alignment,
)


def test_embedding_alignment_and_deterministic_order() -> None:
    manifest = pd.DataFrame({"image_id": ["b", "a"]})
    ordered = deterministic_image_order(manifest)
    assert ordered["image_id"].tolist() == ["a", "b"]
    validate_embedding_alignment(["a", "b"], ordered["image_id"])
    with pytest.raises(ValueError, match="orden"):
        validate_embedding_alignment(["b", "a"], ordered["image_id"])


def test_nearest_neighbors_exclude_query() -> None:
    embeddings = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    neighbors = nearest_neighbors_table(embeddings, ["a", "b", "c"], k=2)
    assert neighbors.loc[0, "nearest_image_1"] == "b"
    assert neighbors.loc[0, "nearest_image_1"] != neighbors.loc[0, "image_id"]
    assert neighbors.loc[0, "nearest_similarity_1"] > 0.9


def test_anomaly_ranking_is_descending_and_stable() -> None:
    assert rank_anomalies([0.2, 0.9, 0.9, 0.1]).tolist() == [3, 1, 2, 4]


def test_export_table_and_missing_metadata(tmp_path) -> None:
    manifest = pd.DataFrame(
        {
            "image_id": ["b", "a"],
            "file_name": ["b.png", "a.png"],
            "relative_path": ["b.png", "a.png"],
            "source_url": ["https://example.org/b", np.nan],
            "earth_date": [np.nan, "2021-01-01"],
        }
    )
    neighbors = nearest_neighbors_table(
        np.asarray([[1.0, 0.0], [0.0, 1.0]]), ["a", "b"], k=1
    )
    table = build_pilot_evidence_table(manifest, [0, -1], [0.2, 0.8], neighbors)
    output_path = tmp_path / "evidence.csv"
    export_evidence_table(table, output_path)
    exported = pd.read_csv(output_path)
    assert output_path.exists()
    assert exported.loc[0, "image_id"] == "b"
    assert pd.isna(exported.loc[1, "source_url"])
    assert exported.loc[0, "human_review_status"] == "pending"
