import numpy as np
import pandas as pd

from mars_evidence.analysis.similarity import find_similar


def test_similarity_excludes_query() -> None:
    embeddings = np.array([[1.0, 0], [0.9, 0.1], [0, 1.0]], dtype=np.float32)
    manifest = pd.DataFrame({"image_id": ["a", "b", "c"], "relative_path": ["a", "b", "c"]})
    result = find_similar(embeddings, manifest, index=0, k=2)
    assert result["image_id"].tolist() == ["b", "c"]
    assert "similarity_score" in result
