from pathlib import Path

import pandas as pd
from PIL import Image

from mars_evidence.data.validation import validate_manifest


def test_validation_reports_missing_and_valid(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    Image.new("RGB", (3, 4)).save(raw / "ok.png")
    manifest = pd.DataFrame(
        [
            {"image_id": "ok", "relative_path": "ok.png", "sha256": ""},
            {"image_id": "missing", "relative_path": "x.png", "sha256": ""},
        ]
    )
    result = validate_manifest(manifest, raw)
    assert result["readable"].tolist() == [True, False]
