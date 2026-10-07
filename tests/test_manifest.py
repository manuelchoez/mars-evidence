from pathlib import Path

from PIL import Image

from mars_evidence.data.manifest import build_manifest, duplicate_groups


def test_build_manifest_and_duplicates(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    image = Image.new("RGB", (8, 6), color="red")
    image.save(raw / "a.jpg")
    image.save(raw / "b.jpg")
    frame = build_manifest(raw, tmp_path / "observations.csv")
    assert list(frame.columns)[:4] == ["image_id", "file_name", "relative_path", "sha256"]
    assert len(duplicate_groups(frame)) == 2
