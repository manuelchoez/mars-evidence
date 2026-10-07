from email.message import Message
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image
from scripts.download_mastcam_z_pilot import (
    MAX_LIMIT,
    run_download,
    sha256_file,
    validate_limit,
    validate_url,
)

ALLOWED_URL = "https://pds-imaging.jpl.nasa.gov/image.png"
METADATA_URL = "https://pds-imaging.jpl.nasa.gov/image.xml"


class FakeResponse:
    """Minimal context manager compatible with urllib responses."""

    def __init__(self, payload: bytes, content_type: str = "image/png") -> None:
        self._stream = BytesIO(payload)
        self.headers = Message()
        self.headers["Content-Type"] = content_type

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        self._stream.close()

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)


def image_bytes() -> bytes:
    """Create a small local PNG fixture in memory."""
    stream = BytesIO()
    Image.new("RGB", (4, 3), color="red").save(stream, format="PNG")
    return stream.getvalue()


def row(file_name: str = "sample.png", source_url: str = ALLOWED_URL) -> dict[str, str]:
    """Return a minimal valid selection row."""
    return {"image_id": file_name, "file_name": file_name, "source_url": source_url, "metadata_url": METADATA_URL}


def opener_for(payload: bytes, content_type: str = "image/png"):
    """Return an opener that never accesses the network."""

    def opener(*args: object, **kwargs: object) -> FakeResponse:
        return FakeResponse(payload, content_type)

    return opener


def test_rejects_http_and_unauthorized_domain() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        validate_url("http://pds-imaging.jpl.nasa.gov/image.png")
    with pytest.raises(ValueError, match="no autorizado"):
        validate_url("https://example.com/image.png")


def test_respects_absolute_limit() -> None:
    validate_limit(MAX_LIMIT)
    with pytest.raises(ValueError, match="entre"):
        validate_limit(MAX_LIMIT + 1)


def test_generates_sha256_and_downloads_image(tmp_path: Path) -> None:
    summary = run_download([row()], tmp_path, opener=opener_for(image_bytes()))
    destination = tmp_path / "sample.png"
    assert summary["descargadas"] == 1
    assert sha256_file(destination)


def test_detects_duplicate_content(tmp_path: Path) -> None:
    rows = [row("one.png"), row("two.png")]
    summary = run_download(rows, tmp_path, opener=opener_for(image_bytes()))
    assert summary["descargadas"] == 1
    assert summary["duplicadas"] == 1
    assert not (tmp_path / "two.png").exists()


def test_does_not_overwrite_existing_valid_file(tmp_path: Path) -> None:
    destination = tmp_path / "sample.png"
    destination.write_bytes(image_bytes())
    original = sha256_file(destination)

    def fail_opener(*args: object, **kwargs: object) -> FakeResponse:
        raise AssertionError("no debe descargar un archivo existente válido")

    summary = run_download([row()], tmp_path, opener=fail_opener)
    assert summary["omitidas"] == 1
    assert sha256_file(destination) == original


def test_rejects_non_image_response(tmp_path: Path) -> None:
    summary = run_download([row()], tmp_path, opener=opener_for(b"not an image", "image/png"))
    assert summary["dañadas"] == 1
    assert not (tmp_path / "sample.png").exists()


def test_dry_run_does_not_download(tmp_path: Path) -> None:
    def fail_opener(*args: object, **kwargs: object) -> FakeResponse:
        raise AssertionError("dry-run no debe abrir la URL")

    summary = run_download([row()], tmp_path, dry_run=True, opener=fail_opener)
    assert summary == {"seleccionadas": 1, "descargadas": 0, "omitidas": 0, "duplicadas": 0, "dañadas": 0, "errores": 0}
