"""Download a bounded, traceable Mastcam-Z pilot selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = PROJECT_ROOT / "data/metadata/mastcam_z_pilot_selection.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/raw/mastcam_z/pilot"
MAX_LIMIT = 50
RETRIES = 3
ALLOWED_HOSTS = {"pds-imaging.jpl.nasa.gov"}
REQUIRED_SELECTION_COLUMNS = {"image_id", "file_name", "source_url", "metadata_url"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


class InvalidImageError(ValueError):
    """Raised when a response cannot be opened as an image."""


def validate_limit(limit: int) -> None:
    """Reject invalid or unsafe batch sizes."""
    if not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"--limit debe estar entre 1 y {MAX_LIMIT}.")


def validate_url(url: str) -> None:
    """Validate HTTPS and the allowlist derived from official sources."""
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"La URL debe usar HTTPS: {url}")
    if parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError(f"Dominio no autorizado: {parsed.hostname}")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(path: Path) -> tuple[int, int, str]:
    """Verify an image and return width, height, and format."""
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.width, image.height, image.format or ""
    except (OSError, ValueError) as exc:
        raise InvalidImageError(f"Archivo de imagen no válido: {path}") from exc


def read_selection(path: Path, limit: int) -> list[dict[str, str]]:
    """Read and validate a relative CSV selection file."""
    if path.is_absolute():
        raise ValueError("--selection-file debe ser una ruta relativa al proyecto.")
    selection_path = PROJECT_ROOT / path
    if not selection_path.exists():
        raise FileNotFoundError(f"No existe el archivo de selección: {path}")
    with selection_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_SELECTION_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Faltan columnas en la selección: {sorted(missing)}")
        rows = list(reader)
    if len(rows) > MAX_LIMIT:
        raise ValueError(f"La selección contiene {len(rows)} filas; el máximo absoluto es {MAX_LIMIT}.")
    return rows[:limit]


def _safe_destination(output_dir: Path, file_name: str) -> Path:
    """Resolve a filename without allowing path traversal."""
    candidate = Path(file_name)
    if candidate.name != file_name or candidate.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Nombre de imagen no permitido: {file_name}")
    return output_dir / candidate.name


def _existing_hashes(output_dir: Path) -> set[str]:
    """Hash valid existing image files for duplicate detection."""
    hashes: set[str] = set()
    for path in output_dir.iterdir() if output_dir.exists() else ():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            try:
                inspect_image(path)
            except InvalidImageError:
                continue
            hashes.add(sha256_file(path))
    return hashes


def _download_to_temp(
    url: str,
    destination_dir: Path,
    opener: Callable[..., BinaryIO],
) -> Path:
    """Download a URL to a temporary file and validate its image content."""
    request = Request(url, headers={"User-Agent": "MARS-EVIDENCE/0.1"})
    last_error: Exception | None = None
    for attempt in range(RETRIES):
        temporary_name: str | None = None
        try:
            with opener(request, timeout=30) as response:
                content_type = response.headers.get_content_type()
                if content_type and not content_type.startswith("image/"):
                    raise InvalidImageError(f"Respuesta no es imagen ({content_type}): {url}")
                with tempfile.NamedTemporaryFile(
                    mode="wb", prefix=".mastcam-z-", suffix=".tmp", dir=destination_dir, delete=False
                ) as temporary:
                    temporary_name = temporary.name
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        temporary.write(chunk)
            temporary_path = Path(temporary_name)
            inspect_image(temporary_path)
            return temporary_path
        except InvalidImageError:
            if temporary_name:
                Path(temporary_name).unlink(missing_ok=True)
            raise
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if temporary_name:
                Path(temporary_name).unlink(missing_ok=True)
            if attempt < RETRIES - 1:
                time.sleep(0.25 * (attempt + 1))
    raise RuntimeError(f"No se pudo descargar después de {RETRIES} intentos: {url}") from last_error


def run_download(
    rows: list[dict[str, str]],
    output_dir: Path,
    dry_run: bool = False,
    opener: Callable[..., BinaryIO] | None = None,
) -> dict[str, int]:
    """Process selected rows and return a category summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"seleccionadas": len(rows), "descargadas": 0, "omitidas": 0, "duplicadas": 0, "dañadas": 0, "errores": 0}
    for row in rows:
        try:
            validate_url(row["source_url"])
            validate_url(row["metadata_url"])
            destination = _safe_destination(output_dir, row["file_name"])
            if destination.exists():
                try:
                    inspect_image(destination)
                except InvalidImageError:
                    summary["errores"] += 1
                    print(f"ERROR archivo existente no válido, no se sobrescribe: {destination}")
                else:
                    summary["omitidas"] += 1
                    print(f"OMITIDA ya existe: {destination}")
                continue
            if dry_run:
                print(f"DRY-RUN {row['source_url']}")
                continue
            temporary = _download_to_temp(row["source_url"], output_dir, opener or urlopen)
            digest = sha256_file(temporary)
            if digest in _existing_hashes(output_dir):
                temporary.unlink(missing_ok=True)
                summary["duplicadas"] += 1
                print(f"DUPLICADA {row['file_name']} sha256={digest}")
                continue
            os.replace(temporary, destination)
            summary["descargadas"] += 1
            print(f"DESCARGADA {destination} sha256={digest}")
        except InvalidImageError as exc:
            summary["dañadas"] += 1
            print(f"DAÑADA {row.get('file_name', '')}: {exc}")
        except (KeyError, ValueError, RuntimeError) as exc:
            summary["errores"] += 1
            print(f"ERROR {row.get('file_name', '')}: {exc}")
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-file", type=Path, default=Path("data/metadata/mastcam_z_pilot_selection.csv"))
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Validate the selection and optionally download its images."""
    args = parse_args()
    validate_limit(args.limit)
    rows = read_selection(args.selection_file, args.limit)
    summary = run_download(rows, DEFAULT_OUTPUT, dry_run=args.dry_run)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
