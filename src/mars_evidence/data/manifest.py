"""Manifest creation for local rover images."""

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from PIL import Image

MANIFEST_COLUMNS = [
    "image_id",
    "file_name",
    "relative_path",
    "sha256",
    "width",
    "height",
    "format",
    "file_size_bytes",
    "product_id",
    "sol",
    "earth_date",
    "camera",
    "eye",
    "source_url",
    "metadata_url",
    "stereo_pair_id",
    "processing_level",
    "credit",
    "downloaded_at_utc",
]
PRESERVED_METADATA_COLUMNS = {
    "product_id",
    "sol",
    "earth_date",
    "camera",
    "eye",
    "source_url",
    "metadata_url",
    "stereo_pair_id",
    "processing_level",
    "credit",
    "downloaded_at_utc",
}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    raw_dir: str | Path,
    output_csv: str | Path,
    metadata_source: str | Path | None = None,
) -> pd.DataFrame:
    """Scan local images and write a metadata manifest."""
    root = Path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(f"No existe la carpeta de imágenes: {root}")
    rows: list[dict[str, object]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS):
        row: dict[str, object] = {column: "" for column in MANIFEST_COLUMNS}
        row.update(
            {
                "image_id": path.stem,
                "file_name": path.name,
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "file_size_bytes": path.stat().st_size,
            }
        )
        try:
            with Image.open(path) as image:
                row.update({"width": image.width, "height": image.height, "format": image.format or ""})
        except (OSError, ValueError):
            row["format"] = path.suffix.lower().lstrip(".")
        rows.append(row)
    frame = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sources = [Path(metadata_source)] if metadata_source else []
    if destination.exists():
        sources.append(destination)
    if sources:
        lookup: dict[str, dict[str, object]] = {}
        for source in sources:
            if source.exists():
                previous = pd.read_csv(source, dtype=str).fillna("")
                for record in previous.to_dict("records"):
                    key = str(record.get("file_name", ""))
                    current = lookup.setdefault(key, {})
                    current.update({column: value for column, value in record.items() if value})
        for row in rows:
            prior = lookup.get(str(row["file_name"]), {})
            for column in PRESERVED_METADATA_COLUMNS:
                if prior.get(column, ""):
                    row[column] = prior[column]
            if not row["downloaded_at_utc"]:
                timestamp = datetime.fromtimestamp((root / str(row["relative_path"])).stat().st_mtime, tz=UTC)
                row["downloaded_at_utc"] = timestamp.isoformat().replace("+00:00", "Z")
        frame = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    frame.to_csv(destination, index=False)
    return frame


def duplicate_groups(manifest: pd.DataFrame) -> pd.DataFrame:
    """Return rows whose SHA-256 occurs more than once."""
    if "sha256" not in manifest.columns:
        raise ValueError("El manifiesto debe contener la columna sha256.")
    valid = manifest[manifest["sha256"].astype(str).ne("")]
    return valid[valid["sha256"].duplicated(keep=False)].copy()
