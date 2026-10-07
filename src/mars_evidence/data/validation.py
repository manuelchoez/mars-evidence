"""Image file validation utilities."""

from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


def validate_manifest(manifest: pd.DataFrame, raw_dir: str | Path) -> pd.DataFrame:
    """Validate existence, readability, and declared metadata for each row."""
    root = Path(raw_dir)
    required = {"image_id", "relative_path", "sha256"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for record in manifest.to_dict("records"):
        path = root / str(record["relative_path"])
        result: dict[str, Any] = {
            "image_id": record["image_id"],
            "path": str(path),
            "exists": path.exists(),
            "readable": False,
            "error": "",
        }
        if not path.exists():
            result["error"] = "archivo_no_encontrado"
        else:
            try:
                with Image.open(path) as image:
                    image.verify()
                result["readable"] = True
            except (OSError, ValueError) as exc:
                result["error"] = f"archivo_dañado_o_formato_no_valido: {exc}"
        rows.append(result)
    return pd.DataFrame(rows)
