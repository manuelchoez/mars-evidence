"""Build the local image manifest from the repository root."""

import argparse
from pathlib import Path

from mars_evidence.data.manifest import build_manifest


def main() -> None:
    """Create the Mastcam-Z pilot manifest by default."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/mastcam_z/pilot"))
    parser.add_argument("--output", type=Path, default=Path("data/metadata/mastcam_z_pilot.csv"))
    parser.add_argument(
        "--metadata-source",
        type=Path,
        default=Path("data/metadata/mastcam_z_pilot_selection.csv"),
    )
    args = parser.parse_args()
    raw_dir = args.raw_dir if args.raw_dir.is_absolute() else root / args.raw_dir
    output = args.output if args.output.is_absolute() else root / args.output
    metadata_source = args.metadata_source if args.metadata_source.is_absolute() else root / args.metadata_source
    frame = build_manifest(raw_dir, output, metadata_source=metadata_source)
    print(f"Manifiesto creado: {output} ({len(frame)} imágenes)")


if __name__ == "__main__":
    main()
