"""Print a concise environment diagnostic."""

from importlib.util import find_spec

from mars_evidence.config import get_device


def main() -> None:
    """Report Python, device, and optional package availability."""
    import sys

    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch device: {get_device()}")
    for package in ("torch", "transformers", "PIL", "cv2", "pandas", "numpy", "sklearn", "umap", "hdbscan"):
        print(f"{package}: {'OK' if find_spec(package) else 'MISSING'}")


if __name__ == "__main__":
    main()
