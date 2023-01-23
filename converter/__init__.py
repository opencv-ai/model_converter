# type: ignore[attr-defined]
"""PyTorch model conversion to different formats"""

from importlib import metadata as importlib_metadata

from .converter import Converter


def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: str = get_version()
