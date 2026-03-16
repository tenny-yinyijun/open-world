"""I/O utilities for loading configs and data."""

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def ensure_dir(path: str) -> Path:
    """Create directory (and parents) if it doesn't exist, return as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
