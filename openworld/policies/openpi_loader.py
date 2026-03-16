from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Optional


DEFAULT_OPENPI_REPO = Path(__file__).resolve().parents[2] / "external" / "openpi"


def ensure_openpi_repo_on_path(repo_path: Optional[str] = None) -> Path:
    repo_root = Path(repo_path or DEFAULT_OPENPI_REPO).resolve()
    if not repo_root.exists():
        raise FileNotFoundError(
            f"OpenPI repo not found at {repo_root}. Clone your OpenPI fork into "
            "`external/openpi` or set `repo_path` explicitly in the policy config."
        )

    candidate_paths = [
        repo_root / "src",
        repo_root / "packages" / "openpi-client" / "src",
    ]
    missing_paths = [path for path in candidate_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "OpenPI repo is missing expected source directories: "
            + ", ".join(str(path) for path in missing_paths)
        )

    for path in reversed(candidate_paths):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    return repo_root


def load_policy_from_checkpoint(
    *,
    config_name: str,
    checkpoint_path: str,
    repo_path: Optional[str] = None,
    default_prompt: Optional[str] = None,
    pytorch_device: Optional[str] = None,
) -> Any:
    ensure_openpi_repo_on_path(repo_path)

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    train_config = _config.get_config(config_name)
    return _policy_config.create_trained_policy(
        train_config,
        checkpoint_path,
        default_prompt=default_prompt,
        pytorch_device=pytorch_device,
    )
