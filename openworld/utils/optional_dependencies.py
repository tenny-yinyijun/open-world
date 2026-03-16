from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Sequence


@dataclass(frozen=True)
class BackendSpec:
    module_path: str
    class_name: str
    extra_name: str | None = None
    required_modules: tuple[str, ...] = ()


class MissingOptionalDependencyError(ImportError):
    """Raised when a selected backend requires an optional dependency group."""


def load_backend_class(spec: BackendSpec) -> type:
    module = importlib.import_module(spec.module_path)
    return getattr(module, spec.class_name)


def require_modules(
    backend_name: str,
    backend_kind: str,
    required_modules: Sequence[str],
    extra_name: str | None,
) -> None:
    missing = [module for module in required_modules if importlib.util.find_spec(module) is None]
    if not missing:
        return

    install_hint = "Install the required upstream runtime dependencies."
    if extra_name:
        install_hint = (
            f"Install with `uv sync --extra {extra_name}` or "
            f'`pip install -e ".[{extra_name}]"`.'
        )

    missing_list = ", ".join(missing)
    raise MissingOptionalDependencyError(
        f"The '{backend_name}' {backend_kind} backend is missing optional dependencies "
        f"({missing_list}). {install_hint}"
    )
