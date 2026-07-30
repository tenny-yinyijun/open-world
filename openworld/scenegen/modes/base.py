"""The scene-generation mode interface + a name -> class registry.

A mode is the *generative* half of scene generation: given a base and an edit,
produce one edited image per view. Everything downstream (suite layout,
``initialization.yaml``, manifest) is mode-independent and lives in
:mod:`openworld.scenegen.suite`.

Implement :meth:`SceneGenMode.generate_case` and register the class:

    @register_mode
    class MyMode(SceneGenMode):
        name = "mymode"
        def generate_case(self, *, base, edit, case_dir, edit_order): ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Type

from openworld.scenegen.views import WRIST_FIRST, Base, ViewSet


@dataclass
class Edit:
    """One requested scene edit — becomes one ``init_<i>`` case.

    Attributes:
        prompt: what to change in the scene (the mode's edit instruction).
        instruction: the *policy* command stored in the case, e.g.
            ``"put the mug in the white container"``. Independent of ``prompt``:
            what the image shows vs. what the robot is asked to do.
        label: short tag recorded under ``metadata.edit_label``.
        keep: per-edit override of the suite's shared "keep everything else"
            clause; pins whatever this edit must leave untouched.
        views: optional subset of the suite's views to edit (others are copied
            from the base unchanged).
        edit_order: per-edit override of the suite's anchor choice
            (``wrist_first`` / ``side_first``).
    """

    prompt: str
    instruction: str
    label: str = ""
    keep: Optional[str] = None
    views: Optional[Sequence[str]] = None
    edit_order: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any], index: int = 0) -> "Edit":
        known = {"prompt", "instruction", "label", "keep", "views", "edit_order"}
        prompt = raw.get("prompt")
        instruction = raw.get("instruction")
        if not prompt or not instruction:
            raise ValueError(
                f"edit {index} needs both 'prompt' and 'instruction': {raw!r}"
            )
        return cls(
            prompt=prompt,
            instruction=instruction,
            label=raw.get("label", ""),
            keep=raw.get("keep"),
            views=raw.get("views"),
            edit_order=raw.get("edit_order"),
            extra={k: v for k, v in raw.items() if k not in known},
        )


@dataclass
class CaseResult:
    """What a mode produced for one case.

    ``view_images`` maps view name -> a path or PIL image. A mode that wrote the
    PNGs straight into ``case_dir`` may leave it empty and set ``in_place``.
    """

    view_images: Dict[str, Any] = field(default_factory=dict)
    in_place: bool = False
    prompts: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SceneGenMode(ABC):
    """Base class for a scene-generation backend."""

    #: registry key, used by ``--mode`` on the CLI
    name: str = ""
    #: human-readable one-liner shown by ``--list-modes``
    description: str = ""
    #: set False for modes that only work with the full 3-view DROID set
    supports_two_view: bool = True

    def __init__(self, **params: Any) -> None:
        self.params = params

    def preflight(self, *, base: Base, view_set: ViewSet) -> None:
        """Fail fast on missing credentials / weights, before any real work."""

    @abstractmethod
    def generate_case(
        self,
        *,
        base: Base,
        edit: Edit,
        case_dir: Path,
        edit_order: str = WRIST_FIRST,
    ) -> CaseResult:
        """Produce the edited views for one case."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name!r}>"


MODES: Dict[str, Type[SceneGenMode]] = {}


def register_mode(cls: Type[SceneGenMode]) -> Type[SceneGenMode]:
    """Class decorator: add ``cls`` to the mode registry under ``cls.name``."""
    if not cls.name:
        raise ValueError(f"{cls.__name__} must set a non-empty `name`")
    if cls.name in MODES and MODES[cls.name] is not cls:
        raise ValueError(f"mode '{cls.name}' is already registered")
    MODES[cls.name] = cls
    return cls


def get_mode(name: str, **params: Any) -> SceneGenMode:
    """Instantiate a registered mode by name."""
    try:
        cls = MODES[name]
    except KeyError:
        raise ValueError(
            f"unknown scene-generation mode '{name}'. Available: {sorted(MODES)}"
        ) from None
    return cls(**params)


def available_modes() -> List[str]:
    return sorted(MODES)
