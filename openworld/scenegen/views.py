"""View sets: which cameras a suite has, and in what order they get edited.

A world-model *view set* is the ordered tuple of camera names a checkpoint was
trained on. Two setups matter here:

* **3-view** ``(exterior_right, exterior_left, wrist)`` — the original DROID
  layout (``configs/evaluation/0617_ar_pi05.yaml``).
* **2-view** ``(exterior_right, wrist)`` — one side camera + wrist, the layout
  the published ``wm_student_2view`` model uses
  (``configs/evaluation/teleop_ar_pi05.yaml``).

The canonical order is always *sides first (right, then left), wrist last*,
which is exactly the ``view_order`` those eval configs declare — so a suite
built here stacks the same way the world model rolls out.

Separate from the view *set* is the **edit order**: the sequence in which a
generative mode visits the views. Editing views independently makes them drift
apart (nanobanana re-invents the added object in each view), so modes edit them
in a chain — the first view is the *anchor*, and every later view is conditioned
on the already-edited ones. Which view anchors is a real choice:

* :data:`WRIST_FIRST` — anchor on the wrist (top-down) view. The overhead view
  pins object identity and placement on the table most precisely, so this is the
  default and the right choice for object add/remove/replace edits.
* :data:`SIDE_FIRST` — anchor on the exterior (side) view. The side view carries
  the room, background, and lighting, so anchor here for background / lighting /
  scene-context edits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

WRIST = "wrist"
# Side cameras in canonical order. A 2-view setup keeps only the first.
SIDE_VIEWS: Tuple[str, ...] = ("exterior_right", "exterior_left")
ALL_VIEWS: Tuple[str, ...] = SIDE_VIEWS + (WRIST,)

# Edit-order strategies (see module docstring).
WRIST_FIRST = "wrist_first"
SIDE_FIRST = "side_first"
EDIT_ORDERS: Tuple[str, ...] = (WRIST_FIRST, SIDE_FIRST)

# World-model frame resolution (W, H). Suite PNGs must match it.
DEFAULT_TARGET_SIZE: Tuple[int, int] = (320, 192)


def canonical_order(views: Sequence[str]) -> Tuple[str, ...]:
    """Sort ``views`` into world-model ``view_order``: sides first, wrist last.

    Raises ``ValueError`` on an unknown or duplicated view name.
    """
    seen = set()
    for v in views:
        if v not in ALL_VIEWS:
            raise ValueError(f"unknown view '{v}' (known: {list(ALL_VIEWS)})")
        if v in seen:
            raise ValueError(f"duplicate view '{v}'")
        seen.add(v)
    if not seen:
        raise ValueError("view set is empty")
    return tuple(v for v in ALL_VIEWS if v in seen)


@dataclass(frozen=True)
class ViewSet:
    """The ordered cameras of a suite, plus which are sides and which is wrist.

    ``views`` is in canonical (world-model ``view_order``) order. Use
    :meth:`edit_sequence` to get the order a mode should *edit* them in.
    """

    views: Tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "views", canonical_order(self.views))

    @property
    def sides(self) -> Tuple[str, ...]:
        return tuple(v for v in self.views if v != WRIST)

    @property
    def has_wrist(self) -> bool:
        return WRIST in self.views

    @property
    def num_cams(self) -> int:
        return len(self.views)

    def edit_sequence(self, edit_order: str = WRIST_FIRST) -> Tuple[str, ...]:
        """Return ``views`` reordered for editing (anchor view first).

        ``wrist_first`` puts the wrist view first, then the sides;
        ``side_first`` puts the sides first, then the wrist. Views absent from
        this set are simply skipped, so both orders work for a 2-view suite.
        """
        if edit_order not in EDIT_ORDERS:
            raise ValueError(
                f"unknown edit_order '{edit_order}' (expected one of {list(EDIT_ORDERS)})"
            )
        if edit_order == WRIST_FIRST:
            ordered = ([WRIST] if self.has_wrist else []) + list(self.sides)
        else:
            ordered = list(self.sides) + ([WRIST] if self.has_wrist else [])
        return tuple(ordered)

    def describe(self) -> str:
        return f"{self.num_cams}-view {list(self.views)}"


# Convenience presets.
VIEWS_3 = ViewSet(ALL_VIEWS)
VIEWS_2 = ViewSet(("exterior_right", WRIST))


def is_side(view: str) -> bool:
    return view in SIDE_VIEWS


@dataclass
class Base:
    """A resolved scene-generation base: unedited views + robot start state."""

    dir: Path
    view_set: ViewSet
    initial_state: dict
    scene: Optional[str] = None
    name: str = ""

    def path(self, view: str) -> Path:
        return self.dir / f"{view}.png"


def discover_views(base_dir: Path) -> List[str]:
    """Return the view names that have a ``<view>.png`` in ``base_dir``."""
    return [v for v in ALL_VIEWS if (base_dir / f"{v}.png").exists()]


def resolve_base(
    base: str,
    *,
    views: Optional[Sequence[str]] = None,
    assets_root: Optional[Path] = None,
) -> Base:
    """Resolve a base name (``tri`` / ``irom``) or a path into a :class:`Base`.

    A base is a directory holding one ``<view>.png`` per camera plus a
    ``template.yaml`` carrying ``initial_state`` and a default ``scene`` tag.
    Named bases live in ``assets/<name>``; anything else is treated as a path.

    ``views`` selects a subset of the base's cameras — this is how a 3-view base
    is used to build a **2-view** suite (``views=["exterior_right", "wrist"]``).
    It defaults to every view present in the directory.
    """
    if assets_root is None:
        assets_root = Path(__file__).resolve().parents[2] / "assets"

    candidate = assets_root / base
    base_dir = candidate if candidate.is_dir() else Path(base).expanduser().resolve()
    if not base_dir.is_dir():
        known = (
            sorted(p.name for p in assets_root.iterdir() if p.is_dir())
            if assets_root.is_dir()
            else []
        )
        raise FileNotFoundError(
            f"base '{base}' not found (looked in assets/{base} and as a path). "
            f"Known assets/ bases: {known or '(none)'}"
        )

    available = discover_views(base_dir)
    if not available:
        raise FileNotFoundError(
            f"base {base_dir} has no view PNGs; expected at least one of "
            f"{[f'{v}.png' for v in ALL_VIEWS]}"
        )

    if views is None:
        selected = available
    else:
        selected = list(views)
        missing = [v for v in selected if v not in available]
        if missing:
            raise FileNotFoundError(
                f"base {base_dir} is missing requested view(s) {missing}; "
                f"it has {available}"
            )

    view_set = ViewSet(tuple(selected))

    initial_state: Optional[dict] = None
    scene: Optional[str] = None
    tpl = base_dir / "template.yaml"
    if tpl.exists():
        loaded = yaml.safe_load(tpl.read_text()) or {}
        initial_state = loaded.get("initial_state")
        scene = loaded.get("scene")
    if initial_state is None:
        raise ValueError(
            f"base {base_dir} has no initial_state; add a template.yaml with an "
            "initial_state block (see assets/tri/template.yaml)."
        )

    return Base(
        dir=base_dir,
        view_set=view_set,
        initial_state=initial_state,
        scene=scene,
        name=base,
    )


def view_mapping_for(view_set: ViewSet, mapping: Dict[str, str]) -> Dict[str, str]:
    """Restrict a ``{view: filename}`` mapping to the views in ``view_set``."""
    missing = [v for v in view_set.views if v not in mapping]
    if missing:
        raise KeyError(f"no source filename mapped for view(s) {missing}")
    return {v: mapping[v] for v in view_set.views}
