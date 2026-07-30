"""Write generated scenes as an Initialization suite the eval scripts can load.

Every scene-generation mode ends here. This module owns the on-disk contract
with :class:`openworld.datasets.InitializationDataset` — the format
``scripts/run_evaluation.py`` and ``scripts/generate_videos.py`` consume — so a
new mode only has to produce images and hand them over.

The contract, as enforced by ``InitializationDataset``:

* the suite root holds one ``init_<i>/`` directory per case, each with an
  ``initialization.yaml``; the directory name becomes the case ``id``.
* each ``initialization.yaml`` is a mapping with ``initial_state`` (the robot
  start pose, cloned from the base's ``template.yaml``), plus optional
  ``instruction`` and ``metadata``.
* views are ``<view>.png`` at the world-model resolution (320x192).

**Why we always write ``initial_observation`` explicitly.** The dataset loader
can *infer* the observation from a case directory, but
``InitializationDataset._infer_observation_from_case_dir`` only does so when all
three of ``wrist``/``exterior_left``/``exterior_right`` are present — it returns
``None`` for a 2-view case, which would leave ``initial_observation`` unset and
break the rollout. So we write an explicit ``initial_observation.views`` block
listing exactly this suite's views. That makes 2-view suites first-class, and
also pins view *order* rather than relying on directory inference.

The view names and their order match the ``view_order`` in the eval configs
(``exterior_right`` -> ``exterior_left`` -> ``wrist``, sides before wrist), which
is the order the world model height-stacks cameras when bootstrapping history
(``openworld/world_models/ar_world_model.py``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml
from PIL import Image

from openworld.scenegen.views import DEFAULT_TARGET_SIZE, ViewSet

# Suites built from a spec land here by default.
REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = REPO_ROOT / "data" / "benchmark"
# Suites built from a single instruction+image land here by default.
INITIALIZATIONS_ROOT = REPO_ROOT / "data" / "initializations"


def resolve_suite_dir(
    *,
    name: Optional[str] = None,
    out_dir: Optional[str] = None,
    root: Path = BENCHMARK_ROOT,
) -> Path:
    """Resolve the suite output directory from ``out_dir`` or ``name``."""
    if out_dir:
        return Path(out_dir).expanduser().resolve()
    if name:
        return (Path(root) / name).resolve()
    raise ValueError(f"provide either `name` (-> {root}/<name>) or `out_dir`")


def save_view(
    src: Any,
    dst: Path,
    *,
    target_size: Optional[Tuple[int, int]] = DEFAULT_TARGET_SIZE,
) -> None:
    """Write one view PNG, resized to the world-model resolution.

    ``src`` is a path or a PIL image. ``target_size`` is ``(W, H)``; pass
    ``None`` to keep the source resolution.
    """
    img = src if isinstance(src, Image.Image) else Image.open(str(src))
    img = img.convert("RGB")
    if target_size is not None:
        w, h = int(target_size[0]), int(target_size[1])
        if img.size != (w, h):
            img = img.resize((w, h), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst)


def write_case(
    *,
    case_dir: Path,
    view_set: ViewSet,
    initial_state: dict,
    instruction: str,
    metadata: Optional[Dict[str, Any]] = None,
    view_sources: Optional[Dict[str, Any]] = None,
    target_size: Optional[Tuple[int, int]] = DEFAULT_TARGET_SIZE,
) -> Path:
    """Write one ``init_<i>/`` case: view PNGs + ``initialization.yaml``.

    ``view_sources`` maps each view in ``view_set`` to a source path or PIL
    image to copy in. Pass ``None`` when the PNGs are already in ``case_dir``
    (the mode wrote them there itself) — they are then validated, not rewritten.

    Returns ``case_dir``.
    """
    case_dir.mkdir(parents=True, exist_ok=True)

    if view_sources is not None:
        missing = [v for v in view_set.views if v not in view_sources]
        if missing:
            raise KeyError(f"view_sources is missing view(s) {missing}")
        for view in view_set.views:
            save_view(view_sources[view], case_dir / f"{view}.png", target_size=target_size)

    absent = [v for v in view_set.views if not (case_dir / f"{v}.png").exists()]
    if absent:
        raise FileNotFoundError(
            f"case {case_dir} is missing view PNG(s) for {absent}; "
            "a scene-generation mode must produce every view in the view set."
        )

    init: Dict[str, Any] = {
        "initial_state": initial_state,
        # Explicit rather than inferred: the loader only infers when all three
        # DROID views exist, so a 2-view case needs this block. Relative paths
        # are resolved against the case directory by the loader.
        "initial_observation": {
            "views": {v: f"{v}.png" for v in view_set.views}
        },
        "instruction": instruction,
    }
    if metadata:
        init["metadata"] = metadata

    with open(case_dir / "initialization.yaml", "w") as f:
        yaml.safe_dump(init, f, sort_keys=False)
    return case_dir


def write_manifest(suite_dir: Path, manifest: Dict[str, Any]) -> Path:
    """Write ``scenegen_manifest.json`` (provenance for how a suite was built)."""
    path = suite_dir / "scenegen_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def verify_suite(suite_dir: Path, *, expect_cases: Optional[int] = None) -> List[str]:
    """Load ``suite_dir`` through the real eval loader and check every case.

    This is the actual acceptance test for "the eval scripts can take this":
    it runs ``InitializationDataset.from_yaml`` and asserts each entry has a
    usable ``initial_state`` and an ``initial_observation`` whose view files all
    exist on disk. Returns the case ids in dataset order.
    """
    from openworld.datasets import InitializationDataset

    dataset = InitializationDataset.from_yaml(str(suite_dir))
    if expect_cases is not None and len(dataset) != expect_cases:
        raise AssertionError(
            f"{suite_dir}: expected {expect_cases} case(s), loader found {len(dataset)}"
        )

    ids: List[str] = []
    for init in dataset:
        if not init.initial_state:
            raise AssertionError(f"{suite_dir}/{init.id}: empty initial_state")
        obs = init.initial_observation
        if not isinstance(obs, dict) or not obs.get("views"):
            raise AssertionError(
                f"{suite_dir}/{init.id}: initial_observation has no views block "
                "(the eval rollout cannot bootstrap history without it)"
            )
        for view, path in obs["views"].items():
            if not Path(path).exists():
                raise AssertionError(
                    f"{suite_dir}/{init.id}: view '{view}' points at a missing "
                    f"file: {path}"
                )
        ids.append(init.id)
    return ids


def copy_base_case(
    *,
    case_dir: Path,
    base,
    instruction: str,
    metadata: Optional[Dict[str, Any]] = None,
    target_size: Optional[Tuple[int, int]] = DEFAULT_TARGET_SIZE,
) -> Path:
    """Write an unedited control case — the base views copied through verbatim.

    Useful as ``init_0`` of a suite: a zero-generative-drift reference the edited
    cases are compared against.
    """
    meta = {"edit_mode": "base_copy", "edit_label": "original_no_change", "edit_prompt": ""}
    meta.update(metadata or {})
    return write_case(
        case_dir=case_dir,
        view_set=base.view_set,
        initial_state=base.initial_state,
        instruction=instruction,
        metadata=meta,
        view_sources={v: base.path(v) for v in base.view_set.views},
        target_size=target_size,
    )
