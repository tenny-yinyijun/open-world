"""Mode-agnostic suite builder: base + edits + a mode -> an eval-ready suite.

This is the orchestration layer. It resolves the base and view set, picks the
scene-generation mode, runs it once per edit, and writes each result as an
Initialization case via :mod:`openworld.scenegen.suite`. Adding a new generative
mode requires no change here — only a :func:`~openworld.scenegen.modes.base.register_mode`
call.

    from openworld.scenegen import build_suite_from_spec
    build_suite_from_spec("configs/scenegen/suites/example_2view.yaml")

Spec keys (all optional unless noted):

======================  ====================================================
``base``  (required)    ``tri`` / ``irom`` under ``assets/``, or a path
``edits`` (required)    list of per-case edits (prompt + instruction + ...)
``name`` / ``out_dir``  output location (``data/benchmark/<name>``)
``views``               view subset, e.g. ``[exterior_right, wrist]`` for the
                        2-view model; defaults to every view the base has
``mode``                scene-generation mode (default ``nanobanana``)
``mode_params``         kwargs passed to the mode's constructor
``edit_order``          ``wrist_first`` (default) or ``side_first``
``keep``                shared "keep everything else" clause
``include_base_case``   prepend an unedited control case as ``init_<start>``
``scene``/``task_type`` metadata tags
``start_index``         first case index (append to an existing suite)
======================  ====================================================
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from openworld.scenegen.modes import get_mode
from openworld.scenegen.modes.base import Edit, SceneGenMode
from openworld.scenegen.suite import (
    BENCHMARK_ROOT,
    copy_base_case,
    resolve_suite_dir,
    verify_suite,
    write_case,
    write_manifest,
)
from openworld.scenegen.views import (
    DEFAULT_TARGET_SIZE,
    EDIT_ORDERS,
    WRIST_FIRST,
    resolve_base,
)


def build_suite(
    *,
    base: str,
    edits: Sequence[dict],
    name: Optional[str] = None,
    out_dir: Optional[str] = None,
    views: Optional[Sequence[str]] = None,
    mode: str = "nanobanana",
    mode_params: Optional[Dict[str, Any]] = None,
    edit_order: str = WRIST_FIRST,
    keep: Optional[str] = None,
    include_base_case: bool = False,
    task_type: str = "manipulation",
    scene: Optional[str] = None,
    start_index: int = 0,
    target_size = DEFAULT_TARGET_SIZE,
    benchmark_root: Path = BENCHMARK_ROOT,
    verify: bool = True,
    verbose: bool = True,
) -> List[Path]:
    """Build an Initialization suite by running ``mode`` over ``edits``.

    Each edit becomes one ``init_<i>`` case. Per-case failures abort the run so a
    directed request fails loudly rather than silently producing fewer cases.

    Returns the list of written case directories.
    """
    if not edits:
        raise ValueError("`edits` is empty; nothing to build")
    if edit_order not in EDIT_ORDERS:
        raise ValueError(
            f"unknown edit_order '{edit_order}' (expected one of {list(EDIT_ORDERS)})"
        )

    suite_dir = resolve_suite_dir(name=name, out_dir=out_dir, root=benchmark_root)
    resolved_base = resolve_base(base, views=views)
    view_set = resolved_base.view_set

    engine: SceneGenMode = get_mode(mode, **(mode_params or {}))
    if view_set.num_cams < 3 and not engine.supports_two_view:
        raise ValueError(
            f"mode '{mode}' requires the full 3-view set, but this suite is "
            f"{view_set.describe()}"
        )
    engine.preflight(base=resolved_base, view_set=view_set)

    suite_scene = scene or resolved_base.scene or "scenegen"
    suite_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(
            f"[scenegen] mode={mode} {view_set.describe()} edit_order={edit_order} "
            f"-> {suite_dir}"
        )

    written: List[Path] = []
    manifest_cases: List[Dict[str, Any]] = []
    idx = start_index

    if include_base_case:
        case_id = f"init_{idx}"
        if verbose:
            print(f"[scenegen] {case_id} (unedited control): copying base views ...")
        case_dir = copy_base_case(
            case_dir=suite_dir / case_id,
            base=resolved_base,
            instruction=Edit.from_dict(dict(edits[0]), 0).instruction,
            metadata={
                "suite": suite_dir.name,
                "scene": suite_scene,
                "task_type": task_type,
                "case_id": case_id,
                "state_length": 7,
                "base": base,
                "views": list(view_set.views),
            },
            target_size=target_size,
        )
        written.append(case_dir)
        manifest_cases.append({"case_id": case_id, "label": "original_no_change"})
        idx += 1

    for i, raw in enumerate(edits):
        edit = Edit.from_dict(dict(raw), i)
        case_id = f"init_{idx + i}"
        case_dir = suite_dir / case_id
        if verbose:
            print(f"[scenegen] {case_id} ({edit.label or 'edit'}):")

        result = engine.generate_case(
            base=resolved_base,
            edit=edit,
            case_dir=case_dir,
            edit_order=edit_order,
        )

        metadata: Dict[str, Any] = {
            "suite": suite_dir.name,
            "scene": suite_scene,
            "task_type": task_type,
            "case_id": case_id,
            "state_length": 7,
            "base": base,
            "views": list(view_set.views),
            "edit_label": edit.label,
            "edit_prompt": edit.prompt,
        }
        metadata.update(result.metadata)
        if result.prompts:
            metadata["view_prompts"] = result.prompts
        # Modes signal throwaway intermediates via a private key; it must not
        # leak into the case's metadata.
        raw_to_clean = metadata.pop("_cleanup_raw", None)

        write_case(
            case_dir=case_dir,
            view_set=view_set,
            initial_state=resolved_base.initial_state,
            instruction=edit.instruction,
            metadata=metadata,
            view_sources=None if result.in_place else result.view_images,
            target_size=target_size,
        )
        if raw_to_clean:
            shutil.rmtree(raw_to_clean, ignore_errors=True)
        written.append(case_dir)
        manifest_cases.append(
            {
                "case_id": case_id,
                "label": edit.label,
                "instruction": edit.instruction,
                "prompt": edit.prompt,
                "anchor_view": result.metadata.get("anchor_view"),
                "edit_chain": result.metadata.get("edit_chain"),
            }
        )
        if verbose:
            print(f"[scenegen] {case_id}: ok -> {case_dir}")

    write_manifest(
        suite_dir,
        {
            "base": base,
            "base_dir": str(resolved_base.dir),
            "views": list(view_set.views),
            "num_cams": view_set.num_cams,
            "mode": mode,
            "edit_order": edit_order,
            "scene": suite_scene,
            "task_type": task_type,
            "keep": keep,
            "include_base_case": include_base_case,
            "num_cases": len(written),
            "cases": manifest_cases,
        },
    )

    if verify:
        # Load the suite back through the real eval loader; a suite that cannot
        # be read by run_evaluation.py is a build failure, not a later surprise.
        ids = verify_suite(suite_dir, expect_cases=len(written))
        if verbose:
            print(f"[scenegen] verified via InitializationDataset: {ids}")

    if verbose:
        print(f"[scenegen] wrote {len(written)} case(s) -> {suite_dir}")
    return written


def build_suite_from_spec(spec_path: str, **overrides) -> List[Path]:
    """Load a YAML suite spec and build it. ``overrides`` win over spec keys."""
    spec = yaml.safe_load(Path(spec_path).read_text()) or {}
    spec.update(overrides)
    edits = spec.pop("edits", None)
    if not edits:
        raise ValueError(f"spec {spec_path} has no 'edits' list")

    # `keep` is a spec-level default for the mode; pass it through to the mode
    # while also recording it in the manifest.
    keep = spec.get("keep")
    if keep:
        mode_params = dict(spec.get("mode_params") or {})
        mode_params.setdefault("keep", keep)
        spec["mode_params"] = mode_params

    return build_suite(edits=edits, **spec)
