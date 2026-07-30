"""Back-compat shim — the FLUX multiview path is now the ``multiview`` *mode*.

``generate_test_cases`` used to be a standalone pipeline that hard-coded three
views and wrote its own ``initialization.yaml``. It is now a thin wrapper over
:func:`openworld.scenegen.builder.build_suite` with ``mode="multiview"``, so it
shares the view-set handling, the per-view prompt layer, and — importantly — the
suite writer that emits an explicit ``initial_observation`` block and verifies
the result against the eval loader.

Prefer :func:`openworld.scenegen.build_suite` directly. The moved names:

========================================  ====================================
old                                       new
========================================  ====================================
``pipeline.generate_test_cases``          ``builder.build_suite(mode="multiview")``
``pipeline.DEFAULT_MULTIVIEW_SCRIPT``     ``modes.multiview.DEFAULT_MULTIVIEW_SCRIPT``
``pipeline.DEFAULT_CHECKPOINT``           ``modes.multiview.DEFAULT_CHECKPOINT``
========================================  ====================================
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from openworld.scenegen.builder import build_suite
from openworld.scenegen.guardrail import DEFAULT_GEMINI_MODEL, build_edit_prompt
from openworld.scenegen.modes.multiview import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DIFFUSERS_DIR,
    DEFAULT_MULTIVIEW_SCRIPT,
    DEFAULT_SIDES,
)
from openworld.scenegen.suite import INITIALIZATIONS_ROOT as DEFAULT_SUITE_ROOT
from openworld.scenegen.views import DEFAULT_TARGET_SIZE, WRIST_FIRST

DEFAULT_TEMPLATE_INIT = (
    Path(__file__).resolve().parents[2] / "configs" / "scenegen" / "template_initialization.yaml"
)

__all__ = [
    "generate_test_cases",
    "DEFAULT_MULTIVIEW_SCRIPT",
    "DEFAULT_DIFFUSERS_DIR",
    "DEFAULT_CHECKPOINT",
    "DEFAULT_SIDES",
    "DEFAULT_TEMPLATE_INIT",
    "DEFAULT_SUITE_ROOT",
    "DEFAULT_TARGET_SIZE",
]


def generate_test_cases(
    *,
    instruction: str,
    init_image: Optional[str] = None,
    base: str = "tri",
    views: Optional[Sequence[str]] = None,
    name: Optional[str] = None,
    out_suite: Optional[str] = None,
    num_cases: int = 1,
    scene_edit: Optional[str] = None,
    guardrail_backend: str = "gemini",
    guardrail_model: str = DEFAULT_GEMINI_MODEL,
    multiview_script: str = str(DEFAULT_MULTIVIEW_SCRIPT),
    diffusers_dir: str = str(DEFAULT_DIFFUSERS_DIR),
    python_exec: Optional[str] = None,
    checkpoint_path: str = str(DEFAULT_CHECKPOINT),
    side_cond: Optional[Sequence[str]] = None,
    num_inference_steps: int = 50,
    seed: int = 0,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
    scene: str = "scenegen",
    task_type: str = "manipulation",
    start_index: int = 0,
    google_api_key_env: str = "GOOGLE_API_KEY",
    keep_raw: bool = False,
    verify: bool = True,
    verbose: bool = True,
    **_ignored,
) -> List[Path]:
    """Generate ``num_cases`` multiview cases from one instruction (+ base).

    ``init_image`` is accepted for backwards compatibility: if given, its parent
    directory is used as the base, since a base is now a directory of views plus
    a ``template.yaml`` rather than a single loose wrist image.
    """
    if init_image and base == "tri":
        base = str(Path(init_image).resolve().parent)

    edit_prompt = build_edit_prompt(
        scene_edit or instruction,
        backend=guardrail_backend,
        model=guardrail_model,
        api_key_env=google_api_key_env,
        verbose=verbose,
    )
    # Cases differ only via the seed, which the mode advances per case.
    edits = [
        {"prompt": edit_prompt, "instruction": instruction, "label": f"case_{i}"}
        for i in range(num_cases)
    ]

    return build_suite(
        base=base,
        views=views,
        edits=edits,
        name=name,
        out_dir=out_suite,
        mode="multiview",
        mode_params={
            "multiview_script": multiview_script,
            "diffusers_dir": diffusers_dir,
            "python_exec": python_exec,
            "checkpoint_path": checkpoint_path,
            "side_cond": side_cond,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "api_key_env": google_api_key_env,
            "keep_raw": keep_raw,
            "verbose": verbose,
        },
        edit_order=WRIST_FIRST,
        scene=scene,
        task_type=task_type,
        start_index=start_index,
        target_size=target_size,
        benchmark_root=DEFAULT_SUITE_ROOT,
        verify=verify,
        verbose=verbose,
    )
