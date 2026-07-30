"""scenegen: build world-model test cases from language + images.

Produces an Initialization *suite* — the handoff format consumed by
``scripts/run_evaluation.py`` and ``scripts/generate_videos.py`` — from a *base*
(an unedited scene) plus a list of *edits*.

Four layers, so a new generative backend only touches the first:

1. **modes** (:mod:`openworld.scenegen.modes`) — the pluggable generative half.
   ``nanobanana`` (Gemini 2.5 Flash Image, no GPU) and ``copy`` (unedited
   control) ship today; register more with ``@register_mode``.
2. **guardrail** (:mod:`openworld.scenegen.guardrail`) — the prompt layer: rewrite
   a plain instruction into an editor-ready prompt, then specialize it per camera
   and chain views so they stay mutually consistent.
3. **views** (:mod:`openworld.scenegen.views`) — which cameras a suite has and in
   what order they get edited. Supports both the 3-view DROID set and the 2-view
   (one side + wrist) set the published ``wm_student_2view`` model uses, and both
   ``wrist_first`` and ``side_first`` anchor orders.
4. **suite** (:mod:`openworld.scenegen.suite`) — the on-disk eval contract, and
   :func:`~openworld.scenegen.suite.verify_suite`, which loads a built suite back
   through the real ``InitializationDataset`` as an acceptance check.

Entry points:
  - CLI:    ``python scripts/scenegen/build_suite.py --spec <spec.yaml>``
  - module: ``from openworld.scenegen import build_suite, build_suite_from_spec``
  - prompt: ``from openworld.scenegen.guardrail import build_edit_prompt``
"""

from openworld.scenegen.builder import build_suite, build_suite_from_spec
from openworld.scenegen.guardrail import build_edit_prompt, view_prompt
from openworld.scenegen.modes import (
    MODES,
    CopyMode,
    NanobananaMode,
    SceneGenMode,
    available_modes,
    get_mode,
    register_mode,
)
from openworld.scenegen.modes.base import Edit
from openworld.scenegen.modes.nanobanana import nanobanana_edit
from openworld.scenegen.suite import verify_suite, write_case
from openworld.scenegen.views import (
    ALL_VIEWS,
    SIDE_FIRST,
    VIEWS_2,
    VIEWS_3,
    WRIST_FIRST,
    Base,
    ViewSet,
    resolve_base,
)

__all__ = [
    # building
    "build_suite",
    "build_suite_from_spec",
    "Edit",
    # modes
    "SceneGenMode",
    "MODES",
    "register_mode",
    "get_mode",
    "available_modes",
    "NanobananaMode",
    "CopyMode",
    "nanobanana_edit",
    # prompts
    "build_edit_prompt",
    "view_prompt",
    # views
    "ViewSet",
    "Base",
    "resolve_base",
    "ALL_VIEWS",
    "VIEWS_2",
    "VIEWS_3",
    "WRIST_FIRST",
    "SIDE_FIRST",
    # suite / eval contract
    "write_case",
    "verify_suite",
]
