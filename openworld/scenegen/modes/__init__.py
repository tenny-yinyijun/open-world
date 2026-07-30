"""Scene-generation modes: pluggable backends that edit a base into a scene.

A **mode** takes a resolved :class:`~openworld.scenegen.views.Base` plus one
*edit* (a prompt + the policy instruction) and produces the edited view images
for one case. Suite assembly, the eval-format ``initialization.yaml``, and the
provenance manifest are shared by all modes and live in
:mod:`openworld.scenegen.suite` — a mode only makes pictures.

Modes available today:

================  ============================================  ==================
name              what it does                                  needs
================  ============================================  ==================
``nanobanana``    chained per-view edit with Gemini 2.5 Flash    ``GOOGLE_API_KEY``
                  Image; anchor view first, later views
                  conditioned on it. Either anchor order.
``multiview``     nanobanana edits the wrist view, FLUX.2-klein  GPU + diffusers
                  synthesizes the sides. Adds new objects.       fork + checkpoint
                  ``wrist_first`` only.
``copy``          no-op passthrough of the base views (a         nothing
                  control / smoke-test mode)
================  ============================================  ==================

Register a new mode by subclassing :class:`SceneGenMode` and calling
:func:`register_mode`; :func:`get_mode` resolves it by name for the CLI, so a
new backend needs no changes in the suite builder.
"""

from __future__ import annotations

from openworld.scenegen.modes.base import (
    MODES,
    SceneGenMode,
    available_modes,
    get_mode,
    register_mode,
)
from openworld.scenegen.modes.copy_mode import CopyMode
from openworld.scenegen.modes.multiview import MultiviewMode
from openworld.scenegen.modes.nanobanana import NanobananaMode

__all__ = [
    "SceneGenMode",
    "MODES",
    "register_mode",
    "get_mode",
    "available_modes",
    "NanobananaMode",
    "MultiviewMode",
    "CopyMode",
]
