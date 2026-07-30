"""``copy`` mode — pass the base views through unedited.

No model, no network, no key. Two uses:

* build a control case (or a whole control suite) with zero generative drift, to
  compare edited cases against;
* smoke-test the suite plumbing end to end — view set, ``initialization.yaml``,
  and eval-loader compatibility — without spending API calls.

The ``prompt`` on an edit is ignored (recorded in metadata for provenance);
``instruction`` still matters, since that is what the policy is asked to do.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from openworld.scenegen.modes.base import CaseResult, Edit, SceneGenMode, register_mode
from openworld.scenegen.views import WRIST_FIRST, Base


@register_mode
class CopyMode(SceneGenMode):
    """Copy the base views through verbatim (control / smoke-test mode)."""

    name = "copy"
    description = "Copy base views through unedited — control case, no model, no key."
    supports_two_view = True

    def generate_case(
        self,
        *,
        base: Base,
        edit: Edit,
        case_dir: Path,
        edit_order: str = WRIST_FIRST,
    ) -> CaseResult:
        case_dir.mkdir(parents=True, exist_ok=True)
        for view in base.view_set.views:
            Image.open(base.path(view)).convert("RGB").save(case_dir / f"{view}.png")
        return CaseResult(
            in_place=True,
            metadata={
                "edit_mode": self.name,
                "edit_label": edit.label or "original_no_change",
                "requested_prompt": edit.prompt,
            },
        )
