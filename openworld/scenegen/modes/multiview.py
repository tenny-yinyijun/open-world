"""``multiview`` mode — nanobanana edits the wrist view, FLUX completes the sides.

The GPU mode. Unlike ``nanobanana`` (which edits views that already exist), this
one *synthesizes* the side views from an edited wrist view with the FLUX.2-klein
multiview model, so it can introduce a genuinely new object and have the side
cameras render it from a consistent 3-D-plausible viewpoint.

Because the generator is anchored on the wrist view by construction, this mode
only supports ``wrist_first``; ask for ``side_first`` and it raises rather than
silently ignoring the request.

Runs the bundled diffusers pipeline
(``external/diffusers/.../multiview_droid_with_nanobanana.py``) as an isolated
subprocess so FLUX's ~8 GB of VRAM is reclaimed between cases. The subprocess
imports the *fork's* diffusers (FLUX.2-klein lives only there): we prepend
``<diffusers-dir>/src`` to its ``PYTHONPATH`` so the fork wins over any installed
diffusers. Override the interpreter with ``python_exec`` if the fork has its own
venv.

Prereqs: ``GOOGLE_API_KEY``, the diffusers fork at ``external/diffusers``, the
checkpoint at ``checkpoints/multiview_droid_v0``, and a GPU.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from openworld.scenegen.guardrail import DEFAULT_KEEP, view_prompt
from openworld.scenegen.modes.base import CaseResult, Edit, SceneGenMode, register_mode
from openworld.scenegen.views import WRIST, WRIST_FIRST, Base, ViewSet

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIFFUSERS_DIR = REPO_ROOT / "external" / "diffusers"
DEFAULT_MULTIVIEW_SCRIPT = (
    DEFAULT_DIFFUSERS_DIR / "examples" / "inference" / "multiview_droid_with_nanobanana.py"
)
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "multiview_droid_v0"
DEFAULT_SIDES = [
    DEFAULT_DIFFUSERS_DIR / "assets" / "droid" / "side1.jpg",
    DEFAULT_DIFFUSERS_DIR / "assets" / "droid" / "side2.jpg",
]

# Which raw subprocess output backs each suite view. side1 -> exterior_left and
# side2 -> exterior_right matches the original 0617 authoring.
RAW_VIEW_FILES: Dict[str, str] = {
    WRIST: "edited_wrist.jpg",
    "exterior_left": "pred_side1.jpg",
    "exterior_right": "pred_side2.jpg",
}


@register_mode
class MultiviewMode(SceneGenMode):
    """Add-object scene generation: edited wrist view + FLUX-synthesized sides."""

    name = "multiview"
    description = (
        "nanobanana edits the wrist view, FLUX.2-klein synthesizes the side views "
        "(needs a GPU + the diffusers fork). Use to introduce a new object."
    )
    # A 2-view suite just keeps one of the two synthesized sides.
    supports_two_view = True

    def __init__(
        self,
        *,
        multiview_script: str = str(DEFAULT_MULTIVIEW_SCRIPT),
        diffusers_dir: str = str(DEFAULT_DIFFUSERS_DIR),
        checkpoint_path: str = str(DEFAULT_CHECKPOINT),
        python_exec: Optional[str] = None,
        side_cond: Optional[Sequence[str]] = None,
        num_inference_steps: int = 50,
        seed: int = 0,
        api_key_env: str = "GOOGLE_API_KEY",
        keep: Optional[str] = None,
        keep_raw: bool = False,
        verbose: bool = True,
        **params: Any,
    ) -> None:
        super().__init__(**params)
        self.script = Path(multiview_script).resolve()
        self.diffusers_dir = Path(diffusers_dir).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.python_exec = python_exec or sys.executable
        self.side_cond = [Path(s).resolve() for s in (side_cond or DEFAULT_SIDES)]
        self.num_inference_steps = int(num_inference_steps)
        self.seed = int(seed)
        self.api_key_env = api_key_env
        self.keep = keep
        self.keep_raw = keep_raw
        self.verbose = verbose
        self._case_counter = 0

    def preflight(self, *, base: Base, view_set: ViewSet) -> None:
        if not view_set.has_wrist:
            raise ValueError(
                "the multiview mode is anchored on the wrist view, but this suite's "
                f"view set has none: {view_set.describe()}"
            )
        if not os.environ.get(self.api_key_env):
            raise RuntimeError(
                f"{self.api_key_env} is not set; nanobanana cannot run. "
                "Export your Gemini API key first."
            )
        if not self.script.exists():
            raise FileNotFoundError(
                f"multiview script not found: {self.script}\n"
                "Clone the diffusers fork into external/diffusers (see "
                "external/README.md) or pass mode_params.multiview_script."
            )
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"multiview checkpoint not found: {self.checkpoint_path}\n"
                "Download it (bash external/download_models.sh) or pass "
                "mode_params.checkpoint_path."
            )

    def _run_subprocess(self, *, prompt: str, out_dir: Path, wrist_input: Path, seed: int) -> None:
        cmd = [
            self.python_exec, str(self.script),
            "--prompt", prompt,
            "--output_dir", str(out_dir),
            "--checkpoint_path", str(self.checkpoint_path),
            "--wrist_input", str(wrist_input),
            "--num_inference_steps", str(self.num_inference_steps),
            "--seed", str(seed),
        ]
        for s in self.side_cond:
            cmd += ["--side_cond", str(s)]

        env = os.environ.copy()
        env["GOOGLE_API_KEY"] = os.environ[self.api_key_env]
        fork_src = self.diffusers_dir / "src"
        if fork_src.is_dir():
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                f"{fork_src}{os.pathsep}{existing}" if existing else str(fork_src)
            )

        proc = subprocess.run(cmd, env=env, cwd=str(self.diffusers_dir))
        if proc.returncode != 0:
            raise RuntimeError(f"multiview subprocess failed (exit {proc.returncode})")

    def generate_case(
        self,
        *,
        base: Base,
        edit: Edit,
        case_dir: Path,
        edit_order: str = WRIST_FIRST,
    ) -> CaseResult:
        requested = edit.edit_order or edit_order
        if requested != WRIST_FIRST:
            raise ValueError(
                f"the multiview mode synthesizes side views *from* the wrist view, so "
                f"it only supports edit_order='{WRIST_FIRST}' (got '{requested}'). "
                "Use the nanobanana mode for side-anchored edits."
            )

        view_set = base.view_set
        keep_clause = edit.keep if edit.keep is not None else (self.keep or DEFAULT_KEEP)
        # The subprocess edits the wrist view, so it gets the wrist geometry clause.
        prompt = view_prompt(edit.prompt, WRIST, keep=keep_clause)
        seed = self.seed + self._case_counter
        self._case_counter += 1

        raw_dir = case_dir / "_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        if self.verbose:
            print(f"  [multiview] seed={seed} anchor=wrist -> {view_set.describe()}")

        self._run_subprocess(
            prompt=prompt,
            out_dir=raw_dir,
            wrist_input=base.path(WRIST),
            seed=seed,
        )

        # Collect only the views this suite actually wants (a 2-view suite drops
        # one synthesized side).
        view_images: Dict[str, Any] = {}
        missing: List[str] = []
        for view in view_set.views:
            src = raw_dir / RAW_VIEW_FILES[view]
            if src.exists():
                view_images[view] = src
            else:
                missing.append(f"{view} ({src.name})")
        if missing:
            raise FileNotFoundError(
                f"multiview output missing for {missing} in {raw_dir}"
            )

        result = CaseResult(
            view_images=view_images,
            in_place=False,
            prompts={WRIST: prompt},
            metadata={
                "edit_mode": self.name,
                "edit_order": WRIST_FIRST,
                "anchor_view": WRIST,
                "edit_chain": [WRIST] + list(view_set.sides),
                "seed": seed,
                "num_inference_steps": self.num_inference_steps,
                "keep": keep_clause,
                "source_image": str(base.path(WRIST)),
            },
        )
        # write_case copies the raw images in; drop the full-res intermediates
        # after, unless asked to keep them.
        if not self.keep_raw:
            result.metadata["_cleanup_raw"] = str(raw_dir)
        return result

    @staticmethod
    def cleanup(case_dir: Path) -> None:
        shutil.rmtree(case_dir / "_raw", ignore_errors=True)
