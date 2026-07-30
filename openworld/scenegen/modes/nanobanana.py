"""nanobanana scene-generation mode — chained per-view edits, no GPU.

Edits each of a base's real views with nanobanana (Gemini 2.5 Flash Image). It
needs only ``GOOGLE_API_KEY``, so it runs anywhere; the scene *content* stays put
and only what the edit prompt targets changes. That makes it the right mode for
background / lighting / material / object edits on an existing scene.

**Views are edited as a chain, not independently.** The anchor view is edited
first from the prompt alone; every later view is edited with the anchor's result
(and any views already done) passed in as extra reference images, plus the
"reproduce that same edited scene from this viewpoint" clause from
:func:`openworld.scenegen.guardrail.chain_prompt`. Editing views independently
makes nanobanana re-invent an added object differently in each camera; chaining
is what keeps the cameras consistent.

Which view anchors is set by ``edit_order``:

* ``wrist_first`` — wrist (top-down) anchors, then the sides. Best for object
  add / remove / replace: the overhead view pins placement on the table.
* ``side_first`` — an exterior (side) view anchors, then the wrist. Best for
  background / lighting / room edits: the side view is where the room is.

Both work for a 2-view suite (one side + wrist) and a 3-view suite.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from openworld.scenegen.guardrail import DEFAULT_KEEP, view_prompt
from openworld.scenegen.modes.base import CaseResult, Edit, SceneGenMode, register_mode
from openworld.scenegen.views import WRIST_FIRST, Base, ViewSet

MODEL = os.environ.get("NANOBANANA_MODEL", "gemini-2.5-flash-image")
# Editing a 320x192 world-model frame directly makes nanobanana reframe/warp, so
# we upscale, edit at higher resolution, then resize back to native.
EDIT_WIDTH = int(os.environ.get("NANOBANANA_EDIT_WIDTH", "1024"))

_client = None


def _get_client(api_key_env: str = "GOOGLE_API_KEY"):
    global _client
    if _client is None:
        from google import genai  # imported lazily: only needed when editing

        _client = genai.Client(api_key=os.environ[api_key_env])
    return _client


def _upscale(img: Image.Image) -> Image.Image:
    w, h = img.size
    return img.resize((EDIT_WIDTH, round(EDIT_WIDTH * h / w)), Image.LANCZOS)


def nanobanana_edit(
    src: str,
    dst: str,
    prompt: str,
    *,
    references: Optional[List[Any]] = None,
    api_key_env: str = "GOOGLE_API_KEY",
    verbose: bool = True,
) -> Image.Image:
    """Edit ``src`` with ``prompt`` and write the result to ``dst`` at src's size.

    ``references`` are extra images (paths or PIL images) appended after the
    edited image — used to condition a follower view on the already-edited anchor
    view. Requires ``GOOGLE_API_KEY`` + ``uv sync --extra scenegen``.

    Returns the edited image (also written to ``dst``).
    """
    img = Image.open(src).convert("RGB") if not isinstance(src, Image.Image) else src.convert("RGB")
    w, h = img.size

    contents: List[Any] = [prompt, _upscale(img)]
    for ref in references or []:
        ref_img = ref if isinstance(ref, Image.Image) else Image.open(str(ref))
        contents.append(_upscale(ref_img.convert("RGB")))

    resp = _get_client(api_key_env).models.generate_content(model=MODEL, contents=contents)
    for part in resp.candidates[0].content.parts:
        if getattr(part, "inline_data", None) is not None:
            out = Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
            out = out.resize((w, h), Image.LANCZOS)
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            out.save(dst)
            if verbose:
                name = src if isinstance(src, str) else "<image>"
                print(f"  edited {os.path.basename(str(name))} -> {dst}  ({w}x{h})")
            return out
        if getattr(part, "text", None) and verbose:
            print(f"  [nanobanana text] {part.text}")
    raise RuntimeError(f"nanobanana returned no image for {src}")


@register_mode
class NanobananaMode(SceneGenMode):
    """Chained per-view scene editing with Gemini 2.5 Flash Image."""

    name = "nanobanana"
    description = (
        "Chained per-view edit with Gemini 2.5 Flash Image (needs GOOGLE_API_KEY, "
        "no GPU). Anchor view first, later views conditioned on it."
    )
    supports_two_view = True

    def __init__(
        self,
        *,
        api_key_env: str = "GOOGLE_API_KEY",
        keep: Optional[str] = None,
        chain: bool = True,
        verbose: bool = True,
        **params: Any,
    ) -> None:
        super().__init__(**params)
        self.api_key_env = api_key_env
        self.keep = keep
        # chain=False edits every view independently from the base (faster, but
        # the views can disagree). Kept as an escape hatch for pure global edits
        # like a uniform color tint, where cross-view drift is not a risk.
        self.chain = chain
        self.verbose = verbose

    def preflight(self, *, base: Base, view_set: ViewSet) -> None:
        if not os.environ.get(self.api_key_env):
            raise RuntimeError(
                f"{self.api_key_env} is not set; nanobanana cannot run. "
                "Export your Gemini API key first (uv sync --extra scenegen)."
            )

    def generate_case(
        self,
        *,
        base: Base,
        edit: Edit,
        case_dir: Path,
        edit_order: str = WRIST_FIRST,
    ) -> CaseResult:
        view_set = base.view_set
        order = view_set.edit_sequence(edit.edit_order or edit_order)
        # Views the caller asked to leave alone are copied from the base as-is.
        targeted = set(edit.views) if edit.views else set(view_set.views)
        keep_clause = edit.keep if edit.keep is not None else (self.keep or DEFAULT_KEEP)

        case_dir.mkdir(parents=True, exist_ok=True)
        edited: Dict[str, Image.Image] = {}
        prompts: Dict[str, str] = {}
        done: List[str] = []

        for view in order:
            src = base.path(view)
            dst = case_dir / f"{view}.png"
            if view not in targeted:
                Image.open(src).convert("RGB").save(dst)
                continue

            references = [edited[v] for v in done] if self.chain else []
            prompt = view_prompt(
                edit.prompt,
                view,
                keep=keep_clause,
                reference_views=tuple(done) if self.chain else (),
            )
            prompts[view] = prompt
            if self.verbose:
                anchor = " (anchor)" if not done else f" (matching {', '.join(done)})"
                print(f"  [{view}]{anchor}")
            edited[view] = nanobanana_edit(
                str(src),
                str(dst),
                prompt,
                references=references,
                api_key_env=self.api_key_env,
                verbose=self.verbose,
            )
            done.append(view)

        return CaseResult(
            in_place=True,
            prompts=prompts,
            metadata={
                "edit_mode": self.name,
                "edit_order": edit.edit_order or edit_order,
                # The anchor is the first view actually *edited* — a view the edit
                # skipped is copied from the base and anchors nothing.
                "anchor_view": done[0] if done else None,
                "edit_chain": done,
                "chained": self.chain,
                "edited_views": sorted(targeted & set(view_set.views)),
                "keep": keep_clause,
                "model": MODEL,
            },
        )
