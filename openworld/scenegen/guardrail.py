"""Guardrail prompt layer: plain instruction -> per-view image-edit prompts.

Image editors are phrasing-sensitive. A bare ``"put the carrot in the bowl"``
makes nanobanana shift the camera, move the gripper, or occlude objects, and —
critically for a multi-view suite — it produces a *different* scene in each view
if each view is edited independently. This module owns the phrasing that
prevents both.

Three layers, smallest to largest:

1. :func:`build_edit_prompt` — rewrite a user instruction into one editor-ready
   edit prompt. Backends: ``gemini`` (a text LLM rewrite following
   :data:`SYSTEM_PROMPT`) or ``template`` (deterministic, offline).
2. :func:`view_prompt` — specialize an edit prompt for *one camera*. The wrist
   camera is top-down (bird's-eye); the exterior cameras look at the table from
   the side at an angle. A prompt that says "seen from directly above" is wrong
   for a side view and vice versa, so each view gets its own geometry clause.
3. :func:`chain_prompt` — the clause that makes view *N* agree with the views
   already edited. Editing views in a chain (anchor first, then the rest
   conditioned on it) is what keeps a generated object from being re-invented
   differently in every camera.

**Anchor choice.** ``wrist_first`` anchors on the overhead view — best for
object add/remove/replace, where exact placement on the table matters most.
``side_first`` anchors on an exterior view — best for background / lighting /
room edits, since the side view is where the room actually is. The anchor gets
the full descriptive prompt; followers get a "match what is already in this
other view" prompt plus a reference image.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

from openworld.scenegen.views import WRIST

# Default model for the LLM-backed guardrail. Text-only Gemini (not the image
# model) — it reuses the same GOOGLE_API_KEY / google-genai SDK as nanobanana.
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

# Rules the guardrail enforces when rewriting a user instruction. Deliberately
# viewpoint-NEUTRAL: `view_prompt` adds the per-camera geometry afterwards, so
# baking one camera's framing in here would corrupt every other view.
SYSTEM_PROMPT = """\
You rewrite a robot task/scene description into a single image-EDIT instruction \
for an image-editing model. The image being edited is one camera view of a \
robot tabletop workspace.

Rewrite the user's instruction into ONE concise edit instruction that obeys ALL \
of these rules:
- Describe ONLY the objects to place/modify on the table: their color, size, \
count, material, and where each sits relative to the others and to the table.
- Describe object placement in terms that are true from ANY viewpoint (e.g. \
"to the left of the bowl", "near the front edge of the table"), NOT in terms of \
one specific camera's framing. Do not mention pixels, image corners, or crops, \
and do not say "top-down" or "from the side" — the viewpoint is added later.
- Keep every object fully visible and non-overlapping. No object may hide, \
cover, or partially block another, and none may be hidden by the robot arm or \
gripper. Do NOT introduce occlusion.
- Do not change the robot arm or gripper, and do not move the camera.
- Do not mention lighting or photographic style unless the user asked to change \
it. Do not invent objects the user did not ask for.

Output ONLY the rewritten edit instruction as plain text — no preamble, no \
quotes, no markdown, no explanation."""

# ---------------------------------------------------------------------------
# Per-view geometry clauses.
#
# The same scene edit must be phrased differently for a top-down wrist camera
# than for a side-mounted exterior camera, or the editor renders an object's top
# surface into a side view (and vice versa).
# ---------------------------------------------------------------------------

VIEW_GEOMETRY = {
    WRIST: (
        "This image is a robot's first-person, top-down (bird's-eye) wrist-camera "
        "view, looking straight down at the tabletop from directly above. Render "
        "any added or modified object from this same overhead viewpoint: show its "
        "top surface as seen from directly above, not its side profile."
    ),
    "exterior_right": (
        "This image is a fixed exterior camera mounted to the RIGHT of a robot "
        "tabletop workspace, viewing the table from the side at an angle, roughly "
        "at table height. Render any added or modified object in correct side-on "
        "perspective for this viewpoint: show its front and side faces as a person "
        "standing at this camera would see them, resting on the table surface with "
        "a correct contact shadow — NOT as a flat top-down cutout."
    ),
    "exterior_left": (
        "This image is a fixed exterior camera mounted to the LEFT of a robot "
        "tabletop workspace, viewing the table from the side at an angle, roughly "
        "at table height. Render any added or modified object in correct side-on "
        "perspective for this viewpoint: show its front and side faces as a person "
        "standing at this camera would see them, resting on the table surface with "
        "a correct contact shadow — NOT as a flat top-down cutout."
    ),
}

# Appended to every per-view prompt: the invariants no edit may break.
PRESERVE_CLAUSE = (
    "Do not move or rotate the camera, and do not change the framing, zoom, "
    "perspective, or scale of the scene. Do not change the robot arm or gripper. "
    "Keep every object fully visible and non-overlapping, and keep everything "
    "not mentioned above exactly the same."
)

# Sensible default shared `keep` clause when a spec doesn't provide one.
DEFAULT_KEEP = (
    "Keep the robot arm and gripper, the camera viewpoint, framing, perspective, "
    "scale, and all object positions exactly the same. Do not move, remove, add, "
    "warp, or recolor any object that the edit does not explicitly target."
)


def view_geometry_clause(view: str) -> str:
    """Return the camera-geometry preamble for ``view``."""
    try:
        return VIEW_GEOMETRY[view]
    except KeyError:
        raise ValueError(
            f"no geometry clause for view '{view}' (known: {sorted(VIEW_GEOMETRY)})"
        ) from None


def _describe_other(view: str) -> str:
    return {
        WRIST: "the top-down wrist camera",
        "exterior_right": "the right-side exterior camera",
        "exterior_left": "the left-side exterior camera",
    }.get(view, f"the {view} camera")


def chain_prompt(view: str, reference_views: Sequence[str]) -> str:
    """Clause telling the editor to match views that were already edited.

    ``reference_views`` are the views edited earlier in this case (the anchor
    first). The corresponding *images* are passed alongside the prompt by the
    mode, in the same order.
    """
    if not reference_views:
        return ""
    names = [_describe_other(v) for v in reference_views]
    others = names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" and {names[-1]}"
    return (
        f"The additional image(s) provided show the SAME scene, already edited, from "
        f"{others}. Reproduce that same edited scene here: the same objects, with the "
        f"same colors, materials, and counts, in the same positions on the table and "
        f"the same spatial relationships to each other. The ONLY difference must be "
        f"the camera viewpoint — this image keeps its own viewpoint as described "
        f"above. Do not add, remove, or relocate anything relative to those "
        f"reference image(s)."
    )


def view_prompt(
    edit_prompt: str,
    view: str,
    *,
    keep: Optional[str] = None,
    reference_views: Sequence[str] = (),
) -> str:
    """Assemble the full prompt used to edit a single ``view``.

    Order: camera geometry -> what to change -> match-the-anchor clause (when
    this view follows others) -> what to preserve.
    """
    parts = [view_geometry_clause(view), edit_prompt.strip()]
    chain = chain_prompt(view, reference_views)
    if chain:
        parts.append(chain)
    if keep:
        parts.append(keep.strip())
    parts.append(PRESERVE_CLAUSE)
    return " ".join(p for p in parts if p)


def template_edit_prompt(instruction: str) -> str:
    """Deterministic fallback wrapper — no LLM, fully reproducible.

    Viewpoint-neutral on purpose: :func:`view_prompt` adds the per-camera
    geometry, so this layer must not bake in one camera's framing.
    """
    instruction = instruction.strip().rstrip(".")
    return (
        f"Edit the image to {instruction}. "
        "Match the scene's existing lighting, shadow direction, and scale so the "
        "result looks physically consistent with the rest of the image. Place "
        "objects so that each one is fully visible and none overlaps or hides "
        "another."
    )


def _gemini_edit_prompt(instruction: str, model: str, api_key_env: str) -> str:
    """Rewrite via text Gemini. Raises on any failure (caller decides fallback)."""
    from google import genai  # lazy: only needed for the gemini backend
    from google.genai import types

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not set")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=instruction.strip(),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
        ),
    )
    text = (getattr(response, "text", None) or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty rewrite")
    return text


def build_edit_prompt(
    instruction: str,
    *,
    backend: str = "gemini",
    model: str = DEFAULT_GEMINI_MODEL,
    api_key_env: str = "GOOGLE_API_KEY",
    verbose: bool = True,
) -> str:
    """Turn a plain instruction into a viewpoint-neutral edit prompt.

    The result still needs :func:`view_prompt` to specialize it for a camera.

    Args:
        instruction: the user's plain-language scene/edit instruction.
        backend: ``"gemini"`` (LLM rewrite) or ``"template"`` (deterministic).
        model: Gemini text model id used by the ``"gemini"`` backend.
        api_key_env: env var holding the Gemini API key.
        verbose: print which backend produced the prompt.

    Returns:
        The rewritten edit instruction (always a non-empty string). The
        ``"gemini"`` backend silently falls back to the template on any error.
    """
    instruction = (instruction or "").strip()
    if not instruction:
        raise ValueError("instruction must be a non-empty string")

    if backend == "template":
        prompt = template_edit_prompt(instruction)
        if verbose:
            print(f"[guardrail] template -> {prompt!r}")
        return prompt

    if backend == "gemini":
        try:
            prompt = _gemini_edit_prompt(instruction, model, api_key_env)
            if verbose:
                print(f"[guardrail] gemini({model}) -> {prompt!r}")
            return prompt
        except Exception as exc:  # noqa: BLE001 - degrade gracefully to template
            prompt = template_edit_prompt(instruction)
            if verbose:
                print(
                    f"[guardrail] gemini backend failed ({exc}); "
                    f"falling back to template -> {prompt!r}"
                )
            return prompt

    raise ValueError(
        f"unknown guardrail backend '{backend}' (expected 'gemini' or 'template')"
    )
