"""Remove a named object from each view of an init to make an empty-table base.

Object-removal nanobanana edit (camera/everything-else held fixed), applied to
each view of a source init dir, producing an "empty base" that
``make_suite_add_object.sh`` then populates with new objects. This is how the
``_base_no_mug`` base for the 0617 suite was made (object="green mug").

    GOOGLE_API_KEY=... python scripts/scenegen/remove_object.py \\
        --object "green mug" --src-dir <suite>/_base_original --dst-dir <suite>/_base_no_mug
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openworld.scenegen.guardrail import view_prompt  # noqa: E402
from openworld.scenegen.modes.nanobanana import nanobanana_edit  # noqa: E402
from openworld.scenegen.views import ALL_VIEWS  # noqa: E402

# The per-camera geometry clause is added by `view_prompt`, so this stays
# viewpoint-neutral. Removal is chained across views (each later view sees the
# already-cleared ones) so the inpainted table surface agrees between cameras.
PROMPT_TMPL = (
    "Remove ONLY the {obj} from the table and inpaint that spot with the "
    "surrounding table surface so it looks like empty table. This is an "
    "object-removal edit only: do NOT move, warp, or re-render anything else. "
    "Keep everything else identical: the robot arm and gripper, any shelf/stand, "
    "the floor, the background, and the lighting. Do not add or recolor any object."
)


def main() -> None:
    p = argparse.ArgumentParser(description="Remove an object from each view of an init.")
    p.add_argument("--object", required=True, help='e.g. "green mug"')
    p.add_argument("--src-dir", required=True, help="dir with the per-view PNGs")
    p.add_argument("--dst-dir", required=True, help="output dir for the edited views")
    p.add_argument("--views", nargs="+", default=list(ALL_VIEWS), choices=list(ALL_VIEWS),
                   metavar="VIEW",
                   help="Views to clear; pass `exterior_right wrist` for a 2-view base.")
    p.add_argument("--edit-order", choices=["wrist_first", "side_first"], default="wrist_first",
                   help="Which view is cleared first and anchors the others.")
    args = p.parse_args()

    from openworld.scenegen.views import ViewSet

    view_set = ViewSet(tuple(args.views))
    base_prompt = PROMPT_TMPL.format(obj=args.object)
    done, images = [], {}
    for v in view_set.edit_sequence(args.edit_order):
        src = Path(args.src_dir) / f"{v}.png"
        if not src.exists():
            print(f"  WARN: missing {src}, skipping"); continue
        prompt = view_prompt(base_prompt, v, reference_views=tuple(done))
        images[v] = nanobanana_edit(
            str(src),
            str(Path(args.dst_dir) / f"{v}.png"),
            prompt,
            references=[images[d] for d in done],
        )
        done.append(v)
    print(f"remove_object done ({len(done)} view(s) cleared: {done}).")
    print(
        "NOTE: a base also needs a template.yaml with the robot start state — copy "
        f"one into {args.dst_dir} (see assets/README.md)."
    )


if __name__ == "__main__":
    main()
