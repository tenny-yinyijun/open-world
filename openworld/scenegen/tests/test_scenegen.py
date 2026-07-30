"""scenegen tests — view sets, prompt assembly, and the eval-format contract.

Runs entirely offline: the ``copy`` mode needs no API key, so the suite-building
and eval-compatibility tests are real end-to-end checks, not mocks.

    python3 -m pytest openworld/scenegen/tests/test_scenegen.py     # with pytest
    python3 openworld/scenegen/tests/test_scenegen.py               # standalone
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from openworld.datasets import InitializationDataset
from openworld.scenegen import build_suite, verify_suite
from openworld.scenegen.guardrail import chain_prompt, view_prompt
from openworld.scenegen.views import (
    SIDE_FIRST,
    VIEWS_2,
    VIEWS_3,
    WRIST_FIRST,
    ViewSet,
    canonical_order,
    resolve_base,
)

EDITS = [
    {"label": "a", "instruction": "put the mug in the white container",
     "prompt": "Change the tabletop to matte green."},
    {"label": "b", "instruction": "put the mug in the white container",
     "prompt": "Replace the background with a forest.", "edit_order": SIDE_FIRST},
]


# ---------------------------------------------------------------- view sets
def test_canonical_order_is_sides_then_wrist():
    assert canonical_order(["wrist", "exterior_right"]) == ("exterior_right", "wrist")
    assert canonical_order(["wrist", "exterior_left", "exterior_right"]) == (
        "exterior_right", "exterior_left", "wrist",
    )


def test_canonical_order_rejects_bad_views():
    for bad in (["nope"], ["wrist", "wrist"], []):
        try:
            canonical_order(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")


def test_two_view_set_shape():
    assert VIEWS_2.num_cams == 2
    assert VIEWS_2.views == ("exterior_right", "wrist")
    assert VIEWS_2.sides == ("exterior_right",)
    assert VIEWS_2.has_wrist


def test_edit_sequence_both_orders():
    # 2-view: both anchor orders are valid and are exact reverses here.
    assert VIEWS_2.edit_sequence(WRIST_FIRST) == ("wrist", "exterior_right")
    assert VIEWS_2.edit_sequence(SIDE_FIRST) == ("exterior_right", "wrist")
    # 3-view: wrist moves between front and back, sides keep canonical order.
    assert VIEWS_3.edit_sequence(WRIST_FIRST) == (
        "wrist", "exterior_right", "exterior_left",
    )
    assert VIEWS_3.edit_sequence(SIDE_FIRST) == (
        "exterior_right", "exterior_left", "wrist",
    )


def test_edit_sequence_covers_every_view():
    for vs in (VIEWS_2, VIEWS_3):
        for order in (WRIST_FIRST, SIDE_FIRST):
            assert sorted(vs.edit_sequence(order)) == sorted(vs.views)


def test_wrist_only_and_side_only_sets_work():
    assert ViewSet(("wrist",)).edit_sequence(SIDE_FIRST) == ("wrist",)
    assert ViewSet(("exterior_left",)).edit_sequence(WRIST_FIRST) == ("exterior_left",)


# ------------------------------------------------------------------ prompts
def test_view_prompt_uses_per_camera_geometry():
    wrist = view_prompt("Add an apple.", "wrist")
    side = view_prompt("Add an apple.", "exterior_right")
    assert "top-down" in wrist and "directly above" in wrist
    # The side view must NOT be told to render a top-down surface.
    assert "from the side at an angle" in side
    assert "not its side profile" not in side
    assert "RIGHT" in side
    assert "LEFT" in view_prompt("Add an apple.", "exterior_left")


def test_anchor_prompt_has_no_chain_clause():
    anchor = view_prompt("Add an apple.", "wrist", reference_views=())
    assert "additional image" not in anchor


def test_follower_prompt_references_the_anchor():
    follower = view_prompt("Add an apple.", "exterior_right", reference_views=("wrist",))
    assert "wrist camera" in follower
    assert "same edited scene" in follower


def test_chain_prompt_lists_multiple_references():
    text = chain_prompt("exterior_left", ("wrist", "exterior_right"))
    assert "wrist camera" in text and "right-side exterior camera" in text
    assert " and " in text
    assert chain_prompt("wrist", ()) == ""


def test_keep_clause_is_included():
    p = view_prompt("Add an apple.", "wrist", keep="Keep the blue mug unchanged.")
    assert "Keep the blue mug unchanged." in p


# ------------------------------------------------- suite / eval-format contract
def _build(tmp: Path, views, order=WRIST_FIRST, base_case=False):
    return build_suite(
        base="tri",
        views=views,
        edits=EDITS,
        out_dir=str(tmp),
        mode="copy",
        edit_order=order,
        include_base_case=base_case,
        verify=True,
        verbose=False,
    )


def test_two_view_suite_is_loadable_by_eval(tmp_path):
    cases = _build(tmp_path / "two", ["exterior_right", "wrist"])
    assert len(cases) == len(EDITS)
    for case in cases:
        assert (case / "wrist.png").exists()
        assert (case / "exterior_right.png").exists()
        # A 2-view case must NOT carry the third view.
        assert not (case / "exterior_left.png").exists()

    ds = InitializationDataset.from_yaml(str(tmp_path / "two"))
    assert len(ds) == len(EDITS)
    for init in ds:
        views = init.initial_observation["views"]
        # Exactly the suite's views, in world-model view_order.
        assert list(views) == ["exterior_right", "wrist"]
        assert init.initial_state["robot"]["state_representation"]
        assert init.instruction
        for path in views.values():
            assert Path(path).is_absolute() and Path(path).exists()


def test_three_view_suite_still_works(tmp_path):
    _build(tmp_path / "three", None)
    ds = InitializationDataset.from_yaml(str(tmp_path / "three"))
    for init in ds:
        assert list(init.initial_observation["views"]) == [
            "exterior_right", "exterior_left", "wrist",
        ]


def test_views_are_written_at_world_model_resolution(tmp_path):
    from PIL import Image

    cases = _build(tmp_path / "res", ["exterior_right", "wrist"])
    for case in cases:
        for png in case.glob("*.png"):
            assert Image.open(png).size == (320, 192)


def test_world_model_bootstrap_indexing(tmp_path):
    """The 2-view suite must support ar_world_model's `views[v]` per view_order."""
    _build(tmp_path / "boot", ["exterior_right", "wrist"])
    ds = InitializationDataset.from_yaml(str(tmp_path / "boot"))
    for init in ds:
        views = init.initial_observation["views"]
        frames = [views[v] for v in ("exterior_right", "wrist")]  # config view_order
        assert len(frames) == 2


def test_base_control_case_is_prepended(tmp_path):
    cases = _build(tmp_path / "ctl", ["exterior_right", "wrist"], base_case=True)
    assert len(cases) == len(EDITS) + 1
    assert cases[0].name == "init_0"
    import yaml

    meta = yaml.safe_load((cases[0] / "initialization.yaml").read_text())["metadata"]
    assert meta["edit_mode"] == "base_copy"


def test_start_index_appends(tmp_path):
    out = tmp_path / "append"
    _build(out, ["exterior_right", "wrist"])
    build_suite(base="tri", views=["exterior_right", "wrist"], edits=EDITS[:1],
                out_dir=str(out), mode="copy", start_index=len(EDITS),
                verify=False, verbose=False)
    ids = verify_suite(out)
    assert ids == ["init_0", "init_1", "init_2"]


def test_per_edit_edit_order_recorded(tmp_path):
    import yaml

    cases = _build(tmp_path / "order", ["exterior_right", "wrist"])
    # EDITS[1] overrides to side_first; the manifest records the resolved chain.
    manifest = yaml.safe_load((tmp_path / "order" / "scenegen_manifest.json").read_text())
    assert manifest["num_cams"] == 2
    assert manifest["views"] == ["exterior_right", "wrist"]
    assert manifest["mode"] == "copy"


def test_verify_suite_rejects_missing_view_file(tmp_path):
    out = tmp_path / "broken"
    _build(out, ["exterior_right", "wrist"])
    (out / "init_0" / "wrist.png").unlink()
    try:
        verify_suite(out)
    except AssertionError:
        return
    raise AssertionError("verify_suite should reject a missing view file")


def test_multiview_mode_rejects_side_first():
    """multiview synthesizes sides *from* the wrist view, so side_first is invalid."""
    from openworld.scenegen import get_mode
    from openworld.scenegen.modes.base import Edit as E
    from openworld.scenegen.views import SIDE_FIRST as SF

    mode = get_mode("multiview")
    base = resolve_base("tri", views=["exterior_right", "wrist"])
    try:
        mode.generate_case(
            base=base,
            edit=E(prompt="p", instruction="i"),
            case_dir=Path("/tmp/never-written"),
            edit_order=SF,
        )
    except ValueError as exc:
        assert "wrist_first" in str(exc)
        return
    raise AssertionError("multiview should reject side_first")


def test_multiview_mode_requires_a_wrist_view():
    from openworld.scenegen import get_mode
    from openworld.scenegen.views import ViewSet as VS

    mode = get_mode("multiview")
    base = resolve_base("tri", views=["exterior_left"])
    try:
        mode.preflight(base=base, view_set=VS(("exterior_left",)))
    except (ValueError, RuntimeError, FileNotFoundError):
        return
    raise AssertionError("multiview should require a wrist view")


def test_resolve_base_rejects_unavailable_view():
    try:
        resolve_base("tri", views=["wrist", "nope"])
    except (ValueError, FileNotFoundError):
        return
    raise AssertionError("expected an error for an unknown view")


def test_unknown_mode_and_edit_order_are_rejected(tmp_path):
    for kwargs in ({"mode": "nope"}, {"edit_order": "sideways"}):
        try:
            build_suite(base="tri", edits=EDITS, out_dir=str(tmp_path / "x"),
                        verbose=False, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")


# ------------------------------------------------------------------ standalone
def _main() -> int:
    import inspect
    import tempfile
    import traceback

    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        with tempfile.TemporaryDirectory() as td:
            try:
                if "tmp_path" in inspect.signature(fn).parameters:
                    fn(Path(td))
                else:
                    fn()
                print(f"  PASS  {name}")
            except Exception:
                failed += 1
                print(f"  FAIL  {name}")
                traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
