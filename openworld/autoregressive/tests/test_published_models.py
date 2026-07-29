"""Guards on the published-model registry and the configs that ship with checkpoints.

Published checkpoints carry their own ``inference_config.py`` on the Hub rather than
having a copy committed here (see :mod:`openworld.autoregressive.models`). That keeps
one source of truth, but it moves a correctness risk off-repo: a published config that
declares the wrong geometry, or the wrong sampling ``stage``, only fails at rollout
time -- and for ``stage`` it fails *silently*, as a blurry colour-wash rather than an
exception.

So the offline tests here pin the invariants a published config must satisfy, and the
networked test (``-m hub``) checks the real published artifact still satisfies them.
"""
import os

import pytest

from openworld.autoregressive.models import (
    PUBLISHED_MODELS,
    get_model,
    is_published_model,
    list_models,
)

# The geometry ``wm_student_2view.pt`` was trained with. These are properties of the
# weights: change one and the checkpoint no longer loads (or loads and generates
# nonsense), so they are asserted rather than derived.
EXPECTED_2VIEW = {
    "backbone": "wan_1_3b",
    "num_cams": 2,
    "multiview_layout": "height_stack",
    "height": 192,
    "width": 320,
    "action_space": "cartesian",
    "action_dim": 7,
    "action_cond_mode": "cross_attn_aligned",
    "frames_per_block": 1,
    "num_history_blocks": 4,
    "rollout_blocks": 12,
    "state_pred": True,
    "state_pred_dim": 8,
    "camera_cond": False,
    "pixel_cond": False,
    # 16 latent channels + 0 extra -> patch_embedding.weight is (1536, 16, 1, 2, 2).
    "model_in_channels": 16,
}


def test_registry_shape():
    """Every registered model points at a folder-per-model layout on the Hub."""
    assert PUBLISHED_MODELS, "registry must not be empty"
    for name, m in PUBLISHED_MODELS.items():
        assert m.name == name, "dict key must match .name (folder is derived from it)"
        assert m.repo_id.count("/") == 1, f"{name}: repo_id should be 'owner/repo'"
        assert m.checkpoint.endswith(".pt")
        assert m.config.endswith(".py"), "config must be an importable .py"
        assert m.folder == name
        # The summary fields feed `list_models()` / the docs table; an empty one
        # silently degrades that table to blanks.
        for field in ("dataset", "views", "action", "summary"):
            assert getattr(m, field), f"{name}: {field} must be described"


def test_is_published_model_discriminates_paths():
    assert is_published_model("wm_student_2view")
    # Real paths must fall through to the filesystem, never the registry.
    assert not is_published_model("configs/inference/ar_wan_droid_2view_cartesian.py")
    assert not is_published_model("checkpoints/ar_wm/some_ckpt.pt")
    assert not is_published_model("")
    assert not is_published_model(None)
    assert not is_published_model(object())


def test_unknown_model_lists_alternatives():
    with pytest.raises(KeyError, match="wm_student_2view"):
        get_model("wm_student_nope")


def test_list_models_renders():
    out = list_models()
    assert "wm_student_2view" in out
    assert "DROID" in out
    # header + separator + one row per model
    assert len(out.splitlines()) == len(PUBLISHED_MODELS) + 2


def _assert_2view_contract(cfg):
    """The invariants any published 2-view config must satisfy."""
    for field, expected in EXPECTED_2VIEW.items():
        assert getattr(cfg, field) == expected, (
            f"{field}: config says {getattr(cfg, field)!r}, "
            f"checkpoint requires {expected!r}"
        )
    # An undistilled student MUST NOT report the self_forcing stage: replay_ar.py
    # keys its sampler off it, and on 'self_forcing' it picks the few-step distilled
    # denoising_step_list, which renders a blurry colour-wash instead of raising.
    assert cfg.stage != "self_forcing", (
        "published undistilled student must not use stage='self_forcing' "
        "(replay_ar.py would select the 4-step distilled sampler)"
    )
    assert cfg.preview_denoising_steps == 32
    # stats.json ships in the same folder; action_space must select it.
    assert cfg.stats_file == "stats.json"


def test_staged_2view_config_matches_checkpoint_contract():
    """A *staged* (not yet uploaded) config satisfies the checkpoint's contract.

    The Hub is the only copy of a published config, so there is nothing in this repo
    to check -- ``scripts/publish_model.py`` instead points ``OPENWORLD_STAGED_CONFIG``
    at the file it is about to upload and runs this test first. That is what catches a
    bad config *before* it is published, which matters because a wrong ``stage`` fails
    silently at rollout time (a blurry colour-wash, not an exception).

    Skips in an ordinary test run, where no upload is pending.
    """
    staged = os.environ.get("OPENWORLD_STAGED_CONFIG")
    if not staged:
        pytest.skip(
            "no OPENWORLD_STAGED_CONFIG (set by scripts/publish_model.py before an "
            "upload); the live config is covered by the -m hub test"
        )
    assert os.path.exists(staged), f"staged config does not exist: {staged}"

    from openworld.autoregressive.train_self_forcing import _load_config

    _assert_2view_contract(_load_config(staged))


@pytest.mark.hub
def test_published_2view_config_matches_checkpoint_contract():
    """The LIVE published config satisfies the checkpoint's contract.

    Network + Hub access required: ``pytest -m hub``. This is the test that would
    catch a stale or hand-edited config on the Hub.
    """
    pytest.importorskip("huggingface_hub")
    from openworld.autoregressive.models import config_path
    from openworld.autoregressive.train_self_forcing import _load_config

    _assert_2view_contract(_load_config(config_path("wm_student_2view")))
