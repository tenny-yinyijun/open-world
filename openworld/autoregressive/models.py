"""Registry of **published** AR world-model checkpoints, resolved from the Hub.

Each published model is a folder on Hugging Face holding everything needed to run
it -- the weights, the *inference config* that reproduces the geometry they were
trained with, and the action-normalization stats::

    tennyyyin/open-world-ar-wm
    └── wm_student_2view/
        ├── wm_student_2view.pt      # ARWorldModel state_dict
        ├── inference_config.py      # get_args() -> ARWMArgs  (canonical)
        └── stats.json               # train-set action percentiles

The config lives *with the checkpoint* rather than in this repo on purpose: a
checkpoint's geometry (view count, action dim, extra input channels, aux heads) is
a property of the weights, not of the codebase, and there are far too many knob
combinations to carry one committed config per trained model. So the Hub folder is
the single source of truth and nothing here duplicates it.

Only models listed in :data:`PUBLISHED_MODELS` resolve by name. The Hub repo may hold
further checkpoints whose sidecar config predates the ``get_args()`` convention (e.g.
``wm_student_3view_bimanual``); those need an explicit ``configs/inference/*`` config and
a downloaded checkpoint path until their config is migrated.

The upshot is that a model **name** is accepted anywhere this repo takes a config,
checkpoint, or stats path::

    python scripts/replay_ar.py --config wm_student_2view --checkpoint wm_student_2view ...

Resolution downloads into the ordinary HF cache (``HF_HOME``), so repeat runs are
free and nothing lands in the working tree. Nothing here touches the network at
import time.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PublishedModel:
    """One published checkpoint folder on the Hub."""

    name: str
    repo_id: str
    checkpoint: str                  # weights filename, inside the folder
    config: str = "inference_config.py"
    stats: str = "stats.json"
    # Short human-facing summary, kept next to the wiring so `list_models()` can
    # print an accurate table without a doc round-trip.
    dataset: str = ""
    views: str = ""
    action: str = ""
    summary: str = ""

    @property
    def folder(self) -> str:
        """Folder inside ``repo_id``. Same as the model name, by convention."""
        return self.name

    def _path(self, filename: str) -> str:
        return f"{self.folder}/{filename}"


PUBLISHED_MODELS: dict[str, PublishedModel] = {
    "wm_student_2view": PublishedModel(
        name="wm_student_2view",
        repo_id="tennyyyin/open-world-ar-wm",
        checkpoint="wm_student_2view.pt",
        dataset="DROID",
        views="2 (1 sampled exterior + wrist, height-stacked)",
        action="cartesian absolute EEF pose, 7-d",
        summary="Undistilled 32-step student. Supports trajectory replay, policy "
                "evaluation, and SpaceMouse teleoperation.",
    ),
}


def is_published_model(spec: object) -> bool:
    """True if ``spec`` names a published model (rather than a filesystem path)."""
    return isinstance(spec, str) and spec in PUBLISHED_MODELS


def get_model(name: str) -> PublishedModel:
    if name not in PUBLISHED_MODELS:
        raise KeyError(
            f"Unknown published model {name!r}. Available: "
            f"{sorted(PUBLISHED_MODELS)}"
        )
    return PUBLISHED_MODELS[name]


def _download(model: PublishedModel, filename: str) -> str:
    import os

    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(repo_id=model.repo_id, filename=model._path(filename))
    except Exception as exc:  # network / offline / auth -- all opaque by default
        # The sbatch launchers export HF_HUB_OFFLINE=1 (compute nodes have no
        # internet), so a first-time resolve inside a job fails here with a
        # message that never mentions the cause. Say it plainly.
        offline = os.environ.get("HF_HUB_OFFLINE") not in (None, "", "0")
        hint = (
            "HF_HUB_OFFLINE is set, so nothing can be fetched. Pre-download it "
            "from a node with internet (the file is then reused from the cache):\n"
            f"    hf download {model.repo_id} {model._path(filename)}\n"
            "and make sure HF_HOME points at that same cache in the job."
            if offline else
            "Check network access and that the repo/file exists."
        )
        raise RuntimeError(
            f"Could not resolve {model._path(filename)!r} for published model "
            f"{model.name!r} from {model.repo_id}. {hint}"
        ) from exc


def config_path(name: str) -> str:
    """Local path to the model's ``inference_config.py`` (downloading if needed)."""
    model = get_model(name)
    return _download(model, model.config)


def checkpoint_path(name: str) -> str:
    """Local path to the model's weights (downloading if needed -- several GB)."""
    model = get_model(name)
    return _download(model, model.checkpoint)


def stats_root(name: str) -> str:
    """Local *directory* holding the model's ``stats.json``.

    ``load_action_stats`` takes a root and joins ``cfg.stats_file``, so this
    returns the containing directory rather than the file itself.
    """
    import os

    model = get_model(name)
    return os.path.dirname(_download(model, model.stats))


# -- convenience: resolve-if-a-name, else pass through ----------------------
# Call sites take user-supplied strings that may be either a published model name
# or an ordinary path; these keep that branch to one line at each site.

def resolve_config(spec: str) -> str:
    return config_path(spec) if is_published_model(spec) else spec


def resolve_checkpoint(spec: str) -> str:
    return checkpoint_path(spec) if is_published_model(spec) else spec


def resolve_stats_root(spec: str) -> str:
    return stats_root(spec) if is_published_model(spec) else spec


def list_models() -> str:
    """A printable table of the published models (no network access)."""
    rows = [("name", "dataset", "views", "action")]
    rows += [(m.name, m.dataset, m.views, m.action) for m in PUBLISHED_MODELS.values()]
    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)
