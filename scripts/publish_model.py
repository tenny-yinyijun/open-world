"""Publish a model's *metadata* (inference config, stats, model card) to the Hub.

Each published checkpoint lives in its own folder in a Hub repo, holding the weights
plus everything needed to run them -- see :mod:`openworld.autoregressive.models`. This
script uploads the small files in that folder; it does **not** upload the multi-GB
weights (those are published once, by hand).

The Hub is the only copy of this metadata -- nothing is staged in the repo. So the
edit loop is download, edit, publish::

    # 1. pull the current live metadata into a scratch dir (not the working tree)
    hf download tennyyyin/open-world-ar-wm --include 'wm_student_2view/*' \
        --exclude '*.pt' --local-dir /tmp/hf-staging

    # 2. edit /tmp/hf-staging/wm_student_2view/inference_config.py, then check it
    python scripts/publish_model.py wm_student_2view \
        --staging-dir /tmp/hf-staging/wm_student_2view --dry-run

    # 3. verify against the checkpoint's contract and upload
    python scripts/publish_model.py wm_student_2view \
        --staging-dir /tmp/hf-staging/wm_student_2view

    # the repo-root model card describes the whole repo, so it is passed explicitly
    python scripts/publish_model.py wm_student_2view \
        --staging-dir /tmp/hf-staging/wm_student_2view --model-card /tmp/README.md

Uploads are refused unless the staged config actually loads and satisfies the
published-model tests, so a config that would render a colour-wash (wrong ``stage``)
or fail to load (wrong geometry) can't reach the Hub. The staged file itself is what
gets tested -- not the live one -- which is the point of doing it before the upload.

Auth: ``HF_TOKEN`` env var, ``--token-file``, or a prior ``hf auth login``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from openworld.autoregressive.models import PUBLISHED_MODELS, get_model  # noqa: E402

# Small files only. The weights are excluded on purpose: they are immutable once
# published, and re-uploading several GB by accident is expensive.
METADATA_SUFFIXES = (".py", ".json", ".md", ".txt", ".yaml")

# The contract test reads the config to check from here, so the file that is about
# to be uploaded is the one that gets verified. Keep in sync with
# openworld/autoregressive/tests/test_published_models.py.
STAGED_CONFIG_ENV = "OPENWORLD_STAGED_CONFIG"


def _read_token(token_file: str | None) -> str | None:
    if token_file:
        return Path(token_file).expanduser().read_text().strip()
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _verify(name: str, staged_config: Path | None) -> None:
    """Run the published-model contract tests against the staged config."""
    print(f"[publish] verifying staged config for {name} ...")
    env = dict(os.environ)
    if staged_config is not None:
        env[STAGED_CONFIG_ENV] = str(staged_config)
    else:
        # Nothing to check against -- the offline test would skip and the gate would
        # pass vacuously, which is worse than saying so.
        print(f"[publish] WARNING: no {get_model(name).config} in the staging dir; "
              "the config contract cannot be checked before upload.")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest",
         "openworld/autoregressive/tests/test_published_models.py",
         "-q", "-m", "not hub", "-p", "no:cacheprovider"],
        cwd=_REPO_ROOT,
        env=env,
    )
    if proc.returncode != 0:
        raise SystemExit(
            "[publish] ABORT: staged config failed its contract tests (see above). "
            "Fix the config before publishing."
        )
    print("[publish] staged config OK")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", choices=sorted(PUBLISHED_MODELS),
                   help="Published model name to upload metadata for.")
    p.add_argument("--staging-dir", required=True,
                   help="Local folder holding the files to upload. Nothing is staged "
                        "in the repo -- download the live folder to a scratch dir, "
                        "edit it there, and point this at it.")
    p.add_argument("--dry-run", action="store_true",
                   help="List what would be uploaded and exit without writing.")
    p.add_argument("--token-file", default=None,
                   help="File containing an HF write token (else $HF_TOKEN / hf auth login).")
    p.add_argument("--skip-verify", action="store_true",
                   help="Skip the contract tests. Not recommended.")
    p.add_argument("--model-card", default=None, metavar="PATH",
                   help="Also upload PATH as the repo-root README.md (the model card, "
                        "which describes the whole repo rather than one checkpoint).")
    a = p.parse_args()

    model = get_model(a.model)
    staging = Path(a.staging_dir).expanduser()
    if not staging.is_dir():
        raise SystemExit(f"[publish] staging dir not found: {staging}")

    files = sorted(
        f for f in staging.iterdir()
        if f.is_file() and f.suffix in METADATA_SUFFIXES
    )
    if not files:
        raise SystemExit(f"[publish] no metadata files in {staging}")

    # (local path, path in repo) pairs -- the model folder, plus the root card.
    uploads = [(f, f"{model.folder}/{f.name}") for f in files]
    if a.model_card:
        card = Path(a.model_card).expanduser()
        if not card.is_file():
            raise SystemExit(f"[publish] model card not found: {card}")
        uploads.append((card, "README.md"))

    print(f"[publish] repo   : {model.repo_id}")
    print(f"[publish] folder : {model.folder}/")
    print(f"[publish] staging: {staging}")
    for f, dest in uploads:
        print(f"           {f.name:24} -> {dest}  ({f.stat().st_size} B)")
    if not a.model_card:
        print("[publish] (root README.md not included; pass --model-card PATH to update it)")

    if a.dry_run:
        print("[publish] --dry-run: nothing uploaded.")
        return

    staged_config = staging / model.config
    if not a.skip_verify:
        _verify(a.model, staged_config if staged_config.is_file() else None)

    token = _read_token(a.token_file)
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    for f, dest in uploads:
        print(f"[publish] uploading {dest} ...")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=dest,
            repo_id=model.repo_id,
            commit_message=f"{model.name}: update {dest}",
        )
    print("[publish] done.")
    print("[publish] verify the live artifact with:  pytest "
          "openworld/autoregressive/tests/test_published_models.py -m hub")


if __name__ == "__main__":
    main()
