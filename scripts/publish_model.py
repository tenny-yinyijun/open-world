"""Publish a model's *metadata* (inference config, stats, model card) to the Hub.

Each published checkpoint lives in its own folder in a Hub repo, holding the weights
plus everything needed to run them -- see :mod:`openworld.autoregressive.models`. This
script uploads the small files in that folder from a local staging dir; it does **not**
upload the multi-GB weights (those are published once, by hand).

    # inspect what would change, no writes
    python scripts/publish_model.py wm_student_2view --dry-run

    # verify the staged config against the checkpoint's contract, then upload
    python scripts/publish_model.py wm_student_2view

    # also refresh the repo-root model card (artifacts/hf/_root/README.md)
    python scripts/publish_model.py wm_student_2view --with-model-card

Uploads are refused unless the staged config actually loads and satisfies the
published-model tests, so a config that would render a colour-wash (wrong ``stage``)
or fail to load (wrong geometry) can't reach the Hub.

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

STAGING_ROOT = _REPO_ROOT / "artifacts" / "hf"
# Repo-root files (the model card) live here rather than in a model folder, since
# they describe the whole repo instead of one checkpoint.
ROOT_STAGING = STAGING_ROOT / "_root"

# Small files only. The weights are excluded on purpose: they are immutable once
# published, and re-uploading several GB by accident is expensive.
METADATA_SUFFIXES = (".py", ".json", ".md", ".txt", ".yaml")


def _read_token(token_file: str | None) -> str | None:
    if token_file:
        return Path(token_file).expanduser().read_text().strip()
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _verify(name: str) -> None:
    """Run the published-model contract tests against the staged config."""
    print(f"[publish] verifying staged config for {name} ...")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest",
         "openworld/autoregressive/tests/test_published_models.py",
         "-q", "-m", "not hub", "-p", "no:cacheprovider"],
        cwd=_REPO_ROOT,
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
    p.add_argument("--staging-dir", default=None,
                   help=f"Local folder holding the files. Default: {STAGING_ROOT}/<model>")
    p.add_argument("--dry-run", action="store_true",
                   help="List what would be uploaded and exit without writing.")
    p.add_argument("--token-file", default=None,
                   help="File containing an HF write token (else $HF_TOKEN / hf auth login).")
    p.add_argument("--skip-verify", action="store_true",
                   help="Skip the contract tests. Not recommended.")
    p.add_argument("--with-model-card", action="store_true",
                   help=f"Also upload the repo-root README.md from {ROOT_STAGING}.")
    a = p.parse_args()

    model = get_model(a.model)
    staging = Path(a.staging_dir) if a.staging_dir else STAGING_ROOT / a.model
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
    if a.with_model_card:
        card = ROOT_STAGING / "README.md"
        if not card.is_file():
            raise SystemExit(f"[publish] model card not found: {card}")
        uploads.append((card, "README.md"))

    print(f"[publish] repo   : {model.repo_id}")
    print(f"[publish] folder : {model.folder}/")
    print(f"[publish] staging: {staging}")
    for f, dest in uploads:
        print(f"           {f.name:24} -> {dest}  ({f.stat().st_size} B)")
    if not a.with_model_card:
        print("[publish] (root README.md not included; pass --with-model-card to update it)")

    if a.dry_run:
        print("[publish] --dry-run: nothing uploaded.")
        return

    if not a.skip_verify:
        _verify(a.model)

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
