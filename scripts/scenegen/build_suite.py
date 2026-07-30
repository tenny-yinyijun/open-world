#!/usr/bin/env python
"""Build an eval-ready Initialization suite from a base + a YAML list of edits.

Pick a *base* (``tri`` or ``irom`` under ``assets/``, or any directory of view
PNGs), list the edits you want, and get one ``init_<i>`` case per edit. The
default ``nanobanana`` mode needs only ``GOOGLE_API_KEY`` — no GPU.

    GOOGLE_API_KEY=... python scripts/scenegen/build_suite.py \
        --spec configs/scenegen/suites/example.yaml

    # 2-view suite (one side camera + wrist), side view anchors the edit:
    GOOGLE_API_KEY=... python scripts/scenegen/build_suite.py \
        --spec configs/scenegen/suites/example_2view.yaml \
        --views exterior_right wrist --edit-order side_first

    # no API calls — check the plumbing and the eval format:
    python scripts/scenegen/build_suite.py \
        --spec configs/scenegen/suites/example_2view.yaml --mode copy

The spec is a YAML file; see ``configs/scenegen/suites/example_2view.yaml`` for a
fully commented one. Anything on the command line overrides the matching key.

Every built suite is loaded back through the real ``InitializationDataset`` before
the command exits (skip with ``--no-verify``), so a suite that ``run_evaluation.py``
could not read fails here instead of at eval time.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openworld.scenegen import available_modes, build_suite_from_spec
from openworld.scenegen.modes import MODES
from openworld.scenegen.views import ALL_VIEWS, EDIT_ORDERS


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build an Initialization suite from a base + a YAML list of edits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--spec", help="YAML suite spec (base + edits list).")
    p.add_argument("--list-modes", action="store_true",
                   help="List the available scene-generation modes and exit.")
    p.add_argument("--base", default=None,
                   help="Override the spec's base (assets/<name> or a path to view PNGs).")
    p.add_argument("--name", default=None,
                   help="Override the suite name (output -> data/benchmark/<name>).")
    p.add_argument("--out-dir", default=None,
                   help="Explicit output directory (overrides --name and the spec).")
    p.add_argument("--views", nargs="+", default=None, choices=list(ALL_VIEWS),
                   metavar="VIEW",
                   help="View subset for the suite, e.g. `--views exterior_right wrist` "
                        "for the 2-view model. Default: every view the base has.")
    p.add_argument("--mode", default=None, choices=list(MODES),
                   help="Scene-generation mode.")
    p.add_argument("--edit-order", default=None, choices=list(EDIT_ORDERS),
                   help="Which view anchors the edit: wrist_first (default, best for "
                        "object edits) or side_first (best for background/lighting).")
    p.add_argument("--start-index", type=int, default=None,
                   help="First case index (use to append to an existing suite).")
    p.add_argument("--include-base-case", action="store_true", default=None,
                   help="Prepend an unedited control case (zero generative drift).")
    p.add_argument("--no-verify", action="store_true",
                   help="Skip loading the suite back through InitializationDataset.")
    args = p.parse_args()

    if args.list_modes:
        print("Available scene-generation modes:")
        for name in available_modes():
            cls = MODES[name]
            two = "2-view ok" if cls.supports_two_view else "3-view only"
            print(f"  {name:<14} [{two}]  {cls.description}")
        return

    if not args.spec:
        p.error("--spec is required (or use --list-modes)")

    overrides = {}
    for flag, key in (
        ("base", "base"),
        ("name", "name"),
        ("out_dir", "out_dir"),
        ("views", "views"),
        ("mode", "mode"),
        ("edit_order", "edit_order"),
        ("start_index", "start_index"),
        ("include_base_case", "include_base_case"),
    ):
        value = getattr(args, flag)
        if value is not None:
            overrides[key] = value
    if args.no_verify:
        overrides["verify"] = False

    build_suite_from_spec(args.spec, **overrides)


if __name__ == "__main__":
    main()
