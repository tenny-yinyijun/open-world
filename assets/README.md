# assets/ — bundled initialization views

- **[`teleop_inits/`](teleop_inits)** — a small ready-to-run Initialization suite
  (per-view PNGs + `initialization.yaml` + action-norm `stats.json`). It is the default
  for [teleoperation](../docs/TELEOPERATION.md) and the bundled
  [policy-eval](../docs/EVAL.md) config, so a fresh clone works with no data download.
- **`tri/` · `irom/`** — scene-generation **bases** (below).

No model weights live here — `assets/` is committed input data only. Weights land in the
gitignored `checkpoints/` (via `bash external/download_models.sh`) or in the HF cache
(published world models, resolved by name). That includes the small action-adapter MLP
needed for policy eval: `checkpoints/action_adapter/model2_15_9.pth`, see
[docs/EVAL.md](../docs/EVAL.md#two-things-you-must-fetch-yourself).

## Scene-generation bases

A base is a fresh, unedited initialization that scene-edit suites are built on top of.
It holds one PNG per world-model view plus a `template.yaml` with the robot start state
and a default `scene` tag.

```
assets/<base>/
├── wrist.png            # 320×192 top-down wrist camera
├── exterior_right.png   # 320×192 side camera, mounted to the right
├── exterior_left.png    # 320×192 side camera — 3-view bases only
└── template.yaml        # initial_state (robot pose) + default scene tag
```

| Base | Views from | Used by |
|------|------------|---------|
| `tri`  | `data/benchmark/0617_generated/_base_original/` | the `0617_generated` suite |
| `irom` | `open-world/data/benchmark/irom_carrot_pnp/init_2/` | irom-princeton DROID setup |

Both bundled bases carry all three views. A suite picks the subset its world model
was trained on, so the same base serves both the 3-view models and the 2-view
`wm_student_2view` (`exterior_right` + `wrist`) — see
[docs/SCENEGEN.md](../docs/SCENEGEN.md#view-sets-3-view-and-2-view).

## Use

Reference a base by name (`tri` / `irom`) in a suite spec, pick the views, and build:

```bash
GOOGLE_API_KEY=... python scripts/scenegen/build_suite.py \
    --spec configs/scenegen/suites/example_2view.yaml
# -> data/benchmark/<name>/init_*/
```

[`example.yaml`](../configs/scenegen/suites/example.yaml) is the 3-view spec;
[`example_2view.yaml`](../configs/scenegen/suites/example_2view.yaml) is the 2-view one
and documents every key. Full workflow: [docs/SCENEGEN.md](../docs/SCENEGEN.md).

## Adding a base

Drop the view `*.png` files into a new `assets/<name>/` (at minimum `wrist.png` and
one `exterior_*.png`; add all three if you want the base to serve 3-view models too),
add a `template.yaml` with an `initial_state` block (copy one from an existing
`initialization.yaml`) and a `scene:` tag, then reference `<name>` from a spec.
`scripts/scenegen/remove_object.py` will chain-erase an object across the views to
produce an empty-table base from an existing init.
