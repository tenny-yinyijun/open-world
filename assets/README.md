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
It holds the three world-model views plus a `template.yaml` with the robot start state
and a default `scene` tag.

```
assets/<base>/
├── wrist.png            # 320×192 top-down wrist camera
├── exterior_left.png    # 320×192
├── exterior_right.png   # 320×192
└── template.yaml        # initial_state (robot pose) + default scene tag
```

| Base | Views from | Used by |
|------|------------|---------|
| `tri`  | `data/benchmark/0617_generated/_base_original/` | the `0617_generated` suite |
| `irom` | `open-world/data/benchmark/irom_carrot_pnp/init_2/` | irom-princeton DROID setup |

## Use

Reference a base by name (`tri` / `irom`) in a suite spec and build with
nanobanana all-views edits — see
[`configs/scenegen/suites/example.yaml`](../configs/scenegen/suites/example.yaml)
and the `build_suite.py` module docstring:

```bash
GOOGLE_API_KEY=... python scripts/scenegen/build_suite.py \
    --spec configs/scenegen/suites/example.yaml
# -> data/benchmark/<name>/init_*/
```

## Adding a base

Drop the three `*.png` views into a new `assets/<name>/`, add a `template.yaml`
with an `initial_state` block (copy one from an existing
`initialization.yaml`) and a `scene:` tag, then reference `<name>` from a spec.
