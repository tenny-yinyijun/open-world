---
license: apache-2.0
tags:
  - robotics
  - world-model
  - video-prediction
  - autoregressive
library_name: pytorch
---

# open-world AR world-model checkpoints

Autoregressive (block-causal) video world models for robot manipulation, built on the
Wan2.1-T2V-1.3B DiT backbone. Use them with the
[`open-world`](https://github.com/irom-princeton/open-world) repo for open-loop
trajectory replay, policy evaluation, and interactive SpaceMouse teleoperation.

Each model is a **self-contained folder**: the weights, the inference config that
reproduces the geometry the checkpoint was trained with, and its action-normalization
stats.

```
wm_student_2view/
├── wm_student_2view.pt     # ARWorldModel state_dict
├── inference_config.py     # get_args() -> ARWMArgs  (canonical config)
└── stats.json              # train-set action percentiles (7-d)
```

## Available models

Both are **undistilled** students: sample with the many-step (32) preview schedule, and
do **not** pass `--distilled`.

| model | robot / data | views | action input | geom. cond | resolvable by name |
|---|---|---|---|---|---|
| `wm_student_2view` | DROID (Franka, single-arm) | 2 — 1 exterior + wrist | cartesian absolute EEF pose, 7-d | — | ✅ |
| `wm_student_3view_bimanual` | TRI bimanual platform | 3 — scene + 2 wrists | cartesian absolute, 20-d (per arm xyz + rot6d + gripper) | camera_cond (9 ch → 25 in) | ⚠️ not yet |

### `wm_student_2view`

A DROID world model at 192×320, 2 height-stacked views. Actions are absolute 7-d
cartesian EEF poses (xyz + Euler-XYZ + gripper) injected per-frame
(`cross_attn_aligned`), with fully-causal single-frame blocks and 4 frames of history.
Plain 16-channel latent input — no geometric conditioning, so no geometry sidecar is
needed at inference.

Supports trajectory replay, policy evaluation, and teleoperation.

**This is an undistilled student: it needs the 32-step sampler.** Its
`inference_config.py` sets `stage="student_init"`, which is what selects that schedule
— do not pass `--distilled` (that few-step deployment schedule is for a distilled
checkpoint and produces a blurry colour-wash here). Few-step (2/4-step) distilled
releases are planned; they will be separate entries in the table above.

### `wm_student_3view_bimanual`

The TRI bimanual platform at 192×320, 3 height-stacked views (scene + 2 wrists), 20-d
cartesian absolute actions, and a 16-d aux state head. This is a **camera_cond** model:
`patch_embedding.weight` is `(1536, 25, 1, 2, 2)` — 16 latent + 9 geometry channels (3
trajectory band + 6 camera ray-map) — so it is **not** loadable with a 16-channel config,
and geometry inputs must be supplied at inference.

Two caveats today:

- Its `inference_config.py` is still the older **constants-style** file (`NUM_CAMS = 3`,
  …) rather than a `get_args() -> ARWMArgs` factory, so passing the model *name* to the
  repo does not work yet — only `wm_student_2view` resolves by name. Use an explicit
  `--config configs/inference/ar_wan_student_3view_bimanual.py` with a downloaded
  checkpoint path.
- **Trajectory replay only.** Policy evaluation and teleoperation go through a rollout
  path that has no camera_cond support (it allocates 16-channel latents and passes no
  geometry), and expects a 7-d DROID state rather than this model's 20-d bimanual vector.

## Usage

The repo resolves a **model name** to this folder and downloads what it needs, so you
do not have to manage paths or keep a config in sync:

```bash
# open-loop trajectory replay
python scripts/replay_ar.py \
    --config wm_student_2view --checkpoint wm_student_2view \
    --latent-root <droid_latents> --split val

# policy evaluation (pi0.5 closed-loop inside the world model)
uv run python scripts/run_evaluation.py --config configs/evaluation/teleop_ar_pi05.yaml

# interactive teleoperation
python scripts/interactive_ar.py --config wm_student_2view --checkpoint wm_student_2view
```

To fetch a whole model folder directly instead:

```bash
hf download tennyyyin/open-world-ar-wm --include 'wm_student_2view/*' \
    --local-dir checkpoints/ar_wm
# the bimanual model must be fetched this way (it does not resolve by name yet):
hf download tennyyyin/open-world-ar-wm --include 'wm_student_3view_bimanual/*' \
    --local-dir checkpoints/ar_wm
```

See the repo docs: [`docs/MODELS.md`](https://github.com/irom-princeton/open-world/blob/main/docs/MODELS.md),
`docs/EVAL.md`, `docs/TRAJECTORY_REPLAY.md`, `docs/TELEOPERATION.md`.
