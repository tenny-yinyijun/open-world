# Supported Models

This repo is primarily an **inference** platform: run a world model for policy
evaluation, trajectory replay, and teleoperation. World-model *training* mostly happens
in the TRI copy of open-world; what lands here are the trained checkpoints.

Workflows: [policy evaluation](EVAL.md) · [trajectory replay](TRAJECTORY_REPLAY.md)
· [teleoperation](TELEOPERATION.md) · [world-model training](#world-model-training).

## Published checkpoints

Published models live on Hugging Face at
[`tennyyyin/open-world-ar-wm`](https://huggingface.co/tennyyyin/open-world-ar-wm), one
**self-contained folder per model** — the weights, the inference config that reproduces
the geometry the checkpoint was trained with, and its action-normalization stats:

```
wm_student_2view/
├── wm_student_2view.pt     # ARWorldModel state_dict
├── inference_config.py     # get_args() -> ARWMArgs  (canonical config)
└── stats.json              # train-set action percentiles
```

The config ships **with the checkpoint**, not in this repo. A checkpoint's geometry
(view count, action dimensionality, extra input channels, aux heads) is a property of
the weights, and there are far too many knob combinations to carry a committed config
per trained model — so the Hub folder is the single source of truth.

| model | robot / data | views | action input | geom. cond | aux state head | sampling |
|---|---|---|---|---|---|---|
| `wm_student_2view` | DROID (Franka, single-arm) | 2 — 1 exterior + wrist | cartesian absolute EEF pose, 7-d | — | 8-d | **32-step** (undistilled) |
| `wm_student_3view_bimanual` | TRI bimanual platform | 3 — scene + 2 wrists | cartesian absolute, 20-d (per arm xyz + rot6d + gripper) | camera_cond (9 ch → 25 in) | 16-d | **32-step** (undistilled) |

### What each model supports

| model | resolves by name | trajectory replay | policy evaluation | teleoperation |
|---|---|---|---|---|
| `wm_student_2view` | ✅ | ✅ | ✅ | ✅ |
| `wm_student_3view_bimanual` | ⚠️ not yet | ✅ | ❌ | ❌ |

### `wm_student_2view`

DROID at 192×320, 2 height-stacked views (1 exterior + wrist). Actions are absolute 7-d
cartesian EEF poses (xyz + Euler-XYZ + gripper) injected per-frame
(`cross_attn_aligned` — latent frame *f* attends to action token *f*, the tightest
action→frame binding). Fully-causal single-frame blocks (`frames_per_block=1`) with 4
history frames. Plain 16-channel latent input, so no geometry sidecar is needed at
inference. Carries an 8-d joint state-prediction head that the rollout doesn't use but
the model must be built with (its config handles this).

> **This is an undistilled student — it needs the 32-step sampler and is therefore
> slow.** Its `inference_config.py` sets `stage="student_init"`, which is what selects
> that schedule. Do **not** pass `--distilled`: that few-step deployment schedule is
> for a distilled checkpoint and yields a blurry colour-wash here.

### `wm_student_3view_bimanual`

The TRI bimanual platform at 192×320, 3 height-stacked views (scene + 2 wrists,
`view_indices=(1,2,3)`), 20-d cartesian absolute actions, 16-d aux state head. A
**camera_cond** model: the patch-embed is widened 16 → 25 input channels (3 trajectory
band + 6 camera ray-map), so it cannot load under a 16-channel config and needs geometry
inputs at inference.

Two limitations today, both tracked above:

- **It does not resolve by model name.** The config published alongside the weights is
  still the older constants-style file rather than a `get_args() -> ARWMArgs` factory, so
  use the local `configs/inference/ar_wan_student_3view_bimanual.py` and an explicit
  downloaded checkpoint path. See
  [configs/inference/README.md](../configs/inference/README.md).
- **Replay only — no policy eval or teleoperation.** Those run through
  `InteractiveRoller`, which has no camera_cond support (it allocates 16-channel latent
  blocks and passes no geometry), and `ARWanWorldModel` expects a 7-d DROID state rather
  than this model's 20-d bimanual vector. The bundled `assets/teleop_inits/` are also
  2-view / 7-d.

### Roadmap: few-step distilled models

The intended end state is that people run **2-step or 4-step distilled** models, for
both DROID and the TRI bimanual platform. Those are not published yet — today both
published checkpoints are 32-step undistilled students, correct but much slower than a
distilled model would be. Distilled releases will appear as new rows in the table above.

Also planned: migrating the bimanual config to the `get_args()` format so it resolves by
name like the 2-view one, and camera_cond support in the closed-loop rollout path so it
can be used for policy evaluation.

## Usage

Anywhere this repo takes a config, checkpoint, or stats path, you can pass the **model
name** instead — it resolves to the Hub folder and downloads into the standard HF cache
(`HF_HOME`), so repeat runs are free and nothing lands in your working tree:

```bash
# policy evaluation (pi0.5 closed-loop inside the world model)
uv run python scripts/run_evaluation.py --config configs/evaluation/teleop_ar_pi05.yaml

# open-loop trajectory replay
python scripts/replay_ar.py \
    --config wm_student_2view --checkpoint wm_student_2view \
    --latent-root <droid_latents> --split val

# interactive teleoperation
python scripts/interactive_ar.py --config wm_student_2view --checkpoint wm_student_2view
```

In an eval YAML, name the model in all three path fields:

```yaml
world_model:
  name: ar_wan
  checkpoint_path: wm_student_2view
  params:
    config_path: wm_student_2view
    stats_root: wm_student_2view
```

To fetch the files explicitly instead:

```bash
hf download tennyyyin/open-world-ar-wm \
    wm_student_2view/wm_student_2view.pt \
    wm_student_2view/inference_config.py \
    wm_student_2view/stats.json \
    --local-dir checkpoints/ar_wm
```

> **On an offline cluster**, resolution can't reach the Hub from inside a job — the
> sbatch launchers export `HF_HUB_OFFLINE=1` because compute nodes have no internet.
> Warm the cache once from a login node and keep `HF_HOME` the same in both places:
> ```bash
> hf download tennyyyin/open-world-ar-wm --include 'wm_student_2view/*'
> ```
> See [`bash_scripts/README.md`](../bash_scripts/README.md).

The registry lives in
[`openworld/autoregressive/models.py`](../openworld/autoregressive/models.py); publish
metadata updates with `scripts/publish_model.py`.

## Model families

### Autoregressive (AR) — *primary*

Block-causal DiT with a KV-cache memory, initialised from a bidirectional video prior
and distilled on its own rollouts (self-forcing / DMD). Code:
`openworld/autoregressive/` (`model.py:ARWorldModel`).

| config | backbone | action-cond modes | platforms | status |
|---|---|---|---|---|
| `wan_1_3b` | Wan2.1-T2V-1.3B | `cross_attn_aligned`, `adaln` | DROID, bimanual | ✅ training<br>❌ few-step<br>✅ checkpoint (see above) |
| `cosmos_predict2_2b` | Cosmos-Predict2-2B | `cross_attn` only | DROID | ❌ training<br>❌ few-step<br>❌ checkpoint |

### SVD Bidirectional

Stable Video Diffusion UNet base. `CrtlWorld` is the base world model; `vidwm` adapts
it into a flow-map / shortcut consistency model for few-step inference.

| backbone | model | action-cond modes | platforms | status |
|---|---|---|---|---|
| `CrtlWorld` (vendored) | SVD-UNet-1.5B | via action adapter | DROID, LIBERO | ✅ base bidirectional flow-matching SVD WM |
| `vidwm` | SVD-UNet-1.5B (flow-map distilled) | via action adapter | DROID, LIBERO | ✅ few-step consistency model built on top of `CrtlWorld` |

## World Model Training

Mostly done in the TRI copy of open-world. The recipes here still work:

**Autoregressive models**: [world_model_training/autoregressive.md](world_model_training/autoregressive.md)

**SVD models**: [world_model_training/svd.md](world_model_training/svd.md)
