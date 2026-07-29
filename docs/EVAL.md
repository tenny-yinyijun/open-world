# Running Policy Evaluation

Run a policy closed-loop inside a world model on initializations.

## Quick Start

```bash
# Run pi0.5 with AR 2-view model on teleop inits
uv run python scripts/run_evaluation.py \
    --config configs/evaluation/teleop_ar_pi05.yaml
```

This runs the `wm_student_2view` world model with the pi0.5 policy on initializations in
`assets/teleop_inits/`. The world model (weights + its inference config + action stats)
is downloaded from Hugging Face on first run — see [MODELS.md](MODELS.md).

Output videos: `outputs/teleop_ar_pi05/videos/`

You also need a local pi0.5 policy checkpoint; the config points at the default openpi
download location (`~/.cache/openpi/openpi-assets/checkpoints/pi05_droid`).

> The published world model is an **undistilled 32-step** student, so evaluation is
> slow. Few-step distilled models are planned — see
> [MODELS.md](MODELS.md#roadmap-few-step-distilled-models).

## Running on Different Initializations

```bash
# Run on a different initialization directory
uv run python scripts/run_evaluation.py \
    --config configs/evaluation/teleop_ar_pi05.yaml \
    --dataset path/to/your/init_dir
```

## Creating a Custom Eval Config

If you need different settings, create a new YAML config:

```yaml
# configs/evaluation/my_eval.yaml
world_model:
  name: ar_wan
  # A published model name resolves weights + config + stats from the Hub.
  # For a local checkpoint, give real paths to all three instead.
  checkpoint_path: wm_student_2view
  params:
    config_path: wm_student_2view
    stats_root: wm_student_2view
    vae_dir: external/Wan2.1-T2V-1.3B-Diffusers
    num_inference_steps: 32
    num_cams: 2
    width: 320
    height: 192
    view_order: [exterior_right, wrist]

policy:
  name: openpi
  checkpoint_path: ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid   # ~ is expanded
  params:
    config_name: pi05_droid
    repo_path: external/openpi
    pytorch_device: cuda
    exterior_view_name: exterior_right
    wrist_view_name: wrist
    stacked_view_order: [exterior_right, wrist]
    resize_height: 224
    resize_width: 224
    joint_position_dim: 7
    action_adapter_checkpoint_path: checkpoints/action_adapter/model2_15_9.pth
    action_adapter_gripper_max: 0.9

reward_model:
  name: dummy
  params: {}

scheduler:
  chunk_size: 8

duration: 25
action_hz: 5
dataset_path: path/to/init_dir
video_dir: outputs/my_eval
```

Then run:
```bash
uv run python scripts/run_evaluation.py --config configs/evaluation/my_eval.yaml
```

## Initialization Directory Structure

```
init_dir/
├── init_0/
│   ├── exterior_left.png
│   ├── exterior_right.png
│   ├── wrist.png
│   └── initialization.yaml
├── init_1/
│   └── ...
└── stats.json  # Optional: action normalization stats
```

## Available Configs

See `configs/evaluation/` for pre-configured examples:
- `teleop_ar_pi05.yaml` - AR `wm_student_2view` + pi0.5 + teleop inits (**start here**)
- `0617_ar_pi05.yaml` - AR 3-view + pi0.5 (needs a local 3-view checkpoint)
- `0617_ctrlworld_pi05.yaml` - Ctrl-World + pi0.5

Only `teleop_ar_pi05.yaml` runs against a published checkpoint; the others point at
local paths from earlier experiments.

---

## Reference: World Model Config Examples

<details>
<summary>AR Wan 2-view (wm_student_2view)</summary>

```yaml
world_model:
  name: ar_wan
  checkpoint_path: wm_student_2view
  params:
    config_path: wm_student_2view
    stats_root: wm_student_2view
    num_cams: 2
    view_order: [exterior_right, wrist]
    num_inference_steps: 32
```
</details>

<details>
<summary>AR Wan 3-view bimanual — not supported for policy eval yet</summary>

The 3-view bimanual camera_cond student is published (`wm_student_3view_bimanual/` on
the Hub) but **cannot** be run through policy evaluation today. Three things are missing,
not just a config:

- `InteractiveRoller` (the adapter's rollout path) has no `camera_cond` support: it
  allocates latent blocks at 16 channels and passes `pixel_cond=None`, but this
  checkpoint's patch-embed takes 25. Only `scripts/replay_ar.py` renders the geometry.
- `ARWanWorldModel` reads a 7-d DROID cartesian state; this model needs the 20-d
  bimanual vector.
- The bundled `assets/teleop_inits/` are 2-view / 7-d.

Use `scripts/replay_ar.py` for this checkpoint — it is replay-only for now (see
[configs/inference/README.md](../configs/inference/README.md)).
</details>

<details>
<summary>Ctrl-World (SVD)</summary>

```yaml
world_model:
  name: ctrlworld
  checkpoint_path: checkpoints/wm/ctrlworld/v0-checkpoint-120000.pt
  params:
    svd_model_path: external/stable-video-diffusion-img2vid
    clip_model_path: external/clip-vit-base-patch32
    num_frames: 5
    num_history: 6
    action_dim: 7
    num_inference_steps: 50
```
</details>
