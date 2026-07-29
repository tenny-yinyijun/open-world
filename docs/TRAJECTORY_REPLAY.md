# Trajectory Replay

Feed a recorded episode's action sequence through a trained world model and compare
the predicted video against ground truth (GT | PRED side-by-side + per-episode
latent/pixel MSE & PSNR). Models: see [MODELS.md](MODELS.md).

## Autoregressive (Wan / Cosmos)

Prime the model with the first ground-truth block(s), feed the full recorded action
sequence open-loop, and let the student generate the rest.

```bash
# published 2-view student: the model NAME resolves weights + config from the Hub
sbatch bash_scripts/ar_gpu.slurm .venv/bin/python scripts/replay_ar.py \
    --config wm_student_2view --checkpoint wm_student_2view \
    --latent-root data/droid_ar_latents --split val \
    --history-blocks 4
```

- For a **published** model, pass its name for both `--config` and `--checkpoint`
  (see [MODELS.md](MODELS.md)); the config that ships with the checkpoint pins the
  view count, action dims, block geometry, and state-pred head the weights need.
  Output defaults to `replay_out/<tag>/`.
- For a checkpoint that doesn't resolve by name, pick the matching `configs/inference/*`
  config (see [configs/inference/README.md](../configs/inference/README.md)) and give a
  real `--checkpoint` path. The bimanual student is published but not yet name-resolvable:
  use `ar_wan_student_3view_bimanual.py` with its own 3-view / 20-dim latents.
- Sampling schedule is selected by the config's `stage`. Published undistilled
  students set `stage="student_init"`, which gets the many-step (32) preview
  schedule. A config left on `stage="self_forcing"` gets the **4-step distilled**
  list instead and renders a blurry colour-wash — pass `--force-many-step
  --denoising-steps 32` to override.
- `--history-blocks` = ground-truth blocks used to prime the cache; match the
  config's `num_history_blocks` (4 for the published students). Without `--checkpoint`
  the untrained backbone runs (validates plumbing; video is noise).
- Core: `openworld/autoregressive/infer/replay.py` (latent-space, decode-free,
  CPU-testable); VAE decode in `data/decode.py`.

**Interactive (live keyboard control):** `scripts/interactive_ar.py`
(`infer/interactive.py:InteractiveRoller`) serves a browser/MJPEG stream you drive
with the keyboard (Wan only). See its module docstring for controls and tunneling.

## SVD bidirectional (`CrtlWorld`)

Closed-loop replay (5-frame chunks) on LIBERO data:

```bash
uv run scripts/replay_libero_wm_traj.py \
    --checkpoint checkpoints/wm/libero/checkpoint-30000.pt \
    --data_root  data/libero_collected \
    --output_dir outputs/libero/replay
```
