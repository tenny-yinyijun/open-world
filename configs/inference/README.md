# Inference configs

Inference-only `ARWMArgs` configs for loading a trained AR world-model student and
rolling it out — teleoperation (`scripts/interactive_ar.py`), open-loop replay
(`openworld/autoregressive/infer/replay.py`), and eval. They are **not** for
training (no mid-training / distillation recipe is intended to be run from here).

Each config derives from `configs/training/ar_wan_droid.py` and pins the knobs that
change model/data geometry at inference time.

## Published checkpoints don't have a config here

A **published** model's config ships with its checkpoint on the Hub, not in this
directory — pass the model name and it is fetched for you:

```bash
python scripts/interactive_ar.py --config wm_student_2view --checkpoint wm_student_2view
```

See [docs/MODELS.md](../../docs/MODELS.md) for the available models and the reasoning.
The configs below are for checkpoints that don't resolve by name (yet).

## 3-view bimanual student config

`ar_wan_student_3view_bimanual.py` is the 3-view bimanual (TRI) camera_cond student.
It is an **undistilled student** — sample with the many-step preview schedule (do
**not** pass `--distilled`).

The weights **are** published (`wm_student_3view_bimanual/` in the same Hub repo), but
the config shipped alongside them is still the older constants-style file, not a
`get_args() -> ARWMArgs` factory — so the model *name* does not resolve yet and you need
this local config plus an explicit checkpoint path:

```bash
hf download tennyyyin/open-world-ar-wm --include 'wm_student_3view_bimanual/*' \
    --local-dir checkpoints/ar_wm
```

| Config | views | `action_dim` | geom. cond | state-pred head | block geometry |
| --- | --- | --- | --- | --- | --- |
| `ar_wan_student_3view_bimanual.py` | 3 (`view_indices=(1,2,3)`) | 20 (cartesian, bimanual) | camera_cond (9-ch → 25 in) | 16 | fpb 1, hist 4, roll 12 |

> Note: this config inherits `stage="self_forcing"` from the training base, so
> `scripts/replay_ar.py` will select the **4-step distilled** sampler and render a
> blurry colour-wash. Pass `--force-many-step` (with `--denoising-steps 32`) until the
> config sets `stage="student_init"`.

This is a **camera_cond** model
(`camera_cond=True`, `camera_cond_channels=9`): the patch-embed widens 16 → 25 input
channels, so the geometry sidecar must be present at inference. Replay it with an
explicit conditioning source:

```bash
python scripts/replay_ar.py \
    --config configs/inference/ar_wan_student_3view_bimanual.py \
    --checkpoint checkpoints/ar_wm/wm_student_3view_bimanual/wm_student_3view_bimanual.pt \
    --latent-root <bimanual_latents> --split val --conditioning episode \
    --force-many-step --denoising-steps 32     # see the stage note above
```

## Legacy DROID configs

These pin only `num_cams` / `action_space` (fpb-2 geometry from `ar_wan_droid.py`),
for older DROID checkpoints:

| Config | `num_cams` | `action_space` | action dim / stats |
| --- | --- | --- | --- |
| `ar_wan_droid_2view_cartesian.py` | 2 | `cartesian` | 7 / `stats.json` |
| `ar_wan_droid_2view_jointpos.py`  | 2 | `joint_pos` | 8 / `stats_joint.json` |
| `ar_wan_droid_3view_cartesian.py` | 3 | `cartesian` | 7 / `stats.json` |
| `ar_wan_droid_3view_jointpos.py`  | 3 | `joint_pos` | 8 / `stats_joint.json` |

The trained weights come from the `--checkpoint` flag, not from the config. Pick the
config whose geometry matches how the checkpoint was trained.

Notes:
- `num_cams` / `view_indices` only change how many (and which) height-stacked views the
  data path feeds, not the model's parameters. `view_indices` pins an exact stored-view
  subset (needed for the bimanual checkpoint's scene+2-wrist layout); plain `num_cams`
  keeps the wrist + samples the sides.
- `action_space` / `action_dim` *must* match the checkpoint's training conditioning:
  7-dim cartesian, 8-dim joint, and 20-dim bimanual are not interchangeable.
- `state_pred` / `state_pred_dim` must match too — a checkpoint trained with the
  auxiliary state-prediction head carries `backbone.state_head.*` weights, so the model
  has to be built with the head (of the right dim) or the load fails. The head is not
  used by the forward-only rollout.
- `camera_cond` / `camera_cond_channels` must match — a camera_cond checkpoint's
  `patch_embedding.weight` has `16 + camera_cond_channels` input channels (25 for the
  9-channel bimanual student), so the config has to widen the patch-embed to the same
  width or the conv weight has the wrong shape. Camera_cond also requires the geometry
  input at inference: a `{split}_camera_cond.npy` sidecar (`--conditioning episode`) or
  a `{split}_joint_actions.npy` chunk FK-synthesized closed-loop (`--conditioning action`).
- The inherited training-only fields (learning rate, distillation schedule,
  `student_init_ckpt` / `teacher_ckpt` paths) are unused by a forward-only rollout.
