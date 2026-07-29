"""Inference config for the ``wm_student_2view`` AR world-model student (DROID).

This file ships **alongside the checkpoint** and is its canonical config: it pins the
exact geometry ``wm_student_2view.pt`` was trained with, so the weights load with no
missing or unexpected keys. Nothing in the open-world repo duplicates it -- point the
tools at the model name and they fetch this file from the Hub:

    python scripts/replay_ar.py     --config wm_student_2view --checkpoint wm_student_2view ...
    python scripts/interactive_ar.py --config wm_student_2view --checkpoint wm_student_2view

Model summary
-------------
* **Data**: DROID, 192x320, 2 height-stacked views (1 sampled exterior + wrist).
* **Action conditioning**: 7-d absolute cartesian EEF pose (xyz + Euler-XYZ +
  gripper), normalized with the ``stats.json`` percentiles in this same folder,
  injected per-frame (``cross_attn_aligned``: latent frame f attends to action
  token f -- the tightest action->frame binding).
* **Rollout geometry**: fully-causal single-frame blocks (``frames_per_block=1``),
  4 history blocks primed into the KV-cache, 12 blocks per training rollout.
* **Latent input**: plain 16 channels. No geometric conditioning
  (``camera_cond``/``pixel_cond`` off), so ``patch_embedding.weight`` is
  (1536, 16, 1, 2, 2) and no geometry sidecar is needed at inference.
* **Aux head**: an 8-d joint state-prediction head (``backbone.state_head.*``) is
  present in the weights. It is unused by the forward-only rollout, but the model
  must be *built* with it (``state_pred=True, state_pred_dim=8``) or the load is
  not clean -- hence it is set here.

Sampling: this is an **undistilled** student, so it needs the many-step schedule
(32 steps). ``stage="student_init"`` is what tells the tooling that -- on a
``self_forcing`` stage, ``scripts/replay_ar.py`` would select the 4-step *distilled*
``denoising_step_list`` and render a blurry colour-wash. Do not pass ``--distilled``
to ``interactive_ar.py`` for the same reason. A few-step (2/4-step) distilled
release is a separate, future checkpoint.

Only inference-relevant fields are set. Training-only knobs (learning rates, the
DMD/distillation schedule, ``student_init_ckpt`` / ``teacher_ckpt``, dataset roots)
are left at their ``ARWMArgs`` defaults and are not read by a forward-only rollout.
"""

from __future__ import annotations

import os

import torch

from openworld.autoregressive.config import ARWMArgs


def get_args() -> ARWMArgs:
    return ARWMArgs(
        # -- backbone ------------------------------------------------------
        backbone="wan_1_3b",
        # Local Wan2.1-1.3B (from external/download_models.sh) if present, else the
        # Hub repo -- so this config works in a fresh clone with no local assets.
        backbone_ckpt=os.environ.get("BACKBONE_CKPT") or (
            "external/Wan2.1-T2V-1.3B-Diffusers"
            if os.path.isdir("external/Wan2.1-T2V-1.3B-Diffusers")
            else "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        ),
        vae_dir=os.environ.get("VAE_DIR") or (
            "external/Wan2.1-T2V-1.3B-Diffusers"
            if os.path.isdir("external/Wan2.1-T2V-1.3B-Diffusers")
            else "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        ),
        # Undistilled student -> many-step preview sampler (see module docstring).
        # This is NOT a cosmetic label: it selects the sampling schedule.
        stage="student_init",
        # -- views / frame geometry ---------------------------------------
        num_cams=2,
        wrist_view_idx=2,
        multiview_layout="height_stack",
        height=192,
        width=320,
        data_format="droid_ctrl_world",
        # -- autoregressive block geometry --------------------------------
        frames_per_block=1,
        num_history_blocks=4,
        rollout_blocks=12,
        # -- action conditioning ------------------------------------------
        action_space="cartesian",          # -> stats.json, action_dim=7
        action_dim=7,
        action_cond_mode="cross_attn_aligned",
        text_cond=True,
        frame_level_cond=True,
        # -- aux state-prediction head (present in the weights) -----------
        state_pred=True,
        state_pred_dim=8,
        # -- no geometric conditioning (plain 16-channel latent input) ----
        camera_cond=False,
        pixel_cond=False,
        # -- sampler ------------------------------------------------------
        num_train_timestep=1000,
        preview_denoising_steps=32,
        # -- dtype: fp32 master weights + bf16 autocast compute ----------
        dtype=torch.float32,
        mixed_precision="bf16",
        tag="wm_student_2view",
    )
