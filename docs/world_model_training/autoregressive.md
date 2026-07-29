# Autoregressive (Wan / Cosmos) World Model Training

The design, the training stages, and how to run them. Most world-model training now
happens in the TRI copy of open-world; this repo is primarily an
[inference](../MODELS.md) platform, but the recipes below still work.

## Why autoregressive

The bidirectional SVD model (`CrtlWorld`) denoises each future chunk from fresh
Gaussian noise with only a handful of sparsely-sampled history frames and **no
persistent latent memory**, so objects drift and disappear over long rollouts. The
recipe here — a DiT initialised from a strong bidirectional video prior, made
block-causal with a KV-cache, and distilled on its own rollouts — fixes that with
three levers:

1. **Block-causal attention + KV-cache** — a real latent memory carried across time
   (the cache *is* the state), instead of re-deriving the scene each chunk.
2. **Self-forcing / DMD distillation** — train the student on the imperfect history it
   actually produces, closing the train/inference gap that drives error accumulation.
3. **Initialise from a bidirectional video prior** (Wan2.1-1.3B / Cosmos-Predict2-2B) —
   strong object semantics for free. "Autoregressive" and "init from a bidirectional
   model" are not in tension: re-mask attention, then distill.

Conditioning stays minimal: per-frame **action** embeddings (+ optional text), with
history frames as clean latents in the cache. For manipulation there is no
ground-truth object state to condition on — the hard thing to predict (object dynamics
under contact) has to be *generated*.

### The causal core

`openworld/autoregressive/tests/test_ar.py` asserts the property the whole design rests
on: the **KV-cache rollout reproduces the block-causal masked forward exactly** —
`max|full − cached_rollout| = 0.0` on the DummyDiT and on the real diffusers Wan and
Cosmos transformers (RoPE-offset included), for both unbounded and sliding-window
(`max_kv_blocks`) memory. The autoregressive memory is mathematically the same
computation as the trained-with mask, just streamed.

A video latent is `F` frames × `tokens_per_frame`; frames are grouped into blocks of
`frames_per_block`; a query attends to its own block plus all earlier blocks (causal),
and bidirectionally *within* a block (including across camera views at the same
timestep). During few-step denoising, intermediate steps attend to the clean cache plus
the current noisy block **without committing** — only the finalized clean block is
appended (`commit=True`).

### Backbones

| key | model | status |
|---|---|---|
| `wan_1_3b` | Wan2.1-T2V-1.3B (`WanTransformer3DModel`) | **recommended.** `forward_train` (block-causal mask) + `forward_cached` (KV-cache, RoPE-offset) both validated against real weights. |
| `cosmos_predict2_2b` | Cosmos-Predict2-2B (`CosmosTransformer3DModel`) | `forward_train` + `forward_cached` validated (RoPE-offset via `cosmos_predict2.py:_offset_rope_cosmos`). |
| `svd` | legacy SVD UNet | intentionally not implemented — UNet + temporal-conv is the wrong substrate for block-causal + cache; bidirectional `CrtlWorld` remains the baseline. |
| `dummy` | tiny CPU DiT | tests only. |

`random_init_backbone=True` builds untrained small models for CI.

## Stages

| Stage | What | Entrypoint |
|---|---|---|
| **1 — student-init** (L2a) | block-causal mid-training; inits the generator | `train_midtrain` (`stage_is_causal=True`) |
| **2 — teacher** (L1b) | bidirectional mid-training; inits the teacher + critic | `train_midtrain` (`stage_is_causal=False`) |
| **3 — self-forcing / DMD** (L0) | few-step distillation on own rollouts (loads 1 + 2) | `train_self_forcing` |

Stages 1 and 2 are independent (both start from the base backbone), so they run as
parallel jobs; stage 3 loads both via `student_init_ckpt` / `teacher_ckpt`. `cfg.stage`
selects the attention pattern (`ARWMArgs.stage_is_causal`) — and, at inference,
[the sampling schedule](../TRAJECTORY_REPLAY.md).

`distill/` implements a faithful reference of the Self-Forcing / DMD2 loop: the
generator (causal student) rolls out few-step with the cache; a **critic** ("fake
score") learns to denoise the student's samples; a frozen **bidirectional teacher**
("real score", CFG'd) anchors the data distribution; the generator's DMD loss is the
score difference in clean-latent space. Gradient is retained only through each block's
final denoising step (Self-Forcing) to keep long rollouts tractable.

> Action conditioning ablations:
[ACTION_COND_EXPERIMENTS.md](../ACTION_COND_EXPERIMENTS.md).

## How to run

**Setup**
```bash
uv sync --extra autoregressive
python -m openworld.autoregressive.train_self_forcing --smoke   # weightless sanity check
.venv/bin/python -m pytest openworld/autoregressive/tests -q    # CPU unit tests
```

**Weights + data**
```bash
bash external/download_models.sh                       # login node (needs internet)
python scripts/preprocess_ar_latents.py --help         # RGB -> 16-ch backbone-VAE latents
python scripts/validate_data.py --help                 # latents -> one real backbone forward
```

**Train** (stages 1 + 2 in parallel, then 3)
```bash
# stage 1 (student-init, causal) and stage 2 (teacher, bidirectional)
torchrun --nproc_per_node=8 -m openworld.autoregressive.train_midtrain \
    --config configs/training/ar_wan_studentinit_droid_aligned.py
torchrun --nproc_per_node=8 -m openworld.autoregressive.train_midtrain \
    --config configs/training/ar_wan_teacher_droid_aligned.py

# stage 3 (self-forcing / DMD); loads both of the above
sbatch bash_scripts/training/train_dmd_aligned.sh
```

Launchers live in [`bash_scripts/`](../../bash_scripts/README.md); ad-hoc GPU commands
go through `sbatch bash_scripts/ar_gpu.slurm <command...>`.

> **Offline-cluster loading gotcha.** Compute nodes typically have no internet, and
> diffusers' sharded-checkpoint loader pings the Hub for a bare repo id *even with*
> `HF_HUB_OFFLINE=1`. Load weights from a **local directory** instead
> (`backbone_ckpt="external/Wan2.1-T2V-1.3B-Diffusers"`, as the Wan configs do).
> `bash_scripts/_env.sh` exports `HF_HUB_OFFLINE=1` / `HF_HOME` for you.

## Dtype

Two knobs: **`cfg.dtype`** is the parameter/optimizer dtype, **`cfg.mixed_precision`**
the compute dtype. The real configs use **fp32 master weights + bf16 autocast**
(`dtype=float32`, `mixed_precision="bf16"`) — the standard mixed-precision recipe, and
what the Self-Forcing / CausVid references use:

* **fp32 params + fp32 AdamW state.** At `lr=6e-6` on O(1) weights a per-step update
  ≈1e-6 is below bf16's ~3-digit precision and would be rounded away — the optimizer
  stalls on the smallest-but-real updates over a 200k-step run.
* **bf16 compute.** The backbone `_call` coerces inputs to the param dtype, then runs
  the transformer matmuls/convs under `torch.autocast("cuda", bf16)` when
  `cfg.autocast_dtype` is set (it is `None` when params already carry the compute
  dtype, so the same path serves a pure-fp32 or pure-bf16 smoke).
* **fp32 loss math.** `make_cfg_score_fn` casts the backbone's bf16 score output to
  fp32, so the precision-sensitive DMD difference `x0_fake - x0_real` and the CFG
  combination — differencing two large, similar tensors — run in fp32.
* Backward and the optimizer step run **outside** autocast (autocast wraps only the
  transformer forward inside `_call`).

To switch: `dtype=float32, mixed_precision="no"` → pure fp32; `dtype=bfloat16` →
uniform bf16 (fast smoke, not for long runs).

## Validated on H200 (real weights)

`scripts/smoke_wan_real.py` on one H200, loading the actual Wan2.1-1.3B (1.42B params,
30 self-attn layers):

* fp32 `forward_train` vs KV-cached rollout: **max err 4.8e-06** — the block-causal
  cache is exact at real scale, not just on the DummyDiT.
* Wan VAE (`AutoencoderKLWan`) loads, **16 latent channels** (the re-encode target).
* **~34 ms / block** (2 latent frames, single forward, bf16, 320² 1 view) → a 2-step
  distilled rollout ≈ ~85 ms/block ≈ **~90 fps effective single-view**.
* **Full data → training-step chain validated** (`scripts/validate_train_step.py`):
  DROID annotations → format adapter → 16-ch latents (height-stacked cams) →
  `ARLatentDataset` → the full self-forcing / DMD loop on three real Wan-1.3B models
  produces finite, decreasing losses.
