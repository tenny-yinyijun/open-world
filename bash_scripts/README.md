# `bash_scripts/` — cluster launchers

Thin, cluster-aware shell / sbatch wrappers. The Python entrypoints they call live in
`scripts/` and `openworld/`; these files only set SBATCH resources and env.

Submit from the **repo root** so relative paths (`scripts/…`, `configs/…`, `external/…`)
resolve. The SBATCH headers here are written for the Princeton `ailab` partition — edit
`--partition` / `--account` / `--gres` for your cluster.

```
bash_scripts/
  _env.sh                       shared env (sourced): offline HF, cd, slurm_outputs, node info
  ar_gpu.slurm                  generic GPU runner for ad-hoc commands
  setup_eval_env.sh             one-time: submodules + venvs + checkpoint symlinks (login node)

  # inference (the main use of this repo)
  eval_wm.sbatch                policy evaluation: --wm {ctrlworld|ar|weaver}
  eval_weaver_0617.sbatch       policy evaluation, weaver on the 0617 benchmark
  interactive_ar.sh             teleoperation server (interactive node, not sbatch)
  inference/replay_wan.sh       open-loop trajectory replay (Wan)
  inference/replay_cosmos.sh    open-loop trajectory replay (Cosmos)
  replay_tri_ar_job.sh          replay on the TRI bimanual latents

  # training (mostly done in the TRI copy of open-world)
  training/train_dmd_aligned.sh        self-forcing / DMD distillation, 8 GPU
  training/train_dmd_aligned_vN_4gpu.sh  same, 4 GPU
```

## Typical run

```bash
# 0. one-time, on a node WITH internet (compute nodes are offline):
bash bash_scripts/setup_eval_env.sh     # venvs + submodules
bash external/download_models.sh        # backbone / VAE weights -> external/

# 1. policy evaluation (the world model downloads from the Hub on first use):
sbatch bash_scripts/eval_wm.sbatch --wm ar

# 2. open-loop replay of a checkpoint against ground truth:
CKPT=checkpoints/ar_wm/ar_wan_droid/checkpoint-50000.pt \
  sbatch bash_scripts/inference/replay_wan.sh

# 3. teleoperation — needs an interactive GPU node, not sbatch:
bash bash_scripts/interactive_ar.sh

# monitor any job:
squeue -j <jobid>   ·   tail -f slurm_outputs/<name>/<jobid>.out
```

For a published model you can pass its **name** instead of paths — e.g.
`CONFIG=wm_student_2view CKPT=wm_student_2view sbatch bash_scripts/inference/replay_wan.sh`
(see [docs/MODELS.md](../docs/MODELS.md)).

> **These launchers set `HF_HUB_OFFLINE=1`** (`_env.sh`), because compute nodes here have
> no internet — so a model name cannot download *from inside a job*. Fetch it once from a
> login node first; the job then reads it from the cache:
> ```bash
> hf download tennyyyin/open-world-ar-wm --include 'wm_student_2view/*'
> ```
> Keep `HF_HOME` the same in both places (`_env.sh` defaults it to
> `external/.hf_cache`). Running the Python entrypoints directly on an
> internet-connected machine needs none of this.

Ad-hoc GPU commands go through the generic runner:

```bash
sbatch bash_scripts/ar_gpu.slurm .venv/bin/python -m pytest openworld/autoregressive/tests -q
```

## Overriding defaults

Every launcher reads its knobs from environment variables (sensible defaults baked in)
so you rarely edit the files. Set them inline before `sbatch`:

| script | useful env vars / flags (defaults) |
|---|---|
| `eval_wm.sbatch` | flags: `--wm {ctrlworld\|ar\|weaver}` (required), `--config`, `--checkpoint`, `--dataset`, `--video-dir`, `--duration`; env: `PI_DIR`, `PORT` (8123) |
| `inference/replay_wan.sh` · `replay_cosmos.sh` | `CKPT`, `CONFIG`, `LATENT_ROOT` (`data/droid_ar_latents`), `SPLIT` (`val`), `HISTORY_BLOCKS` (1), `NUM_EPISODES`, `EPISODE_ID`, `MAX_BLOCKS`, `OUTPUT_DIR`, `SEPARATE` |
| `interactive_ar.sh` | positional: `[aligned\|adaln] [checkpoint.pt] [port]` (mode `aligned`, latest ckpt, port 8000) |
| `training/train_dmd_aligned.sh` | `GPUS` (8, keep `--gres` in sync), `TORCHRUN`, `MASTER_PORT` |

`PY` (default `.venv/bin/python`) and `HF_HOME` are honored by all of them via `_env.sh`,
which also forces offline HF loading. SBATCH resource directives (`--time`, `--mem`,
`--gres`) are at the top of each file — edit there for longer runs or multi-GPU.

Data preprocessing has no launcher of its own; call the entrypoints under `ar_gpu.slurm`:

```bash
sbatch bash_scripts/ar_gpu.slurm .venv/bin/python scripts/preprocess_ar_latents.py --help
sbatch bash_scripts/ar_gpu.slurm .venv/bin/python scripts/validate_data.py --help
```
