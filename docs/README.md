# OpenWorld docs

This repo is primarily an **inference** platform: run a trained video world model for
policy evaluation, trajectory replay, and teleoperation. World-model *training* mostly
happens in the TRI copy of open-world; what lands here are the trained checkpoints.

Start with **[MODELS.md](MODELS.md)** — what checkpoints are published, what data and
action space each was trained on, and which workflows each supports.

```
  published checkpoint          ┌──────────────────────────────┐
  (weights + config + stats,    │  POLICY EVALUATION           │ rollout
   resolved by model name)  ───►│  policy ⇄ world model (loop) │ videos
            +                   │  → EVAL.md                   │ (+reward)
  Initialization suite      ───►└──────────────────────────────┘
  (per-view PNGs +                        the same world model also drives
   initialization.yaml)                   → TRAJECTORY_REPLAY.md · TELEOPERATION.md
```

A policy eval needs two things: a **world model** (name a published one and its weights,
inference config, and action stats are fetched for you) and an **Initialization suite** —
a directory of per-case `initialization.yaml` + per-view PNGs that an eval config's
`dataset_path` points at. `assets/teleop_inits/` ships a small bundled suite, so a fresh
clone can run without downloading any data.

## Workflows

| Doc | What | Entry points |
|-----|------|--------------|
| **[MODELS.md](MODELS.md)** | Published checkpoints — what each model is and supports | `openworld/autoregressive/models.py` |
| **[EVAL.md](EVAL.md)** | Run a policy closed-loop inside a world model over a suite | `bash_scripts/eval_wm.sbatch` → `scripts/run_evaluation.py` → `scripts/generate_videos.py` · `configs/evaluation/` |
| **[TRAJECTORY_REPLAY.md](TRAJECTORY_REPLAY.md)** | Feed a recorded action sequence open-loop, compare against ground truth | `scripts/replay_ar.py` |
| **[TELEOPERATION.md](TELEOPERATION.md)** | Drive the world model live from a SpaceMouse / keyboard | `scripts/interactive_ar.py` · `bash_scripts/interactive_ar.sh` |

## Quick start

```bash
# 1. (one time) set up the eval stack: submodules + venvs + checkpoint symlinks
bash bash_scripts/setup_eval_env.sh

# 2. run a policy eval — the world model downloads from the Hub on first use
uv run python scripts/run_evaluation.py --config configs/evaluation/teleop_ar_pi05.yaml
```

## Other references

| Doc | Topic |
|-----|-------|
| [TRAIN_POLICY.md](TRAIN_POLICY.md) | Training / loading policy checkpoints |
| [ACTION_COND_EXPERIMENTS.md](ACTION_COND_EXPERIMENTS.md) | Action-conditioning ablations |
| [LIBERO.md](LIBERO.md) | LIBERO benchmark specifics |
| [world_model_training/](world_model_training/) | World-model training (AR self-forcing / DMD, SVD) |

### Authoring your own Initialization suites

`openworld/scenegen/` builds suites from a language instruction + an image: a guardrail
rewrites the instruction into an edit prompt, nanobanana edits the views, and the result
is assembled into an `initialization.yaml` + PNG suite. There is no separate doc — the
CLIs carry their usage in their module docstrings:

- `scripts/scenegen/build_suite.py` — suite from a YAML spec of scene edits (see
  `configs/scenegen/suites/example.yaml`); needs only `GOOGLE_API_KEY`, no GPU
- `scripts/generate_test_case.py` — single case via the multiview add-object path
- base view sets: [`assets/`](../assets/README.md)
