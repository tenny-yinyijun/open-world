# 🌐 open-world

OpenWorld is an open-source platform for building, fine-tuning, and evaluating robotic policies using video world models.

## Installation

Requirements:

- Python 3.11+
- uv for environment management

Base environment only:

```bash
uv sync
```

Dependencies for different policies/reward models:

```bash
uv sync --extra policy-dp
uv sync --extra policy-openpi
uv sync --extra reward-robometer
```

Install assets to support base world model:

```bash
sudo apt-get install git-lfs -y
bash external/download_models.sh
```


## Workflows

- 🤖 [Policy Training](docs/TRAIN_POLICY.md) (TODO)
- 📋 [Policy Evaluation](docs/EVAL.md)
- ⚙️ [Policy Fine-tuning](docs/FT.md) (TODO)
- 🌎 [World Model Training](docs/TRAIN_WM.md) (TODO)

## Next Steps

- [ ] Add more reward function support
- [ ] Example benchmarks
- [ ] Task Generator

## Next Steps

<!-- ## Caveats

- The base install is world-model-first. Policy and reward stacks should stay out of `[project.dependencies]` unless they are truly shared across backends.
- Upstream backends have different Python, CUDA, and accelerator expectations. In particular, Robometer's upstream repo currently targets a different environment than the base world-model stack, so OpenWorld keeps its reward integration client-side and optional.
- When adding a new backend, update three places together: the extra in `pyproject.toml`, the lazy registry entry, and the adapter module with a guarded dependency check or clear runtime error.
- Keep adapter imports isolated. Do not import backend-specific packages from package `__init__` files or common interfaces. -->
