# 🌐 open-world

OpenWorld is an open-source platform for building, fine-tuning, and evaluating robotic policies using video world models. We currently support:

* Fast world model inference with shortcut models
* Policy evaluation on custom benchmarks with automatic scoring

**🔧 Some of the functions are still under construction!**

## Installation

Requirements:

- Python 3.11+
- uv for environment management

```bash
# Dependencies for base environment only:
uv sync

# Include extra dependencies for using different policy/reward model. Example:
uv sync --extra policy-dp --extra reward-robometer
uv sync --extra policy-openpi --extra reward-robometer
```

Finally, install required assets for the base world model:

```bash
sudo apt-get install git-lfs -y
bash external/download_models.sh
```


## Workflows

- 🤖 [Policy Training](docs/TRAIN_POLICY.md)
- 📋 [Policy Evaluation](docs/EVAL.md)
- ⚙️ [Policy Fine-tuning](docs/FT.md) (TODO)
- 🌎 [World Model Training](docs/TRAIN_WM.md) (TODO)

## Next Steps

- [ ] Add more reward function support
- [ ] Example benchmarks
- [ ] Data & checkpoints

## Acknowledgements

This repo is based on [Ctrl-World](https://github.com/Robert-gyj/Ctrl-World), [dppo](https://github.com/irom-princeton/dppo), [dsrl](https://github.com/ajwagen/dsrl), [openpi](https://github.com/Physical-Intelligence/openpi), and [robometer](https://github.com/robometer/robometer). 

Core contributors: