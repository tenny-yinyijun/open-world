# 🌐 open-world

OpenWorld is an open-source platform for building, fine-tuning, and evaluating robotic policies using video world models. We offer support for various world model families and robot platforms - see [MODELS.md](docs/MODELS.md) for details.

This repo is primarily for **inference** — policy evaluation, trajectory replay, and
teleoperation with a trained world model. A pretrained DROID checkpoint is published on
Hugging Face and downloads on first use; pass its **model name** (`wm_student_2view`)
anywhere a config, checkpoint, or stats path is expected. See
[MODELS.md](docs/MODELS.md) for what is available and
[docs/](docs/README.md) for the workflows.


## Installation

Requirements:

- Python 3.11+
- uv for environment management

```bash
# Dependencies for base environment only:
uv sync

# Include extra dependencies for using different policies/reward models. Example:
uv sync --extra policy-dp --extra reward-robometer
uv sync --extra policy-openpi --extra reward-robometer
```

Finally, install required assets for the base world model:

```bash
sudo apt-get install git-lfs -y
bash external/download_models.sh
```


## Supported Workflows

|  | bidirectional-svd | AR-wan | AR-cosmos |
|---|---|---|---|
| ***🏋️ Training*** | | | |
| [World Model Training](docs/MODELS.md#world-model-training) | ✅ | ✅ | ❌ TODO |
| [Policy Training](docs/TRAIN_POLICY.md) | ✅ | ✅ | ✅ |
| ***✨ Inference*** | | | |
| [Trajectory Replay](docs/TRAJECTORY_REPLAY.md) | ✅ | ✅ | ❌ TODO |
| [Policy Evaluation](docs/EVAL.md) | ✅ | ✅ | ❌ TODO |
| [Teleoperation](docs/TELEOPERATION.md) | ❌ TODO | ✅ | ❌ TODO |
| ***📦 Checkpoints*** | | | |
| [Published checkpoint](docs/MODELS.md#published-checkpoints) | ❌ TODO | ✅ 32-step student | ❌ TODO |
| [Few-step distilled (2/4-step)](docs/MODELS.md#roadmap-few-step-distilled-models) | ✅ (`vidwm`) | ❌ TODO | ❌ TODO |

The published AR checkpoint is an **undistilled 32-step student** — correct but slow.
Few-step distilled releases are [planned](docs/MODELS.md#roadmap-few-step-distilled-models).

## Acknowledgements

This repo is based on [Ctrl-World](https://github.com/Robert-gyj/Ctrl-World), [dppo](https://github.com/irom-princeton/dppo), [dsrl](https://github.com/ajwagen/dsrl), [openpi](https://github.com/Physical-Intelligence/openpi), and [robometer](https://github.com/robometer/robometer). 
