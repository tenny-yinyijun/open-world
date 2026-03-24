# Policy Training

Follow instructions for policy training in the specific policy architecture that is being used. Our model currently suports any openpi / diffusion model (dppo) checkpoint with joint velocity control and joint state input.

## Working with openpi

```bash
git clone https://github.com/tenny-yinyijun/openpi external/openpi
```

For testing, you can directly use the provided pi05_droid checkpoint at `gs://openpi-assets/checkpoints/pi05_droid`

## Working with diffusion policies

```bash
git clone --recurse-submodules https://github.com/tenny-yinyijun/dsrl external/dsrl
```

We provide an example below for data pre-processing and training for an example demonstration dataset collected under the [droid](https://github.com/droid-dataset/droid) setup:
```bash
# preprocessing
uv run python scripts/process_droid_for_dppo.py \
  --input_dir /path/to/dataset \
  --output_dir data/dppo_processed/<dataset_name> \
  --camera_types wrist ext1 \
  --img_resolution 192 192

# launch training
uv run python scripts/train_dppo.py \
  --config configs/training/<your_training_config>.yaml
```