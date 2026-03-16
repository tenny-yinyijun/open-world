# Installation

## Base

This project is managed by uv. First install all the dependencies with
```bash
uv sync
```
And include other optional dependencies for supporting different policies and reward functions
```bash
uv sync --extra policy-dp
uv sync --extra policy-openpi
uv sync --extra reward-robometer
uv sync --extra reward-topreward
```

## World-model assets

```bash
sudo apt-get install git-lfs -y
bash external/download_models.sh
```

## VidWM config fields

```yaml
world_model:
  name: vidwm
  checkpoint_path: /path/to/vidwm_checkpoint.pt
  params:
    svd_model_path: external/stable-video-diffusion-img2vid
    clip_model_path: external/clip-vit-base-patch32
```
