#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Downloading world-model assets into: ${ROOT_DIR}"

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "git-lfs is required but not installed."
  echo "Install it first, for example on Ubuntu:"
  echo "  sudo apt-get install git-lfs -y"
  exit 1
fi

git lfs install

cd "${ROOT_DIR}"

if [ ! -d "clip-vit-base-patch32" ]; then
  git clone https://huggingface.co/openai/clip-vit-base-patch32
fi
git -C clip-vit-base-patch32 lfs pull

if [ ! -d "stable-video-diffusion-img2vid" ]; then
  git clone https://huggingface.co/stabilityai/stable-video-diffusion-img2vid
fi
git -C stable-video-diffusion-img2vid lfs pull

echo "Done."
echo "CLIP path: ${ROOT_DIR}/clip-vit-base-patch32"
echo "SVD path: ${ROOT_DIR}/stable-video-diffusion-img2vid"
