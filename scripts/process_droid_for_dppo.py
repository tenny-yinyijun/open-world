"""
Process DROID-format demonstration data into the NPZ format expected by training.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm


def _resize_image(args: tuple) -> np.ndarray:
    img, resolution = args
    return cv2.resize(img, (resolution[1], resolution[0]))


def resize_images_parallel(
    images: np.ndarray,
    resolution: tuple[int, int],
    num_workers: int = 10,
) -> np.ndarray:
    args = [(images[i], resolution) for i in range(images.shape[0])]
    with Pool(processes=min(num_workers, cpu_count())) as pool:
        resized = pool.map(_resize_image, args)
    return np.array(resized, dtype=np.uint8)


def load_video_frames(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return np.array(frames, dtype=np.uint8)


def load_hdf5_trajectory(
    file_path: str,
    action_keys: list[str],
    observation_keys: list[str],
) -> dict[str, np.ndarray]:
    """Load state/action data from a DROID trajectory.h5 file."""
    output: dict[str, np.ndarray] = {}
    h5 = h5py.File(file_path, "r")

    for key in action_keys:
        h5_key = f"action/{key}"
        # DROID sometimes stores joint_position as joint_positions
        alt_key = f"action/{key}s" if key == "joint_position" else None
        if h5_key in h5:
            output[h5_key] = h5[h5_key][()]
        elif alt_key and alt_key in h5:
            output[h5_key] = h5[alt_key][()]
        else:
            raise KeyError(f"Action key '{h5_key}' not found in {file_path}")

    for key in observation_keys:
        h5_key = f"observation/robot_state/{key}"
        if h5_key in h5:
            output[h5_key] = h5[h5_key][()]
        else:
            raise KeyError(f"Observation key '{h5_key}' not found in {file_path}")

    # Load skip_action mask if available; otherwise keep all timesteps
    skip_key = "observation/timestamp/skip_action"
    if skip_key in h5:
        output[skip_key] = h5[skip_key][()]
    else:
        # Infer length from the first loaded array
        first = next(iter(output.values()))
        output[skip_key] = np.zeros(first.shape[0], dtype=bool)

    h5.close()
    # Belt-and-suspenders: close any lingering handles
    for obj in gc.get_objects():
        if isinstance(obj, h5py.File):
            try:
                obj.close()
            except Exception:
                pass

    return output


def parse_camera_serials(traj_dir: str) -> dict[str, str] | None:
    """Extract camera serial numbers from metadata JSON."""
    import glob as _glob

    for meta_path in _glob.glob(os.path.join(traj_dir, "metadata*.json")):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            return {
                "wrist": meta.get("wrist_cam_serial"),
                "ext1": meta.get("ext1_cam_serial"),
                "ext2": meta.get("ext2_cam_serial"),
            }
        except Exception:
            continue
    return None


def detect_camera_serials(
    traj_dirs: list[str],
    camera_types: list[str],
) -> list[str]:
    for traj_dir in traj_dirs:
        info = parse_camera_serials(traj_dir)
        if info:
            serials = [info[ct] for ct in camera_types if info.get(ct)]
            if serials:
                print(f"Auto-detected camera serials: {serials}")
                return serials
    raise RuntimeError(
        "Could not auto-detect camera serials from metadata. "
        "Provide --camera_serials explicitly."
    )


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_droid_dataset(
    input_dir: str | list[str],
    output_dir: str,
    action_keys: list[str],
    observation_keys: list[str],
    horizon_steps: int = 16,
    img_resolution: tuple[int, int] = (192, 192),
    camera_serials: list[str] | None = None,
    camera_types: list[str] | None = None,
    num_workers: int = 10,
    skip_image: bool = False,
    max_trajectories: int | None = None,
    obs_gripper_threshold: float = 0.2,
    seed: int = 42,
) -> str:
    """Process DROID data into DPPO-compatible NPZ format.

    Returns the path to the output directory.
    """
    np.random.seed(seed)
    save_image = not skip_image

    input_dirs = [input_dir] if isinstance(input_dir, str) else list(input_dir)

    # Discover trajectory directories
    traj_dirs: list[str] = []
    for d in input_dirs:
        for name in sorted(os.listdir(d)):
            sub = os.path.join(d, name)
            if os.path.isdir(sub) and os.path.exists(os.path.join(sub, "trajectory.h5")):
                traj_dirs.append(sub)
    traj_dirs.sort()
    if max_trajectories:
        traj_dirs = traj_dirs[:max_trajectories]
    print(f"Found {len(traj_dirs)} trajectories")

    # Camera setup
    if save_image:
        if camera_serials is None:
            camera_types = camera_types or ["wrist", "ext1"]
            camera_serials = detect_camera_serials(traj_dirs, camera_types)
        print(f"Using cameras: {camera_serials}")

    # Accumulators
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    traj_lengths: list[int] = []
    all_images: dict[int, list[np.ndarray]] = {
        i: [] for i in range(len(camera_serials or []))
    }
    diff_mins: list[np.ndarray] = []
    diff_maxs: list[np.ndarray] = []

    for traj_dir in tqdm(traj_dirs, desc="Processing trajectories"):
        traj_path = os.path.join(traj_dir, "trajectory.h5")
        try:
            traj = load_hdf5_trajectory(traj_path, action_keys, observation_keys)
        except Exception as e:
            print(f"Skipping {traj_path}: {e}")
            continue

        # Filter idle timesteps
        skip = traj["observation/timestamp/skip_action"]
        keep = ~skip.astype(bool)
        traj_len = int(keep.sum())
        if traj_len < horizon_steps:
            print(f"Skipping short trajectory ({traj_len} < {horizon_steps}): {traj_dir}")
            continue
        traj_lengths.append(traj_len)

        # Binarize gripper
        if "gripper_position" in action_keys:
            traj["action/gripper_position"] = (
                traj["action/gripper_position"] > 0.5
            ).astype(np.float32)
        if "gripper_position" in observation_keys:
            traj["observation/robot_state/gripper_position"] = (
                traj["observation/robot_state/gripper_position"] > obs_gripper_threshold
            ).astype(np.float32)

        # Fix cartesian roll sign
        if "cartesian_position" in action_keys:
            traj["action/cartesian_position"][:, 3] = np.abs(
                traj["action/cartesian_position"][:, 3]
            )
        if "cartesian_position" in observation_keys:
            traj["observation/robot_state/cartesian_position"][:, 3] = np.abs(
                traj["observation/robot_state/cartesian_position"][:, 3]
            )

        # Ensure shapes
        if "joint_positions" in observation_keys:
            jp = traj["observation/robot_state/joint_positions"]
            if jp.ndim == 3:
                jp = jp.squeeze()
            traj["observation/robot_state/joint_positions"] = jp
        if "gripper_position" in action_keys:
            g = traj["action/gripper_position"].squeeze()
            traj["action/gripper_position"] = g[:, None] if g.ndim == 1 else g
        if "gripper_position" in observation_keys:
            g = traj["observation/robot_state/gripper_position"].squeeze()
            traj["observation/robot_state/gripper_position"] = g[:, None] if g.ndim == 1 else g

        # Concatenate state and action vectors
        states = np.concatenate(
            [traj[f"observation/robot_state/{k}"][keep] for k in observation_keys],
            axis=1,
        )
        actions = np.concatenate(
            [traj[f"action/{k}"][keep] for k in action_keys],
            axis=1,
        )
        all_states.append(states)
        all_actions.append(actions)

        # State-action diffs for normalization
        if "joint_positions" in observation_keys and "joint_position" in action_keys:
            s = traj["observation/robot_state/joint_positions"][keep]
            a = traj["action/joint_position"][keep]
        elif "cartesian_position" in observation_keys and "cartesian_position" in action_keys:
            s = traj["observation/robot_state/cartesian_position"][keep]
            a = traj["action/cartesian_position"][keep]
        else:
            s = states[:, : actions.shape[1]]
            a = actions[:, : states.shape[1]]

        diffs = np.empty((0, a.shape[1]))
        for step in range(horizon_steps // 4, horizon_steps):
            if step < len(a):
                diff = a[step:] - s[:-step]
                diffs = np.concatenate([diffs, diff], axis=0)
        if diffs.size:
            diff_mins.append(np.min(diffs, axis=0))
            diff_maxs.append(np.max(diffs, axis=0))

        # Load images
        if save_image and camera_serials:
            mp4_dir = os.path.join(traj_dir, "recordings", "MP4")
            for cam_idx, serial in enumerate(camera_serials):
                video_path = os.path.join(mp4_dir, f"{serial}.mp4")
                if not os.path.exists(video_path):
                    print(f"  Warning: missing video {video_path}")
                    # Append placeholder zeros so lengths stay aligned
                    all_images[cam_idx].append(
                        np.zeros((traj_len, 3, img_resolution[0], img_resolution[1]), dtype=np.uint8)
                    )
                    continue
                raw_frames = load_video_frames(video_path)
                # Align frame count with keep mask
                raw_frames = raw_frames[: len(keep)]
                raw_frames = raw_frames[keep[: len(raw_frames)]]
                # Resize
                resized = resize_images_parallel(raw_frames, img_resolution, num_workers)
                # (T, H, W, C) -> (T, C, H, W)
                resized = resized.transpose(0, 3, 1, 2)
                all_images[cam_idx].append(resized)

    if not all_states:
        raise RuntimeError("No trajectories were successfully processed.")

    # Concatenate
    traj_lengths_arr = np.array(traj_lengths)
    all_states_arr = np.concatenate(all_states, axis=0)
    all_actions_arr = np.concatenate(all_actions, axis=0)

    print(f"\nTotal trajectories: {len(traj_lengths)}")
    print(f"Total timesteps: {all_states_arr.shape[0]}")
    print(f"State dim: {all_states_arr.shape[1]}, Action dim: {all_actions_arr.shape[1]}")

    # Normalize to [-1, 1]
    obs_min = np.min(all_states_arr, axis=0)
    obs_max = np.max(all_states_arr, axis=0)
    action_min = np.min(all_actions_arr, axis=0)
    action_max = np.max(all_actions_arr, axis=0)

    raw_states = all_states_arr.copy()
    raw_actions = all_actions_arr.copy()
    all_states_arr = 2 * (all_states_arr - obs_min) / (obs_max - obs_min + 1e-6) - 1
    all_actions_arr = 2 * (all_actions_arr - action_min) / (action_max - action_min + 1e-6) - 1

    delta_min = np.min(np.stack(diff_mins), axis=0) if diff_mins else np.zeros_like(obs_min)
    delta_max = np.max(np.stack(diff_maxs), axis=0) if diff_maxs else np.ones_like(obs_max)

    # Build output dict
    output: dict[str, object] = {
        "traj_lengths": traj_lengths_arr,
        "states": all_states_arr,
        "actions": all_actions_arr,
        "raw_states": raw_states,
        "raw_actions": raw_actions,
    }

    if save_image and camera_serials:
        images_dict: dict[int, np.ndarray] = {}
        for cam_idx in range(len(camera_serials)):
            if all_images[cam_idx]:
                images_dict[cam_idx] = np.concatenate(all_images[cam_idx], axis=0)
                print(f"Camera {cam_idx} images: {images_dict[cam_idx].shape}")
        output["images"] = images_dict

    # Save
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.npz")
    print(f"Saving train data to {train_path} ...")
    np.savez_compressed(train_path, **output)

    norm_path = os.path.join(output_dir, "normalization.npz")
    np.savez(
        norm_path,
        obs_min=obs_min,
        obs_max=obs_max,
        action_min=action_min,
        action_max=action_max,
        delta_min=delta_min,
        delta_max=delta_max,
    )
    print(f"Saved normalization to {norm_path}")

    # Config summary
    config_path = os.path.join(output_dir, "config.txt")
    with open(config_path, "w") as f:
        for k, v in {
            "input_dirs": input_dirs,
            "action_keys": action_keys,
            "observation_keys": observation_keys,
            "img_resolution": img_resolution,
            "camera_serials": camera_serials,
            "num_trajectories": len(traj_lengths),
            "total_timesteps": int(all_states_arr.shape[0]),
            "obs_dim": int(all_states_arr.shape[1]),
            "action_dim": int(all_actions_arr.shape[1]),
            "obs_min": obs_min.tolist(),
            "obs_max": obs_max.tolist(),
            "action_min": action_min.tolist(),
            "action_max": action_max.tolist(),
        }.items():
            f.write(f"{k}: {v}\n")

    print(f"Done! Output saved to {output_dir}")
    return output_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process DROID demonstrations for DPPO training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir", type=str, nargs="+", required=True,
        help="DROID data directory (or directories) with timestamped subdirectories",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for processed NPZ files",
    )
    parser.add_argument(
        "--action_keys", type=str, nargs="+",
        default=["joint_position", "gripper_position"],
    )
    parser.add_argument(
        "--observation_keys", type=str, nargs="+",
        default=["joint_positions", "gripper_position"],
    )
    parser.add_argument("--horizon_steps", type=int, default=16)
    parser.add_argument("--img_resolution", type=int, nargs=2, default=[192, 192])
    parser.add_argument(
        "--camera_serials", type=str, nargs="+", default=None,
        help="Camera serial numbers (auto-detected from metadata if omitted)",
    )
    parser.add_argument(
        "--camera_types", type=str, nargs="+", default=["wrist", "ext1"],
        choices=["wrist", "ext1", "ext2"],
    )
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--skip_image", action="store_true")
    parser.add_argument("--max_trajectories", type=int, default=None)
    parser.add_argument("--obs_gripper_threshold", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    input_dir = args.input_dir[0] if len(args.input_dir) == 1 else args.input_dir

    process_droid_dataset(
        input_dir=input_dir,
        output_dir=args.output_dir,
        action_keys=args.action_keys,
        observation_keys=args.observation_keys,
        horizon_steps=args.horizon_steps,
        img_resolution=tuple(args.img_resolution),
        camera_serials=args.camera_serials,
        camera_types=args.camera_types,
        num_workers=args.num_workers,
        skip_image=args.skip_image,
        max_trajectories=args.max_trajectories,
        obs_gripper_threshold=args.obs_gripper_threshold,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
