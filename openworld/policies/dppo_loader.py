from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any, Optional

import hydra
from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import torch


DEFAULT_DPPO_REPO = Path(__file__).resolve().parents[2] / "external" / "dsrl" / "dppo"
DEFAULT_POLICY_JSON = DEFAULT_DPPO_REPO / "asset" / "policy.json"


def ensure_dppo_repo_on_path(repo_path: Optional[str] = None) -> Path:
    repo_root = Path(repo_path or DEFAULT_DPPO_REPO).resolve()
    if not repo_root.exists():
        raise FileNotFoundError(
            f"DPPO repo not found at {repo_root}. Clone your DPPO fork into "
            "`external/dsrl/dppo` or set `repo_path` explicitly in the policy config."
        )
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return repo_root


def register_omegaconf_resolvers() -> None:
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("round_up", math.ceil, replace=True)
    OmegaConf.register_new_resolver("round_down", math.floor, replace=True)


def load_policy_paths_from_alias(
    policy_alias: str,
    policy_json_path: Optional[str] = None,
) -> dict[str, str]:
    policy_json = Path(policy_json_path or DEFAULT_POLICY_JSON).resolve()
    if not policy_json.exists():
        raise FileNotFoundError(
            f"Policy JSON file not found at: {policy_json}. "
            "Expected it under your DPPO fork in `external/dsrl/dppo/asset/policy.json` "
            "unless `policy_json` is overridden."
        )

    with policy_json.open("r") as handle:
        policy_data = json.load(handle)

    if policy_alias not in policy_data:
        available = [key for key in policy_data if not key.startswith("_")]
        raise ValueError(
            f"Policy alias '{policy_alias}' not found in {policy_json}. "
            f"Available aliases: {available}"
        )

    alias_data = policy_data[policy_alias]
    required = ("config_path", "checkpoint_path", "norm_stats")
    missing = [field for field in required if field not in alias_data]
    if missing:
        raise ValueError(
            f"Policy alias '{policy_alias}' is missing required fields: {missing}"
        )

    return {field: alias_data[field] for field in required}


def load_policy_from_checkpoint(
    *,
    config_path: str,
    checkpoint_path: str,
    normalization_stats_path: str,
    device: str = "cuda",
    repo_path: Optional[str] = None,
    ordered_obs_keys: Optional[list[str]] = None,
    camera_indices: Optional[list[int]] = None,
    act_steps: Optional[int] = None,
) -> Any:
    ensure_dppo_repo_on_path(repo_path)
    register_omegaconf_resolvers()

    from dppo.serving.policy_wrapper import DiffusionPolicyWrapper

    resolved_config_path = Path(config_path).resolve()
    with initialize_config_dir(
        config_dir=str(resolved_config_path.parent),
        version_base=None,
    ):
        cfg = compose(config_name=resolved_config_path.stem)

    normalization_stats = np.load(normalization_stats_path, allow_pickle=True)
    model = hydra.utils.instantiate(cfg.model)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "ema" in checkpoint:
        model.load_state_dict(checkpoint["ema"], strict=True)
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)

    resolved_ordered_obs_keys = ordered_obs_keys
    if resolved_ordered_obs_keys is None:
        if hasattr(cfg, "ordered_obs_keys") and cfg.ordered_obs_keys is not None:
            resolved_ordered_obs_keys = list(cfg.ordered_obs_keys)
        else:
            resolved_ordered_obs_keys = ["joint_positions", "gripper_position"]

    resolved_camera_indices = camera_indices
    if resolved_camera_indices is None:
        if hasattr(cfg, "camera_indices") and cfg.camera_indices is not None:
            resolved_camera_indices = [
                int(idx) if isinstance(idx, str) else idx
                for idx in cfg.camera_indices
            ]
        elif hasattr(cfg, "train_dataset") and hasattr(cfg.train_dataset, "num_img_views"):
            resolved_camera_indices = list(range(int(cfg.train_dataset.num_img_views)))
        else:
            resolved_camera_indices = [0, 1]

    resolved_act_steps = act_steps
    if resolved_act_steps is None:
        if hasattr(cfg, "act_steps") and cfg.act_steps is not None:
            resolved_act_steps = min(int(cfg.act_steps), int(cfg.horizon_steps))
        else:
            resolved_act_steps = int(cfg.horizon_steps)

    use_img = (
        hasattr(cfg, "train_dataset")
        and hasattr(cfg.train_dataset, "use_img")
        and bool(cfg.train_dataset.use_img)
    )

    return DiffusionPolicyWrapper(
        model=model,
        normalization_stats=normalization_stats,
        ordered_obs_keys=resolved_ordered_obs_keys,
        camera_indices=resolved_camera_indices if use_img else [],
        n_cond_step=int(cfg.cond_steps),
        n_img_cond_step=int(cfg.img_cond_steps) if use_img else 1,
        act_steps=resolved_act_steps,
        device=device,
        use_delta_actions=False,
    )
