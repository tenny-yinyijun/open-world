import argparse
import json
import logging
from pathlib import Path

from openworld.datasets import Initialization, InitializationDataset
from openworld.envs import ActionChunkScheduler, WorldModelEnv
from openworld.policies import build_policy
from openworld.rewards import build_reward_model
from openworld.runners import Evaluator
from openworld.utils.io import ensure_dir, load_yaml
from openworld.world_models import build_world_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _resolve_dataset_path(config_path: str, dataset_path: str) -> str:
    dataset = Path(dataset_path)
    if dataset.is_absolute():
        return str(dataset)
    if dataset.exists():
        return str(dataset.resolve())
    return str((Path(config_path).resolve().parent / dataset).resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run world-model evaluation")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to evaluation YAML config"
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    config_path = Path(args.config).resolve()

    # --- Build components ---
    wm_cfg = cfg.get("world_model", {})
    world_model = build_world_model(wm_cfg.get("name", "dummy"), **wm_cfg.get("params", {}))
    if wm_cfg.get("checkpoint_path"):
        world_model.load_checkpoint(wm_cfg["checkpoint_path"])

    scheduler = ActionChunkScheduler(
        chunk_size=cfg.get("scheduler", {}).get("chunk_size", 15)
    )

    env = WorldModelEnv(world_model=world_model, action_chunk_scheduler=scheduler)

    pol_cfg = cfg.get("policy", {})
    policy = build_policy(pol_cfg.get("name", "dp"), **pol_cfg.get("params", {}))
    if pol_cfg.get("checkpoint_path"):
        policy.load_checkpoint(pol_cfg["checkpoint_path"])

    reward_model = None
    rm_cfg = cfg.get("reward_model")
    if rm_cfg:
        reward_model = build_reward_model(rm_cfg.get("name", "dummy"), **rm_cfg.get("params", {}))

    # --- Build dataset ---
    dataset_path = cfg.get("dataset_path")
    dataset_entries = cfg.get("dataset", [])
    if dataset_path:
        dataset = InitializationDataset.from_yaml(
            _resolve_dataset_path(str(config_path), dataset_path)
        )
    elif dataset_entries:
        dataset = InitializationDataset.from_list(dataset_entries)
    else:
        # Fallback: single dummy initialization for smoke-testing
        dataset = InitializationDataset([
            Initialization(
                id="dummy_init_0",
                initial_state=None,
                initial_observation=None,
                instruction="pick up the cup",
            )
        ])

    # --- Run evaluation ---
    evaluator = Evaluator(env=env, policy=policy, reward_model=reward_model)

    chunk_size = cfg.get("scheduler", {}).get("chunk_size", 15)
    action_hz = cfg.get("action_hz", 15)
    num_frames = wm_cfg.get("params", {}).get("num_frames", 5)

    # Convert duration (seconds) to max_steps.  Fall back to legacy max_steps.
    if "duration" in cfg:
        duration = cfg["duration"]
        max_steps = int(duration * action_hz)
    else:
        max_steps = cfg.get("max_steps", 50)
        duration = max_steps / action_hz

    # Video fps: each rollout covers chunk_size/action_hz seconds and produces
    # num_frames frames, so fps = num_frames / (chunk_size / action_hz).
    video_fps = int(num_frames * action_hz / chunk_size)

    video_dir = cfg.get("video_dir")
    logger.info(
        "Evaluation config: world_model=%s policy=%s chunk_size=%s "
        "duration=%.1fs max_steps=%d video_fps=%d dataset=%s",
        wm_cfg.get("name", "dummy"),
        pol_cfg.get("name", "dp"),
        chunk_size,
        duration,
        max_steps,
        video_fps,
        dataset_path or "<inline>",
    )

    results = evaluator.run_dataset(
        dataset, max_steps=max_steps, video_dir=video_dir, video_fps=video_fps,
    )

    # --- Print summary ---
    logger.info("Evaluation complete: %d episodes", len(results))
    for r in results:
        summary = {"id": r["initialization_id"], "num_frames": len(r["frames"])}
        if "reward_info" in r:
            summary["reward"] = r["reward_info"].get("reward")
            summary["success"] = r["reward_info"].get("success")
        logger.info(json.dumps(summary))


if __name__ == "__main__":
    main()
