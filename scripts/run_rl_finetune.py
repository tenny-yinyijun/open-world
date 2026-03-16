import argparse
import logging

from openworld.envs import ActionChunkScheduler, WorldModelEnv
from openworld.policies import build_policy
from openworld.rewards import build_reward_model
from openworld.runners import RLFineTuneRunner
from openworld.utils.io import load_yaml
from openworld.world_models import build_world_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RL fine-tuning")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to RL YAML config"
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)

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

    rm_cfg = cfg.get("reward_model", {})
    reward_model = build_reward_model(rm_cfg.get("name", "dummy"), **rm_cfg.get("params", {}))

    # --- Run training ---
    runner = RLFineTuneRunner(
        env=env,
        policy=policy,
        reward_model=reward_model,
        config=cfg.get("train_params", {}),
    )

    logger.info("Starting RL fine-tuning...")
    runner.train()


if __name__ == "__main__":
    main()
