"""Stub for DSRL-style RL fine-tuning.

The training loop will:
1. Collect rollouts in the world-model environment.
2. Score them with a reward model.
3. Update the policy using an RL algorithm (e.g. SAC with diffusion noise).

See /n/fs/iromdata/projects/dsrl for the reference implementation.
"""

from typing import Any, Dict, Optional

from openworld.envs.world_model_env import WorldModelEnv
from openworld.policies.base_policy import Policy
from openworld.rewards.base_reward_model import RewardModel


class RLFineTuneRunner:
    """Scaffold for RL fine-tuning of policies inside a world-model env."""

    def __init__(
        self,
        env: WorldModelEnv,
        policy: Policy,
        reward_model: Optional[RewardModel] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.env = env
        self.policy = policy
        self.reward_model = reward_model
        self.config = config or {}

    def train(self) -> None:
        # TODO: implement DSRL-style training loop:
        #   1. Rollout policy in world-model env
        #   2. Compute rewards via reward model
        #   3. Update policy parameters via RL objective
        raise NotImplementedError(
            "RLFineTuneRunner.train() is not yet implemented. "
            "This is a structural placeholder for future DSRL integration."
        )
