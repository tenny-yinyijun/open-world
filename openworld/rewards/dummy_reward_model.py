from typing import Any, Dict

from openworld.rewards.base_reward_model import RewardModel


class DummyRewardModel(RewardModel):
    """Returns a placeholder reward for testing."""

    def compute(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        frames = trajectory.get("frames", [])
        return {
            "reward": 0.0,
            "per_frame_progress": [0.0] * len(frames),
            "success": False,
        }
