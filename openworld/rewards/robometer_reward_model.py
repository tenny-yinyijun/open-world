from __future__ import annotations

from typing import Any, Dict

from openworld.rewards.base_reward_model import RewardModel


class RobometerRewardModel(RewardModel):
    """Placeholder for the Robometer reward model.

    Robometer scoring runs in a separate venv via subprocess
    (see ``scripts/score_videos_robometer.py``).  This class exists so the
    registry entry resolves, but ``compute()`` should not be called directly
    during the normal two-phase evaluation pipeline.
    """

    def __init__(self, **_: Any) -> None:
        pass

    def compute(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        raise RuntimeError(
            "RobometerRewardModel.compute() should not be called directly. "
            "Robometer scoring runs as a subprocess after video generation. "
            "See scripts/run_evaluation.py for the two-phase pipeline."
        )
