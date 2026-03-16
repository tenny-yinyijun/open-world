from __future__ import annotations

from typing import Any, Dict, Optional

from openworld.rewards.base_reward_model import RewardModel


class TOPRewardModel(RewardModel):
    """Adapter for a TOPReward-compatible reward service."""

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        timeout_s: float = 30.0,
        **_: Any,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.timeout_s = timeout_s

    def compute(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        if not self.endpoint_url:
            raise NotImplementedError(
                "TOPRewardModel.compute() requires a configured TOPReward "
                "service endpoint. Install the optional dependencies with "
                '`uv sync --extra reward-topreward` and provide '
                "`endpoint_url` in the reward-model config."
            )

        raise NotImplementedError(
            "TOPReward reward-service wiring is not implemented yet. "
            "This adapter keeps the TOPReward backend optional and isolated "
            "from the default world-model environment."
        )
