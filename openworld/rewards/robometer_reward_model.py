from __future__ import annotations

from typing import Any, Dict, Optional

from openworld.rewards.base_reward_model import RewardModel


class RobometerRewardModel(RewardModel):
    """Adapter for a Robometer-compatible reward service.

    The initial integration assumes Robometer is exposed behind an HTTP API.
    Keeping the adapter client-side avoids pulling the full upstream training
    environment into the default OpenWorld install.
    """

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
                "RobometerRewardModel.compute() requires a configured Robometer "
                "service endpoint. Install the optional dependencies with "
                '`uv sync --extra reward-robometer` and provide '
                "`endpoint_url` in the reward-model config."
            )

        raise NotImplementedError(
            "Robometer reward-service wiring is not implemented yet. "
            "This adapter exists so the backend can be selected lazily and "
            "packaged independently from the base environment."
        )
