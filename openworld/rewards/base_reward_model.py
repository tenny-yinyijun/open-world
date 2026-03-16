from abc import ABC, abstractmethod
from typing import Any, Dict


class RewardModel(ABC):
    """Base interface for reward / progress models.

    A reward model scores a trajectory (sequence of frames + metadata)
    and returns per-frame or aggregate reward signals.
    """

    @abstractmethod
    def compute(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        """Compute reward signals for a trajectory.

        Args:
            trajectory: Dict containing at least:
                "frames": list of RGB frames.
                "instruction": optional language instruction.

        Returns:
            Dict containing reward signals, e.g.:
                "reward": aggregate scalar reward.
                "per_frame_progress": list of per-frame progress values.
                "success": boolean success indicator.
        """
        pass
