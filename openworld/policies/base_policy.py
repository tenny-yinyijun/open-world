from abc import ABC, abstractmethod
from typing import Any, Optional


class Policy(ABC):
    """Unified interface for robot policies.

    All external policy implementations (OpenPI, DP/DPPO, etc.) should be
    wrapped behind this API so that the rest of the codebase never depends
    on fork-specific internals.
    """

    @abstractmethod
    def reset(self, instruction: Optional[str] = None) -> None:
        """Reset internal policy state for a new episode."""
        pass

    @abstractmethod
    def act(
        self,
        observation: Any,
        state: Any,
        instruction: Optional[str] = None,
    ) -> Any:
        """Return an action given the current observation and state.

        Args:
            observation: Current visual observation (e.g. RGB images).
            state: Current robot state (e.g. joint positions).
            instruction: Optional language instruction.

        Returns:
            A single action (will be buffered by the ActionChunkScheduler).
        """
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load policy weights from a checkpoint."""
        pass
