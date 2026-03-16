from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class WorldModel(ABC):
    """Base interface for video world models.

    A world model takes the current state (typically VAE latents), an observation
    (image or latent), and an action chunk, then predicts future frames and the
    next state.
    """

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model weights from a checkpoint."""
        pass

    @abstractmethod
    def rollout(
        self,
        state: Any,
        observation: Any,
        action_chunk: Any,
        instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a single world-model rollout given a state and action chunk.

        Args:
            state: Current environment state (e.g. VAE latent history).
            observation: Current visual observation (e.g. image latent).
            action_chunk: A sequence of actions to condition the prediction on.
            instruction: Optional language instruction for the task.

        Returns:
            Dict containing at least:
                "frames": predicted future frames (latent or decoded).
                "next_state": the predicted state after executing the actions.
        """
        pass
