from typing import Any, Dict, Optional

import numpy as np

from openworld.world_models.base_world_model import WorldModel


class DummyWorldModel(WorldModel):
    """A minimal world model that returns random frames for testing."""

    def __init__(
        self,
        num_pred_frames: int = 5,
        frame_shape: tuple = (576, 320, 3),
    ):
        self.num_pred_frames = num_pred_frames
        self.frame_shape = frame_shape

    def load_checkpoint(self, checkpoint_path: str) -> None:
        pass

    def rollout(
        self,
        state: Any,
        observation: Any,
        action_chunk: Any,
        instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        frames = [
            np.random.randint(0, 256, self.frame_shape, dtype=np.uint8)
            for _ in range(self.num_pred_frames)
        ]
        return {
            "frames": frames,
            "next_state": state,
        }
