from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Initialization:
    """A single episode initialization for the world-model environment.

    Attributes:
        id: Unique identifier for this initialization.
        initial_state: Starting state (e.g. VAE latent history or robot state).
        initial_observation: Starting visual observation (e.g. image or latent).
        instruction: Optional language instruction describing the task.
        metadata: Optional extra metadata (dataset name, camera info, etc.).
    """

    id: str
    initial_state: Any
    initial_observation: Any
    instruction: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
