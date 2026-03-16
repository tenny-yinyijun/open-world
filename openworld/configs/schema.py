"""Lightweight config schema definitions.

These dataclasses mirror the YAML config structure and make it easy
to validate / auto-complete config values.  Full Hydra integration
can be layered on top later.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class WorldModelConfig:
    name: str = "dummy"
    checkpoint_path: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyConfig:
    name: str = "dp"
    checkpoint_path: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RewardModelConfig:
    name: str = "dummy"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SchedulerConfig:
    chunk_size: int = 15


@dataclass
class EvaluationConfig:
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    reward_model: Optional[RewardModelConfig] = None
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    max_steps: int = 50
    video_dir: Optional[str] = None
    dataset_path: Optional[str] = None


@dataclass
class RLConfig:
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    reward_model: RewardModelConfig = field(default_factory=RewardModelConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    max_steps: int = 50
    train_params: Dict[str, Any] = field(default_factory=dict)
