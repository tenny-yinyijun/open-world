from typing import Any, Dict, Optional

import numpy as np

from openworld.policies.openpi_action_adapter import resolve_initial_joint_state
from openworld.envs.action_chunk_scheduler import ActionChunkScheduler
from openworld.world_models.base_world_model import WorldModel


class WorldModelEnv:
    """Gym-like environment wrapper that drives a world model.

    On each ``step`` call the action is buffered in an
    :class:`ActionChunkScheduler`.  Once a full chunk is accumulated the
    world model is called to produce predicted frames and a next state.
    Until then, steps return the current observation with metadata
    indicating that no rollout has occurred yet.
    """

    def __init__(
        self,
        world_model: WorldModel,
        action_chunk_scheduler: ActionChunkScheduler,
    ):
        self.world_model = world_model
        self.scheduler = action_chunk_scheduler

        self._current_state: Any = None
        self._current_observation: Any = None
        self._instruction: Optional[str] = None
        self._predicted_frames: list = []
        self._step_count: int = 0
        self._action_history: list[np.ndarray] = []
        self._robot_state_history: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, initialization) -> Dict[str, Any]:
        """Reset the environment to a given initialization.

        Args:
            initialization: An :class:`~openworld.datasets.Initialization`
                instance (or any object exposing ``initial_state``,
                ``initial_observation``, and optionally ``instruction``).

        Returns:
            Info dict with the initial observation.
        """
        self._current_state = self._bootstrap_initial_state(initialization)
        self._current_observation = initialization.initial_observation
        self._instruction = getattr(initialization, "instruction", None)
        self._predicted_frames = []
        self._step_count = 0
        self._action_history = []
        self._robot_state_history = []
        self.scheduler.reset()
        initial_robot_state = self._extract_robot_state_vector(self._current_state)
        if initial_robot_state is not None:
            self._robot_state_history.append(initial_robot_state)

        return {
            "observation": self._current_observation,
            "state": self._current_state,
            "did_rollout": False,
            "predicted_frames": [],
        }

    def step(self, action: Any) -> Dict[str, Any]:
        """Execute one environment step.

        The action is appended to the internal buffer.  If enough actions
        have been accumulated, the world model is called to produce the
        next observation and state.

        Args:
            action: A single action to buffer.

        Returns:
            A dict with keys:
                observation: Current visual observation.
                state: Current environment state.
                did_rollout: Whether a world-model rollout was triggered.
                predicted_frames: List of frames from the latest rollout
                    (empty if no rollout occurred this step).
        """
        env_action = self._extract_env_action(action)
        self._current_state = self._advance_policy_state(self._current_state, action)
        action_array = np.asarray(env_action, dtype=np.float32).reshape(-1)
        self._action_history.append(action_array)
        self.scheduler.append(action_array)
        self._step_count += 1
        current_robot_state = self._extract_robot_state_vector(self._current_state)
        if current_robot_state is not None:
            self._robot_state_history.append(current_robot_state)

        if not self.scheduler.is_ready():
            return {
                "observation": self._current_observation,
                "state": self._current_state,
                "did_rollout": False,
                "predicted_frames": [],
            }

        # Trigger world-model rollout
        action_chunk = self.scheduler.get_chunk()
        rollout_state = self._augment_rollout_state(self._current_state, action_chunk)
        result = self.world_model.rollout(
            state=rollout_state,
            observation=self._current_observation,
            action_chunk=action_chunk,
            instruction=self._instruction,
        )

        self._predicted_frames = result["frames"]
        self._current_state = self._merge_state(result["next_state"], self._current_state)
        # Use the last predicted frame as the new observation.
        if self._predicted_frames:
            self._current_observation = self._predicted_frames[-1]

        return {
            "observation": self._current_observation,
            "state": self._current_state,
            "did_rollout": True,
            "predicted_frames": list(self._predicted_frames),
        }

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_current_observation(self) -> Any:
        return self._current_observation

    def get_current_state(self) -> Any:
        return self._current_state

    def get_predicted_frames(self) -> list:
        return list(self._predicted_frames)

    @staticmethod
    def _merge_state(next_state: Any, previous_state: Any) -> Any:
        if not isinstance(next_state, dict):
            return next_state
        if not isinstance(previous_state, dict):
            return next_state

        merged = dict(next_state)

        # Preserve robot-side state for policies that need proprioception, while
        # still letting the world model update its latent state fields.
        if "robot" in previous_state and "robot" not in merged:
            merged["robot"] = previous_state["robot"]
        if "state" in previous_state and "state" not in merged:
            merged["state"] = previous_state["state"]
        if "robot_state" in previous_state and "robot_state" not in merged:
            merged["robot_state"] = previous_state["robot_state"]

        return merged

    @classmethod
    def _advance_policy_state(cls, state: Any, action: Any) -> Any:
        if not isinstance(state, dict):
            return state

        if isinstance(action, dict):
            state_update = action.get("state_update")
            env_action = action.get("env_action")
            updated = dict(state)
            if isinstance(state_update, dict):
                updated = cls._deep_merge_state(updated, state_update)
            if env_action is None:
                return updated
            if (
                isinstance(updated.get("robot"), dict)
                and "state" in updated["robot"]
                and "state_representation" in updated["robot"]
                and updated["robot"].get("state_representation") == "cartesian_position_with_gripper"
            ):
                updated["robot"]["state"] = np.asarray(env_action, dtype=np.float32).reshape(-1)
            return updated

        action_vector = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_vector.size == 0:
            return state

        updated = dict(state)

        if "robot" in state and isinstance(state["robot"], dict):
            updated["robot"] = cls._advance_robot_mapping(state["robot"], action_vector)

        if "robot_state" in state and isinstance(state["robot_state"], dict):
            updated["robot_state"] = cls._advance_robot_state_dict(
                state["robot_state"],
                action_vector,
            )

        if "state" in state:
            updated_state = cls._advance_state_vector(state["state"], action_vector)
            if updated_state is not None:
                updated["state"] = updated_state

        return updated

    @staticmethod
    def _extract_env_action(action: Any) -> Any:
        if isinstance(action, dict) and "env_action" in action:
            return action["env_action"]
        return action

    @staticmethod
    def _deep_merge_state(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base)
        for key, value in update.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = WorldModelEnv._deep_merge_state(merged[key], value)
            else:
                merged[key] = value
        return merged

    @staticmethod
    def _bootstrap_initial_state(initialization: Any) -> Any:
        state = initialization.initial_state
        if not isinstance(state, dict):
            return state

        updated = dict(state)
        robot = dict(updated.get("robot", {})) if isinstance(updated.get("robot"), dict) else {}
        joint_position, gripper_position = resolve_initial_joint_state(state, getattr(initialization, "metadata", None))
        if joint_position is not None:
            robot.setdefault("joint_position", joint_position)
            robot.setdefault("joint_positions", joint_position)
        if gripper_position is not None:
            robot.setdefault("gripper_position", gripper_position)
        if robot:
            updated["robot"] = robot
        return updated

    @classmethod
    def _advance_robot_mapping(
        cls,
        robot_state: dict[str, Any],
        action: np.ndarray,
    ) -> dict[str, Any]:
        updated = dict(robot_state)

        if "state" in robot_state:
            advanced = cls._advance_state_vector(robot_state["state"], action)
            if advanced is not None:
                updated["state"] = advanced

        if "cartesian_position" in robot_state:
            cartesian = np.asarray(robot_state["cartesian_position"], dtype=np.float32).reshape(-1)
            if cartesian.size >= 6 and action.size >= 6:
                updated["cartesian_position"] = cartesian.copy()
                updated["cartesian_position"][:6] = cartesian[:6] + action[:6]

        if "gripper_position" in robot_state and action.size >= 1:
            updated["gripper_position"] = np.asarray([action[-1]], dtype=np.float32)

        return updated

    @staticmethod
    def _advance_robot_state_dict(
        robot_state: dict[str, Any],
        action: np.ndarray,
    ) -> dict[str, Any]:
        updated = dict(robot_state)

        if "joint_positions" in robot_state:
            joints = np.asarray(robot_state["joint_positions"], dtype=np.float32).reshape(-1)
            if joints.size >= action.size - 1 and action.size >= 2:
                next_joints = joints.copy()
                next_joints[: action.size - 1] = joints[: action.size - 1] + action[:-1]
                updated["joint_positions"] = next_joints

        if "gripper_position" in robot_state and action.size >= 1:
            updated["gripper_position"] = np.asarray([action[-1]], dtype=np.float32)

        return updated

    @staticmethod
    def _advance_state_vector(state: Any, action: np.ndarray) -> Any:
        vector = np.asarray(state, dtype=np.float32).reshape(-1)
        if vector.size == 0:
            return None

        next_vector = vector.copy()
        shared_dims = min(max(action.size - 1, 0), max(vector.size - 1, 0))
        if shared_dims > 0:
            next_vector[:shared_dims] = vector[:shared_dims] + action[:shared_dims]
        if vector.size >= 1 and action.size >= 1:
            next_vector[-1] = action[-1]
        return next_vector

    def _augment_rollout_state(self, state: Any, action_chunk: Any) -> Any:
        if not isinstance(state, dict):
            return state

        chunk = np.asarray(action_chunk, dtype=np.float32)
        if chunk.ndim == 1:
            chunk = chunk.reshape(1, -1)
        prior_count = max(len(self._action_history) - int(chunk.shape[0]), 0)
        prior_actions = np.asarray(self._action_history[:prior_count], dtype=np.float32)
        prior_robot_states = np.asarray(self._robot_state_history[:prior_count + 1], dtype=np.float32)

        augmented = dict(state)
        augmented["_action_history"] = prior_actions
        augmented["_robot_state_history"] = prior_robot_states
        return augmented

    @staticmethod
    def _extract_robot_state_vector(state: Any) -> Optional[np.ndarray]:
        if not isinstance(state, dict):
            return None

        robot = state.get("robot")
        if isinstance(robot, dict) and "state" in robot:
            return np.asarray(robot["state"], dtype=np.float32).reshape(-1)

        if "state" in state:
            return np.asarray(state["state"], dtype=np.float32).reshape(-1)

        return None
