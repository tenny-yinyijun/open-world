from typing import Any, Dict, List, Optional

from openworld.datasets.initialization import Initialization
from openworld.datasets.initialization_dataset import InitializationDataset
from openworld.envs.world_model_env import WorldModelEnv
from openworld.policies.base_policy import Policy
from openworld.rewards.base_reward_model import RewardModel
from openworld.utils.video import render_observation_frame
from openworld.utils.video import save_rollout_video


class Evaluator:
    """Runs policy rollouts inside a world-model environment and collects results.

    Optionally scores each episode with a :class:`RewardModel`.
    """

    def __init__(
        self,
        env: WorldModelEnv,
        policy: Policy,
        reward_model: Optional[RewardModel] = None,
    ):
        self.env = env
        self.policy = policy
        self.reward_model = reward_model

    def run_episode(
        self,
        initialization: Initialization,
        max_steps: int = 50,
    ) -> Dict[str, Any]:
        """Run a single episode from the given initialization.

        Returns:
            Dict with keys: ``frames``, ``metadata``, ``reward_info``.
        """
        info = self.env.reset(initialization)
        self.policy.reset(instruction=initialization.instruction)

        all_frames: List[Any] = [render_observation_frame(info["observation"])]

        for step in range(max_steps):
            obs = self.env.get_current_observation()
            state = self.env.get_current_state()

            action = self.policy.act(
                observation=obs,
                state=state,
                instruction=initialization.instruction,
            )

            step_info = self.env.step(action)

            if step_info["did_rollout"]:
                all_frames.extend(step_info["predicted_frames"])

        result: Dict[str, Any] = {
            "initialization_id": initialization.id,
            "frames": all_frames,
            "num_steps": max_steps,
            "metadata": initialization.metadata,
        }

        if self.reward_model is not None:
            trajectory = {
                "frames": all_frames,
                "instruction": initialization.instruction,
            }
            result["reward_info"] = self.reward_model.compute(trajectory)

        return result

    def run_dataset(
        self,
        dataset: InitializationDataset,
        max_steps: int = 50,
        video_dir: Optional[str] = None,
        video_fps: int = 5,
    ) -> List[Dict[str, Any]]:
        """Run episodes for every initialization in the dataset.

        Args:
            dataset: The initialization dataset to iterate over.
            max_steps: Maximum number of environment steps per episode.
            video_dir: If provided, save rollout videos to this directory.
            video_fps: Frames per second for saved videos.

        Returns:
            List of per-episode result dicts.
        """
        results: List[Dict[str, Any]] = []

        for init in dataset:
            episode_result = self.run_episode(init, max_steps=max_steps)
            results.append(episode_result)

            if video_dir and episode_result["frames"]:
                save_rollout_video(
                    frames=episode_result["frames"],
                    output_path=f"{video_dir}/{init.id}.mp4",
                    fps=video_fps,
                )

        return results
