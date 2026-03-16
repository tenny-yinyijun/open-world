"""VidWM (Video World Model) adapter for OpenWorld.

This wraps the shortcut flow-matching video model and exposes it through the
:class:`WorldModel` interface.

The model operates entirely in VAE latent space:
  - State: dict with keys ``current_latent`` (B,4,72,40),
    ``history_latents`` (B,num_history,4,72,40), and
    ``history_action_embeds`` (B,num_history,1024).
  - Observation: the current image latent (B,4,72,40).
  - Action chunk: raw actions (chunk_size, action_dim) -- encoded internally.
  - Output frames: decoded RGB numpy arrays (H,W,3) uint8.

Requires the ``vidwm`` package to be importable (add its repo to sys.path or
install it).  All heavy imports are deferred to avoid import errors when the
package is not available.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image
import torch

from openworld.world_models.base_world_model import WorldModel
from openworld.world_models.vidwm_loader import ensure_vidwm_repo_on_path

logger = logging.getLogger(__name__)


@dataclass
class VidWMConfig:
    """Configuration for the VidWM world model."""

    repo_path: Optional[str] = None
    debug: bool = False
    debug_log_limit: int = 3

    # Pretrained model paths
    svd_model_path: Optional[str] = None
    clip_model_path: Optional[str] = None
    unet_model_path: Optional[str] = None

    # Model architecture
    num_frames: int = 5
    num_history: int = 6
    action_dim: int = 7
    width: int = 320
    height: int = 192
    frame_level_cond: bool = True
    text_cond: bool = True
    his_cond_zero: bool = False
    dtype: str = "bfloat16"

    # Inference parameters
    motion_bucket_id: int = 127
    fps: int = 7
    guidance_scale: float = 2.0
    num_inference_steps: int = 4
    decode_chunk_size: int = 7

    # Flow-matching / shortcut parameters
    flow_map_type: str = "flow_map"
    flow_map_loss_type: str = "psd"
    distance_conditioning: bool = True
    SIGMA_MAX: float = 700.0
    SIGMA_MIN: float = 0.02

    # Action encoder
    action_encoder_type: str = "unaligned"
    action_encoder_hidden_dim: int = 1024

    # Device
    device: Optional[str] = None

    # Whether to decode latents to RGB (set False to stay in latent space)
    decode_to_rgb: bool = True
    view_order: tuple[str, ...] = ("exterior_left", "exterior_right", "wrist")
    action_downsample: int = 1

    # Action normalization – DROID dataset percentile statistics used during
    # Ctrl-World training.  Actions are normalized to [-1, 1] via:
    #   normed = 2*(action - p01) / (p99 - p01 + eps) - 1
    action_normalize: bool = True
    action_stat_path: Optional[str] = None
    action_state_p01: tuple[float, ...] = (
        0.2676655471324921, -0.441715806722641, -0.042784303426742554,
        -3.1373724937438965, -1.214390754699707, -2.11581574678421, 0.0,
    )
    action_state_p99: tuple[float, ...] = (
        0.781050771474838, 0.4366717040538788, 0.7843997478485107,
        3.137465238571167, 0.9039141565561147, 1.9915846586227417,
        0.9911894202232361,
    )


class VidWMWorldModel(WorldModel):
    """World model backed by the VidWM shortcut flow-matching video model.

    The model is loaded lazily on the first call to :meth:`load_checkpoint`
    (or on construction if ``checkpoint_path`` is passed to the config).
    """

    def __init__(self, config: Optional[Union[VidWMConfig, Dict[str, Any]]] = None, **kwargs: Any):
        if config is None:
            config = VidWMConfig(**kwargs)
        elif isinstance(config, dict):
            config = VidWMConfig(**config)

        # Load normalization stats from JSON file if provided
        if config.action_stat_path is not None:
            import json
            with open(config.action_stat_path) as f:
                stats = json.load(f)
            config.action_state_p01 = tuple(stats["state_01"])
            config.action_state_p99 = tuple(stats["state_99"])

        self.config = config

        # These are populated by load_checkpoint
        self.pipeline = None
        self.action_encoder = None
        self.text_encoder = None
        self.tokenizer = None
        self.text_encoder_is_vit: bool = True
        self._device: torch.device = torch.device(
            config.device if config.device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._dtype = torch.bfloat16 if "bfloat16" in config.dtype else torch.float16
        self._debug_logs_emitted = 0

    # ------------------------------------------------------------------
    # WorldModel interface
    # ------------------------------------------------------------------

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load VidWM pipeline, UNet weights, action encoder, and CLIP.

        ``checkpoint_path`` should point to the combined UNet+action-encoder
        ``.pt`` file (e.g. ``checkpoint-120000.pt``).  The SVD base model and
        CLIP paths are taken from ``self.config``.
        """
        cfg = self.config

        ensure_vidwm_repo_on_path(cfg.repo_path)

        # ---- lazy imports from the vidwm package ----
        from vidwm.video_models.vidwm_diffusion import VidWMDiffusionPipeline
        from vidwm.video_models.utils.svd_unet_utils import UNetSpatioTemporalConditionModel
        from vidwm.video_models.utils.svd_model_utils import load_clip

        # ---- load pretrained SVD pipeline ----
        svd_path = cfg.svd_model_path
        if svd_path is None:
            from huggingface_hub import snapshot_download
            svd_path = snapshot_download(repo_id="stabilityai/stable-video-diffusion-img2vid")

        logger.info("Loading SVD pipeline from %s", svd_path)
        self.pipeline = VidWMDiffusionPipeline.from_pretrained(svd_path)

        # ---- replace UNet with distance-conditioned variant ----
        unet = UNetSpatioTemporalConditionModel(
            distance_conditioning=cfg.distance_conditioning,
        )

        # Load fine-tuned UNet weights
        logger.info("Loading UNet weights from %s", checkpoint_path)
        state_dict = torch.load(checkpoint_path, map_location="cpu")

        unet_state_dict = {
            k[len("unet."):]: v
            for k, v in state_dict.items()
            if k.startswith("unet.")
        }
        unet.load_state_dict(unet_state_dict, strict=False)

        self.pipeline.unet = unet

        # ---- load CLIP text encoder ----
        clip_path = cfg.clip_model_path
        if clip_path is None:
            from huggingface_hub import snapshot_download
            clip_path = snapshot_download(repo_id="openai/clip-vit-base-patch32")

        self.text_encoder_is_vit = "RN50" not in str(clip_path)
        logger.info("Loading CLIP from %s (is_vit=%s)", clip_path, self.text_encoder_is_vit)
        self.text_encoder, self.tokenizer = load_clip(clip_model_path=clip_path)

        # ---- load action encoder ----
        action_num = cfg.num_history + cfg.num_frames
        if cfg.action_encoder_type == "unaligned":
            from vidwm.action_encoders.unaligned_action_encoder import ActionEncoderUnaligned

            self.action_encoder = ActionEncoderUnaligned(
                action_dim=cfg.action_dim,
                action_num=action_num,
                hidden_dim=cfg.action_encoder_hidden_dim,
                text_cond=cfg.text_cond,
            )
        elif "clip_aligned" in cfg.action_encoder_type:
            from vidwm.action_encoders.clip_aligned_action_encoder import ActionEncoderCLIPAligned

            self.action_encoder = ActionEncoderCLIPAligned(
                action_dim=cfg.action_dim,
                action_num=action_num,
                hidden_dim=cfg.action_encoder_hidden_dim,
                text_cond=cfg.text_cond,
            )
        else:
            raise ValueError(f"Unknown action encoder type: {cfg.action_encoder_type}")

        # Load action encoder weights from same checkpoint
        ae_state_dict = {
            k[len("action_encoder."):]: v
            for k, v in state_dict.items()
            if k.startswith("action_encoder.")
        }
        if ae_state_dict:
            logger.info("Loading action encoder weights (%d keys)", len(ae_state_dict))
            self.action_encoder.load_state_dict(ae_state_dict, strict=False)

        # ---- free raw state dict memory ----
        state_dict.clear()
        gc.collect()
        torch.cuda.empty_cache()

        # ---- move to device and set precision ----
        self.pipeline.vae.to(self._dtype)
        self.pipeline.image_encoder.to(self._dtype)
        self.pipeline.unet.to(self._dtype)
        self.action_encoder.to(self._dtype)

        self._move_to_device()

        # ---- freeze everything for inference ----
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.image_encoder.requires_grad_(False)
        self.pipeline.unet.requires_grad_(False)
        self.action_encoder.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        self.pipeline.unet.eval()
        self.action_encoder.eval()

        logger.info("VidWM world model loaded successfully")

    def rollout(
        self,
        state: Any,
        observation: Any,
        action_chunk: Any,
        instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a single world-model rollout.

        Args:
            state: Dict with keys:
                ``current_latent``: (B, 4, H, W) or (4, H, W) current image latent.
                ``history_latents``: (B, num_history, 4, H, W) history latents.
                  If not enough history is available, the current frame is
                  broadcast to fill the gap.
            observation: Ignored (redundant with state["current_latent"]).
            action_chunk: (chunk_size, action_dim) or (B, chunk_size, action_dim)
                raw actions.  The chunk is split internally into history-action
                and future-action portions for the model.
            instruction: Optional language instruction string.

        Returns:
            Dict with:
                ``frames``: list of (H, W, 3) uint8 numpy arrays (if decode_to_rgb)
                    or list of (4, latH, latW) latent tensors.
                ``next_state``: updated state dict for the next rollout call.
                ``latents``: (num_frames, 4, H, W) predicted latent tensor.
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load_checkpoint() first.")

        cfg = self.config
        device = self._device

        # ---- unpack state ----
        current_latent, history_latents = self._unpack_state(state, observation)
        B = current_latent.shape[0]

        # ---- prepare actions ----
        state_history = None
        if isinstance(state, dict):
            state_history = state.get("_robot_state_history")
        actions = self._prepare_actions(action_chunk, B, state_history=state_history)
        self._debug_log_rollout_inputs(
            state=state,
            observation=observation,
            action_chunk=action_chunk,
            prepared_actions=actions,
            instruction=instruction,
        )

        # ---- encode actions ----
        with torch.no_grad():
            action_embeds = self.action_encoder(
                actions,
                texts=[instruction] if instruction else None,
                text_tokenizer=self.tokenizer,
                text_encoder=self.text_encoder,
                frame_level_cond=cfg.frame_level_cond,
                text_encoder_is_vit=self.text_encoder_is_vit,
                device=device,
            )
            action_latent = action_embeds["action_with_text_embeds"]  # (B, T, 1024)

        # ---- split action embeddings into history + future ----
        his_action = action_latent[:, :cfg.num_history]
        future_action = action_latent[:, cfg.num_history:]

        # ---- build combined action embedding ----
        action_combined = torch.cat([his_action, future_action], dim=1)

        # ---- call the diffusion pipeline ----
        with torch.no_grad():
            frame_out, pred_latents = self.pipeline(
                image=current_latent,
                text=action_combined,
                width=cfg.width,
                height=cfg.height * 3,  # 3 camera views stacked
                num_frames=cfg.num_frames,
                history=history_latents,
                num_inference_steps=cfg.num_inference_steps,
                decode_chunk_size=cfg.decode_chunk_size,
                max_guidance_scale=cfg.guidance_scale,
                fps=cfg.fps,
                motion_bucket_id=cfg.motion_bucket_id,
                output_type="latent",
                return_dict=False,
                frame_level_cond=cfg.frame_level_cond,
                his_cond_zero=cfg.his_cond_zero,
                flow_map_type=cfg.flow_map_type,
                flow_map_loss_type=cfg.flow_map_loss_type,
            )

        # pred_latents shape: (B, num_frames, 4, 72, 40)

        # ---- build next state ----
        # Update history: drop oldest frames, append new predictions (minus last)
        # The last predicted frame becomes the new current_latent
        new_current = pred_latents[:, -1]  # (B, 4, 72, 40)
        new_history = self._update_history(
            history_latents, pred_latents[:, :-1], cfg.num_history,
        )

        next_state = {
            "current_latent": new_current,
            "history_latents": new_history,
        }

        # ---- decode frames if requested ----
        if cfg.decode_to_rgb:
            frames = self._decode_latents(pred_latents)
        else:
            frames = [pred_latents[:, i] for i in range(pred_latents.shape[1])]

        return {
            "frames": frames,
            "next_state": next_state,
            "latents": pred_latents,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _move_to_device(self) -> None:
        """Move all model components to the configured device."""
        self.pipeline.unet.to(self._device)
        self.pipeline.vae.to(self._device)
        self.pipeline.image_encoder.to(self._device)
        self.action_encoder.to(self._device)
        self.text_encoder.to(self._device)

    def _unpack_state(self, state: Any, observation: Any) -> tuple:
        """Unpack and validate state dict, returning (current_latent, history_latents)."""
        cfg = self.config
        device = self._device

        if isinstance(state, dict):
            current_latent = state.get("current_latent")
            history_latents = state.get("history_latents", None)
        else:
            current_latent = None
            history_latents = None

        if current_latent is None:
            current_latent = self._bootstrap_current_latent(observation)
            history_latents = None

        # Ensure batch dimension
        if current_latent.dim() == 3:
            current_latent = current_latent.unsqueeze(0)
        current_latent = current_latent.to(device=device, dtype=self._dtype)
        B = current_latent.shape[0]

        # Build or validate history
        if history_latents is None:
            # Broadcast current frame as history
            history_latents = current_latent.unsqueeze(1).expand(
                B, cfg.num_history, -1, -1, -1
            ).contiguous()
        else:
            if history_latents.dim() == 3:
                history_latents = history_latents.unsqueeze(0)
            history_latents = history_latents.to(device=device, dtype=self._dtype)

            # Pad if not enough history
            if history_latents.shape[1] < cfg.num_history:
                pad_count = cfg.num_history - history_latents.shape[1]
                pad = current_latent.unsqueeze(1).expand(B, pad_count, -1, -1, -1)
                history_latents = torch.cat([pad, history_latents], dim=1)

        return current_latent, history_latents

    def _bootstrap_current_latent(self, observation: Any) -> torch.Tensor:
        """Create an initial latent state from raw RGB observations."""
        stacked_rgb = self._stack_observation_views(observation)
        rgb_tensor = (
            torch.from_numpy(stacked_rgb)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            / 255.0 * 2.0 - 1.0  # normalize to [-1, 1] as expected by the VAE
        )

        with torch.no_grad():
            latent = self.encode_image(rgb_tensor)
        return latent

    def _stack_observation_views(self, observation: Any) -> np.ndarray:
        cfg = self.config

        if isinstance(observation, dict):
            views = observation.get("views", observation)
            if isinstance(views, dict):
                frames = []
                for view_name in cfg.view_order:
                    if view_name not in views:
                        raise ValueError(
                            f"Observation is missing required view '{view_name}'. "
                            f"Available views: {list(views)}"
                        )
                    frames.append(self._load_rgb_frame(views[view_name]))
                return np.concatenate(frames, axis=0)

        frame = self._load_rgb_frame(observation)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(
                "VidWM bootstrap expects an RGB frame or a dict of named RGB views."
            )
        return frame

    @staticmethod
    def _load_rgb_frame(value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, str):
            with Image.open(value) as image:
                return np.asarray(image.convert("RGB"))
        raise ValueError(f"Unsupported observation frame value: {type(value)!r}")

    def _prepare_actions(
        self,
        action_chunk: Any,
        batch_size: int,
        state_history: Any = None,
    ) -> torch.Tensor:
        """Convert action chunk to tensor (B, num_history + num_frames, action_dim)."""
        cfg = self.config
        stride = max(int(cfg.action_downsample), 1)

        future_actions = np.asarray(action_chunk, dtype=np.float32)
        if future_actions.ndim == 1:
            future_actions = future_actions.reshape(1, -1)
        future_actions = future_actions[::stride]
        if future_actions.shape[0] > cfg.num_frames:
            future_actions = future_actions[: cfg.num_frames]

        if state_history is None:
            history_actions = np.empty((0, future_actions.shape[-1]), dtype=np.float32)
        else:
            history_actions = np.asarray(state_history, dtype=np.float32)
            if history_actions.size == 0:
                history_actions = np.empty((0, future_actions.shape[-1]), dtype=np.float32)
            elif history_actions.ndim == 1:
                history_actions = history_actions.reshape(1, -1)
        if history_actions.size > 0:
            history_actions = history_actions[::stride]
            history_actions = history_actions[-cfg.num_history :]

        if future_actions.shape[0] == 0:
            raise ValueError("VidWM requires at least one future action.")

        if history_actions.shape[0] < cfg.num_history:
            pad_count = cfg.num_history - history_actions.shape[0]
            history_pad = np.repeat(future_actions[:1], pad_count, axis=0)
            history_actions = np.concatenate([history_pad, history_actions], axis=0)

        if future_actions.shape[0] < cfg.num_frames:
            future_pad = np.repeat(future_actions[-1:], cfg.num_frames - future_actions.shape[0], axis=0)
            future_actions = np.concatenate([future_actions, future_pad], axis=0)

        action_array = np.concatenate([history_actions, future_actions], axis=0)

        # Normalize actions to [-1, 1] using DROID dataset percentile stats,
        # matching the Ctrl-World training convention.
        if cfg.action_normalize:
            p01 = np.asarray(cfg.action_state_p01, dtype=np.float32)
            p99 = np.asarray(cfg.action_state_p99, dtype=np.float32)
            eps = 1e-8
            action_array = np.clip(
                2.0 * (action_array - p01) / (p99 - p01 + eps) - 1.0,
                -1.0, 1.0,
            )

        action_tensor = torch.as_tensor(action_array, dtype=self._dtype, device=self._device).unsqueeze(0)

        if action_tensor.shape[0] == 1 and batch_size > 1:
            action_tensor = action_tensor.expand(batch_size, -1, -1)

        return action_tensor

    def _update_history(
        self,
        old_history: torch.Tensor,
        new_frames: torch.Tensor,
        num_history: int,
    ) -> torch.Tensor:
        """Append new predicted frames to history, keeping only num_history frames."""
        combined = torch.cat([old_history, new_frames], dim=1)
        return combined[:, -num_history:]

    def _decode_latents(self, latents: torch.Tensor) -> List[np.ndarray]:
        """Decode latent tensor to a list of RGB uint8 numpy frames.

        Args:
            latents: (B, T, 4, 72, 40) predicted latent frames.

        Returns:
            List of (H, W, 3) uint8 numpy arrays, one per frame.
            Frames from all batches are flattened into the list.
        """
        from vidwm.video_models.utils.svd_model_utils import svd_tensor2vid

        B, T = latents.shape[:2]
        cfg = self.config

        # The pipeline's decode_latents expects (B, T, C, H, W)
        latents_for_decode = latents.to(self.pipeline.vae.dtype)
        decoded = self.pipeline.decode_latents(
            latents_for_decode, T, cfg.decode_chunk_size,
        )
        # decoded shape: (B, C, T, H, W) float32

        # Convert to video frames using the SVD utility
        frames_list = svd_tensor2vid(decoded, self.pipeline.video_processor, output_type="np")

        # frames_list is a list of length B, each element is a list of T frames
        # Flatten to a single list of frames (from batch 0)
        all_frames = []
        for batch_frames in frames_list:
            for frame in batch_frames:
                # frame is (H, W, 3) float in [0, 1] or uint8
                if frame.dtype != np.uint8:
                    frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                all_frames.append(frame)

        return all_frames

    def _debug_log_rollout_inputs(
        self,
        *,
        state: Any,
        observation: Any,
        action_chunk: Any,
        prepared_actions: torch.Tensor,
        instruction: Optional[str],
    ) -> None:
        if not self.config.debug or self._debug_logs_emitted >= self.config.debug_log_limit:
            return

        def _shape(value: Any) -> Any:
            try:
                return tuple(np.asarray(value).shape)
            except Exception:
                return getattr(value, "shape", type(value).__name__)

        obs_summary = type(observation).__name__
        if isinstance(observation, dict):
            if isinstance(observation.get("views"), dict):
                obs_summary = f"views={sorted(observation['views'].keys())}"
            else:
                obs_summary = f"obs_keys={sorted(observation.keys())}"

        state_summary = type(state).__name__
        if isinstance(state, dict):
            state_summary = f"state_keys={sorted(state.keys())}"
            if "robot" in state and isinstance(state["robot"], dict):
                state_summary += f" robot_keys={sorted(state['robot'].keys())}"
                if "state_representation" in state["robot"]:
                    state_summary += f" state_rep={state['robot']['state_representation']}"
            if "current_latent" in state:
                state_summary += f" current_latent={tuple(state['current_latent'].shape)}"
            if "history_latents" in state:
                state_summary += f" history_latents={tuple(state['history_latents'].shape)}"

        action_np = prepared_actions.detach().float().cpu().numpy()
        logger.info(
            "VidWM debug[%d]: flow_map_type=%s loss_type=%s num_inference_steps=%d num_frames=%d num_history=%d instruction=%r raw_action_shape=%s prepared_action_shape=%s prepared_action_min=%.4f prepared_action_max=%.4f observation=%s state=%s",
            self._debug_logs_emitted,
            self.config.flow_map_type,
            self.config.flow_map_loss_type,
            self.config.num_inference_steps,
            self.config.num_frames,
            self.config.num_history,
            instruction,
            _shape(action_chunk),
            tuple(prepared_actions.shape),
            float(action_np.min()),
            float(action_np.max()),
            obs_summary,
            state_summary,
        )
        self._debug_logs_emitted += 1

    # ------------------------------------------------------------------
    # Convenience: encode a raw RGB image into a latent
    # ------------------------------------------------------------------

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode an RGB image tensor to VAE latent space.

        Args:
            image: (B, 3, H, W) float tensor in [0, 1] or [-1, 1].

        Returns:
            (B, 4, H//8, W//8) latent tensor.
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load_checkpoint() first.")

        image = image.to(device=self._device, dtype=self.pipeline.vae.dtype)
        # The SVD VAE expects input in [-1, 1]. If the image is in [0, 1],
        # convert it first.
        if image.min() >= 0.0 and image.max() <= 1.0:
            image = image * 2.0 - 1.0
        latent = self.pipeline.vae.encode(image).latent_dist.mode()
        # Scale to match Ctrl-World training convention: latents are stored
        # multiplied by the VAE scaling factor.  The pipeline divides by this
        # factor when building the conditioning signal, so we must apply it
        # here to keep the round-trip correct.
        latent = latent * self.pipeline.vae.config.scaling_factor
        return latent

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode a single latent frame to RGB.

        Args:
            latent: (B, 4, H, W) latent tensor.

        Returns:
            (B, 3, H, W) float32 RGB tensor.
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load_checkpoint() first.")

        latent = latent.to(device=self._device, dtype=self.pipeline.vae.dtype)
        scaled = latent / self.pipeline.vae.config.scaling_factor
        decoded = self.pipeline.vae.decode(scaled).sample
        return decoded.float()
