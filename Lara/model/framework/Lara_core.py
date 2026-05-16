# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from Lara.model.framework.act import ActionHeadAdapter
from Lara.model.framework.base_framework import baseframework
from Lara.model.framework.qwen import QwenActionTokenizer
from Lara.model.framework.vj2 import VJ2WorldModel
from Lara.model.tools import FRAMEWORK_REGISTRY
from Lara.training.trainer_utils.trainer_tools import resize_images


@FRAMEWORK_REGISTRY.register("Lara")
class Lara(baseframework):
    """Qwen3-VL + V-JEPA2 world model + swappable action head assembly."""

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.qwen = QwenActionTokenizer(config=self.config)
        self.vj2 = VJ2WorldModel(config=self.config, action_embed_dim=self.qwen.hidden_size)
        self.qwen.configure_world_tokens(num_prediction_steps=self.vj2.num_prediction_steps)
        self.action_head = ActionHeadAdapter(config=self.config, context_hidden_size=self.qwen.hidden_size)

        action_cfg = config.framework.action_model
        self.action_horizon = action_cfg.get("action_horizon", action_cfg.future_action_window_size + 1)
        self.execution_horizon = action_cfg.get("execution_horizon", self.action_horizon)
        self.future_action_window_size = action_cfg.future_action_window_size
        self.past_action_window_size = action_cfg.past_action_window_size
        self.chunk_len = self.action_horizon
        self.wm_loss_weight = self._loss_scale("wm", fallback_key="vlm", default=1.0)

    def _loss_scale(self, key: str, fallback_key: str | None = None, default: float = 1.0) -> float:
        loss_scale = getattr(getattr(self.config, "trainer", None), "loss_scale", None)
        if loss_scale is None:
            return default
        if key in loss_scale:
            return float(loss_scale.get(key))
        if fallback_key is not None and fallback_key in loss_scale:
            return float(loss_scale.get(fallback_key))
        return default

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        batch_images = [example["image"] for example in examples]
        batch_videos = [example["video"] for example in examples]
        instructions = [example["lang"] for example in examples]

        def optional_batch_field(name: str):
            return [example[name] for example in examples] if all(name in example for example in examples) else None

        action_key = "future_actions" if "future_actions" in examples[0] else "action"
        actions = [example[action_key] for example in examples] if action_key in examples[0] else None
        actions_are_future = action_key == "future_actions"
        action_mask_key = "future_action_mask" if "future_action_mask" in examples[0] else "action_mask"
        action_mask = optional_batch_field(action_mask_key)
        past_actions = optional_batch_field("past_actions")
        state_key = "current_state" if "current_state" in examples[0] else "state"
        state = [example[state_key] for example in examples] if state_key in examples[0] else None
        trajectory_ids = optional_batch_field("trajectory_id")
        execution_state_target = optional_batch_field("execution_state_target")
        prediction_state_target = optional_batch_field("prediction_state_target")
        execution_state_target_mask = (
            [example.get("execution_state_target_mask", True) for example in examples]
            if execution_state_target is not None
            else None
        )
        prediction_state_target_mask = (
            [example.get("prediction_state_target_mask", True) for example in examples]
            if prediction_state_target is not None
            else None
        )
        utility_scores = optional_batch_field("utility_scores")
        utility_candidate_mask = optional_batch_field("utility_candidate_mask")
        utility_cost_scores = optional_batch_field("utility_cost_scores")
        utility_value_targets = optional_batch_field("utility_value_targets")
        utility_progress_targets = optional_batch_field("utility_progress_targets")
        utility_uncertainty_targets = optional_batch_field("utility_uncertainty_targets")
        utility_target_mask = optional_batch_field("utility_target_mask")
        previous_router_probs = optional_batch_field("previous_router_probs")
        pool_mask = optional_batch_field("pool_mask")
        episode_start_images = optional_batch_field("episode_start_image")

        if actions is not None:
            prompt_template = self.config.datasets.vla_data.get("CoT_prompt", "")
        else:
            prompt_template = self.config.datasets.video_data.get("CoT_prompt", "")

        qwen_context = self.qwen.encode(
            images=batch_images,
            instructions=instructions,
            prompt_template=prompt_template,
            include_embodied_tokens=actions is not None,
        )

        initial_context_tokens = None
        if actions is not None and self.action_head.lara_moe is not None and episode_start_images is not None:
            # The episode-start context conditions the resident pool router; it
            # is not an action reconstruction target, so keep this second Qwen
            # encode stop-gradient to avoid doubling the training activation graph.
            with torch.no_grad():
                episode_start_context = self.qwen.encode(
                    images=episode_start_images,
                    instructions=instructions,
                    prompt_template=prompt_template,
                    include_embodied_tokens=True,
                )
            initial_context_tokens = self.action_head.conditioning_tokens(
                embodied_action_tokens=episode_start_context.embodied_action_tokens,
                latent_action_tokens=episode_start_context.action_tokens,
            ).detach()
            del episode_start_context

        with torch.autocast("cuda", dtype=torch.bfloat16):
            wm_loss = self.vj2(batch_videos, qwen_context.action_tokens)
        wm_loss_weighted = wm_loss * self.wm_loss_weight

        if actions is None:
            return {
                "wm_loss": wm_loss_weighted,
                "metric/wm_loss_raw": wm_loss.detach(),
                "metric/wm_loss_weight": torch.as_tensor(self.wm_loss_weight, device=wm_loss.device),
            }

        with torch.autocast("cuda", dtype=torch.float32):
            action_output = self.action_head(
                embodied_action_tokens=qwen_context.embodied_action_tokens,
                latent_action_tokens=qwen_context.action_tokens,
                actions=actions,
                actions_are_future=actions_are_future,
                action_mask=action_mask,
                past_actions=past_actions,
                state=state,
                trajectory_ids=trajectory_ids,
                initial_context_tokens=initial_context_tokens,
                pool_mask=pool_mask,
                utility_scores=utility_scores,
                utility_candidate_mask=utility_candidate_mask,
                utility_cost_scores=utility_cost_scores,
                utility_value_targets=utility_value_targets,
                utility_progress_targets=utility_progress_targets,
                utility_uncertainty_targets=utility_uncertainty_targets,
                utility_target_mask=utility_target_mask,
                execution_state_target=execution_state_target,
                prediction_state_target=prediction_state_target,
                execution_state_target_mask=execution_state_target_mask,
                prediction_state_target_mask=prediction_state_target_mask,
                previous_router_probs=previous_router_probs,
                return_aux=True,
            )

        if isinstance(action_output, dict):
            output = {
                "action_loss": action_output["total_action_loss"],
                "wm_loss": wm_loss_weighted,
                "metric/wm_loss_raw": wm_loss.detach(),
                "metric/wm_loss_weight": torch.as_tensor(self.wm_loss_weight, device=wm_loss.device),
            }
            for key, value in action_output.items():
                if key == "total_action_loss":
                    continue
                if torch.is_tensor(value) and value.numel() == 1:
                    output[f"metric/{key}"] = value.detach()
            return output

        return {
            "action_loss": action_output,
            "wm_loss": wm_loss_weighted,
            "metric/wm_loss_raw": wm_loss.detach(),
            "metric/wm_loss_weight": torch.as_tensor(self.wm_loss_weight, device=wm_loss.device),
        }

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: List[List[Image.Image]],
        instructions: List[str],
        state: Optional[np.ndarray] = None,
        **kwargs,
    ) -> dict:
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
        batch_size = len(instructions)

        qwen_context = self.qwen.encode(
            images=batch_images,
            instructions=instructions,
            prompt_template=self.config.datasets.vla_data.get("CoT_prompt", ""),
            include_embodied_tokens=True,
        )
        resident_pool_mask = kwargs.get("resident_pool_mask", None)
        previous_router_probs = kwargs.get("previous_router_probs", None)
        forced_router_probs = kwargs.get("forced_router_probs", None)
        forced_expert_id = kwargs.get("forced_expert_id", None)
        if forced_router_probs is not None and self.action_head.lara_moe is None:
            raise ValueError("forced_router_probs requires use_lara_moe=True")
        if forced_expert_id is not None:
            if forced_router_probs is not None:
                raise ValueError("Pass only one of forced_expert_id or forced_router_probs")
            if self.action_head.lara_moe is None:
                raise ValueError("forced_expert_id requires use_lara_moe=True")
            expert_ids = np.asarray(forced_expert_id)
            if expert_ids.ndim == 0:
                expert_ids = np.full(batch_size, int(expert_ids), dtype=np.int64)
            elif expert_ids.shape != (batch_size,):
                raise ValueError(f"forced_expert_id must be scalar or shape ({batch_size},), got {expert_ids.shape}")
            num_experts = self.action_head.lara_moe.num_experts
            if np.any(expert_ids < 0) or np.any(expert_ids >= num_experts):
                raise ValueError(f"forced_expert_id must be in [0, {num_experts})")
            forced_router_probs = np.zeros((batch_size, num_experts), dtype=np.float32)
            forced_router_probs[np.arange(batch_size), expert_ids.astype(np.int64)] = 1.0
        if forced_router_probs is not None and resident_pool_mask is None:
            resident_pool_mask = np.asarray(forced_router_probs) > 0
        resident_pool = None

        with torch.autocast("cuda", dtype=torch.float32):
            if self.action_head.lara_moe is not None and resident_pool_mask is None:
                resident_pool = self.action_head.select_resident_pool(
                    embodied_action_tokens=qwen_context.embodied_action_tokens,
                    latent_action_tokens=qwen_context.action_tokens,
                )
                resident_pool_mask = resident_pool.mask
            action_result = self.action_head.predict_action(
                embodied_action_tokens=qwen_context.embodied_action_tokens,
                latent_action_tokens=qwen_context.action_tokens,
                state=state,
                pool_mask=resident_pool_mask,
                previous_router_probs=previous_router_probs,
                forced_router_probs=forced_router_probs,
                return_aux=True,
            )
        if isinstance(action_result, dict):
            pred_actions = action_result["actions"]
            router_probs = action_result.get("router_probs")
            active_mask = action_result.get("active_mask")
        else:
            pred_actions = action_result
            router_probs = None
            active_mask = None
        pred_actions_np = pred_actions.detach().cpu().numpy()

        output = {
            "normalized_actions": pred_actions_np,
            "execution_normalized_actions": pred_actions_np[:, : self.execution_horizon, :],
            "prediction_horizon": self.action_horizon,
            "execution_horizon": self.execution_horizon,
            "embodied_action_tokens": qwen_context.embodied_action_tokens.to(dtype=torch.float32)
            .detach()
            .cpu()
            .numpy(),
        }
        if resident_pool_mask is not None:
            resident_pool_tensor = (
                resident_pool_mask
                if isinstance(resident_pool_mask, torch.Tensor)
                else torch.as_tensor(resident_pool_mask)
            )
            output["resident_pool_mask"] = resident_pool_tensor.detach().cpu().numpy()
        if resident_pool is not None:
            output["resident_pool_probs"] = resident_pool.probs.detach().cpu().numpy()
        if router_probs is not None:
            output["router_probs"] = router_probs.detach().cpu().numpy()
        if active_mask is not None:
            output["active_expert_mask"] = active_mask.detach().cpu().numpy()
        if forced_router_probs is not None:
            output["forced_router_probs"] = (
                forced_router_probs.detach().cpu().numpy()
                if torch.is_tensor(forced_router_probs)
                else np.asarray(forced_router_probs)
            )
        if forced_expert_id is not None:
            output["forced_expert_id"] = expert_ids
        return output

    @property
    def qwen_vl_interface(self):
        return self.qwen.qwen_vl_interface

    @property
    def vj_encoder(self):
        return self.vj2.vj_encoder

    @property
    def vj_processor(self):
        return self.vj2.vj_processor

    @property
    def vj_predictor(self):
        return self.vj2.vj_predictor

    @property
    def action_model(self):
        return self.action_head
