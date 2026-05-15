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

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        batch_images = [example["image"] for example in examples]
        batch_videos = [example["video"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples] if "action" in examples[0] else None
        state = [example["state"] for example in examples] if "state" in examples[0] else None

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

        with torch.autocast("cuda", dtype=torch.bfloat16):
            wm_loss = self.vj2(batch_videos, qwen_context.action_tokens)

        if actions is None:
            return {"wm_loss": wm_loss}

        with torch.autocast("cuda", dtype=torch.float32):
            action_loss = self.action_head(
                embodied_action_tokens=qwen_context.embodied_action_tokens,
                latent_action_tokens=qwen_context.action_tokens,
                actions=actions,
                state=state,
            )

        return {"action_loss": action_loss, "wm_loss": wm_loss * 0.1}

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: List[List[Image.Image]],
        instructions: List[str],
        state: Optional[np.ndarray] = None,
        **kwargs: str,
    ) -> dict:
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        qwen_context = self.qwen.encode(
            images=batch_images,
            instructions=instructions,
            prompt_template=self.config.datasets.vla_data.get("CoT_prompt", ""),
            include_embodied_tokens=True,
        )

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_head.predict_action(
                embodied_action_tokens=qwen_context.embodied_action_tokens,
                latent_action_tokens=qwen_context.action_tokens,
                state=state,
            )
        pred_actions_np = pred_actions.detach().cpu().numpy()

        return {
            "normalized_actions": pred_actions_np,
            "execution_normalized_actions": pred_actions_np[:, : self.execution_horizon, :],
            "prediction_horizon": self.action_horizon,
            "execution_horizon": self.execution_horizon,
            "embodied_action_tokens": qwen_context.embodied_action_tokens.to(dtype=torch.float32)
            .detach()
            .cpu()
            .numpy(),
        }

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
