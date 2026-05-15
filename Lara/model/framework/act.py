# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

import copy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from Lara.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, get_action_model


class ActionHeadAdapter(nn.Module):
    """Latent-action conditioned action head.

    Qwen emits two token streams for VLA-JEPA style robot fine-tuning:
    latent action tokens from the `{actions}` prompt slot, and embodied action
    tokens from the `{e_actions}` prompt slot. The latent tokens are trained by
    the V-JEPA world-model objective and are prepended here so the action head
    predicts follower-arm action chunks from the learned transition code plus
    the current follower state.
    """

    def __init__(self, config, context_hidden_size: int):
        super().__init__()
        self.config = copy.deepcopy(config)
        action_cfg = self.config.framework.action_model
        action_cfg.diffusion_model_cfg.cross_attention_dim = context_hidden_size
        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)
        self.future_action_window_size = action_cfg.future_action_window_size
        self.action_horizon = action_cfg.get("action_horizon", self.future_action_window_size + 1)
        self.use_latent_action_tokens = action_cfg.get("use_latent_action_tokens", True)
        self.max_latent_action_tokens = action_cfg.get("max_latent_action_tokens", None)
        self.repeated_diffusion_steps = action_cfg.get(
            "repeated_diffusion_steps",
            self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4,
        )
        self.condition_norm = nn.LayerNorm(context_hidden_size)
        self.latent_norm = nn.LayerNorm(context_hidden_size)
        self.latent_type_embed = nn.Parameter(torch.zeros(1, 1, context_hidden_size))
        self.embodied_type_embed = nn.Parameter(torch.zeros(1, 1, context_hidden_size))

    @staticmethod
    def _as_tensor(value, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=dtype)
        if isinstance(value, (list, tuple)) and value and all(isinstance(item, torch.Tensor) for item in value):
            return torch.stack([item.to(device=device, dtype=dtype) for item in value], dim=0)
        return torch.as_tensor(np.asarray(value), device=device, dtype=dtype)

    def _actions_to_tensor(self, actions, context_tokens: torch.Tensor) -> torch.Tensor:
        action_tensor = self._as_tensor(actions, device=context_tokens.device, dtype=torch.float32)
        if action_tensor.ndim != 3:
            raise ValueError(f"Expected actions with shape [B, T, D], got {tuple(action_tensor.shape)}")
        return action_tensor

    def _state_to_tensor(self, state, context_tokens: torch.Tensor) -> Optional[torch.Tensor]:
        if state is None:
            return None
        state_tensor = self._as_tensor(state, device=context_tokens.device, dtype=context_tokens.dtype)
        if state_tensor.ndim == 2:
            state_tensor = state_tensor.unsqueeze(1)
        if state_tensor.ndim != 3:
            raise ValueError(f"Expected state with shape [B, D] or [B, T, D], got {tuple(state_tensor.shape)}")
        return state_tensor

    def _conditioning_tokens(
        self,
        embodied_action_tokens: torch.Tensor,
        latent_action_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if embodied_action_tokens.ndim != 3:
            raise ValueError(
                "Expected embodied_action_tokens with shape [B, T, D], "
                f"got {tuple(embodied_action_tokens.shape)}"
            )
        embodied_action_tokens = self.condition_norm(embodied_action_tokens)
        embodied_action_tokens = embodied_action_tokens + self.embodied_type_embed.to(
            device=embodied_action_tokens.device,
            dtype=embodied_action_tokens.dtype,
        )
        if not self.use_latent_action_tokens or latent_action_tokens is None:
            return embodied_action_tokens

        if latent_action_tokens.ndim != 3:
            raise ValueError(
                "Expected latent_action_tokens with shape [B, T, D], "
                f"got {tuple(latent_action_tokens.shape)}"
            )
        if latent_action_tokens.shape[0] != embodied_action_tokens.shape[0]:
            raise ValueError(
                "latent_action_tokens batch size must match embodied_action_tokens: "
                f"{latent_action_tokens.shape[0]} != {embodied_action_tokens.shape[0]}"
            )
        if latent_action_tokens.shape[-1] != embodied_action_tokens.shape[-1]:
            raise ValueError(
                "latent_action_tokens hidden size must match embodied_action_tokens: "
                f"{latent_action_tokens.shape[-1]} != {embodied_action_tokens.shape[-1]}"
            )
        latent_action_tokens = latent_action_tokens.to(
            device=embodied_action_tokens.device,
            dtype=embodied_action_tokens.dtype,
        )
        if self.max_latent_action_tokens is not None:
            latent_action_tokens = latent_action_tokens[:, : self.max_latent_action_tokens, :]
        latent_action_tokens = self.latent_norm(latent_action_tokens)
        latent_action_tokens = latent_action_tokens + self.latent_type_embed.to(
            device=latent_action_tokens.device,
            dtype=latent_action_tokens.dtype,
        )
        return torch.cat([latent_action_tokens, embodied_action_tokens], dim=1)

    def forward(
        self,
        embodied_action_tokens: torch.Tensor,
        actions,
        state=None,
        latent_action_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        actions = self._actions_to_tensor(actions, embodied_action_tokens)
        if actions.shape[1] < self.action_horizon:
            raise ValueError(
                f"Expected at least {self.action_horizon} action steps, got {actions.shape[1]}"
            )
        actions_target = actions[:, -self.action_horizon :, :]
        actions_target_repeated = actions_target.repeat_interleave(self.repeated_diffusion_steps, dim=0)
        conditioning_tokens = self._conditioning_tokens(embodied_action_tokens, latent_action_tokens)
        context_repeated = conditioning_tokens.repeat_interleave(self.repeated_diffusion_steps, dim=0)

        state_tensor = self._state_to_tensor(state, embodied_action_tokens)
        state_repeated = (
            state_tensor.repeat_interleave(self.repeated_diffusion_steps, dim=0)
            if state_tensor is not None
            else None
        )

        return self.action_model(context_repeated, actions_target_repeated, state_repeated)

    @torch.no_grad()
    def predict_action(
        self,
        embodied_action_tokens: torch.Tensor,
        state: Optional[np.ndarray] = None,
        latent_action_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        conditioning_tokens = self._conditioning_tokens(embodied_action_tokens, latent_action_tokens)
        state_tensor = self._state_to_tensor(state, embodied_action_tokens)
        return self.action_model.predict_action(conditioning_tokens, state_tensor)
