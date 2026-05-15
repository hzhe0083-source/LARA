# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

import copy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from Lara.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, get_action_model
from Lara.model.modules.action_model.lara_latent import LatentActionHead
from Lara.model.modules.action_model.lara_moe import LatentActionMoE


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
        self.use_latent_action_head = action_cfg.get("use_latent_action_head", False)
        self.use_lara_moe = action_cfg.get("use_lara_moe", False)
        self.use_expert_loss_posterior = action_cfg.get("lara_use_expert_loss_posterior", True)
        self.repeated_diffusion_steps = action_cfg.get(
            "repeated_diffusion_steps",
            self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4,
        )
        self.condition_norm = nn.LayerNorm(context_hidden_size)
        self.latent_norm = nn.LayerNorm(context_hidden_size)
        self.latent_type_embed = nn.Parameter(torch.zeros(1, 1, context_hidden_size))
        self.embodied_type_embed = nn.Parameter(torch.zeros(1, 1, context_hidden_size))
        self.latent_action_head = (
            LatentActionHead(
                context_dim=context_hidden_size,
                action_dim=action_cfg.action_dim,
                action_horizon=self.action_horizon,
                num_latent_tokens=action_cfg.get("lara_num_latent_tokens", 4),
                codebook_size=action_cfg.get("lara_codebook_size", 128),
                hidden_dim=action_cfg.get("lara_latent_hidden_dim", action_cfg.get("hidden_size", context_hidden_size)),
                commitment_weight=action_cfg.get("lara_commitment_weight", 0.25),
                vq_loss_weight=action_cfg.get("lara_vq_loss_weight", 1.0),
                prior_loss_weight=action_cfg.get("lara_prior_loss_weight", 1.0),
            )
            if self.use_latent_action_head
            else None
        )
        self.lara_moe = (
            LatentActionMoE(
                hidden_size=context_hidden_size,
                num_experts=action_cfg.get("lara_num_experts", 8),
                top_k=action_cfg.get("lara_top_k", 2),
                episode_pool_size=action_cfg.get("lara_episode_pool_size", None),
                expert_hidden_size=action_cfg.get("lara_expert_hidden_dim", action_cfg.get("hidden_size", 1024)),
                router_hidden_size=action_cfg.get("lara_router_hidden_dim", action_cfg.get("hidden_size", 1024)),
                router_loss_weight=action_cfg.get("lara_router_loss_weight", 1.0),
                pool_loss_weight=action_cfg.get("lara_pool_loss_weight", 1.0),
                posterior_temperature=action_cfg.get("lara_posterior_temperature", 1.0),
                residual_scale=action_cfg.get("lara_expert_residual_scale", 0.1),
            )
            if self.use_lara_moe
            else None
        )

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
        return_aux: bool = False,
    ):
        actions = self._actions_to_tensor(actions, embodied_action_tokens)
        if actions.shape[1] < self.action_horizon:
            raise ValueError(
                f"Expected at least {self.action_horizon} action steps, got {actions.shape[1]}"
            )
        # Prefer passing an explicit future-only action window from the dataloader
        # (`future_actions`). This fallback keeps older batches working.
        actions_target = actions[:, -self.action_horizon :, :]
        aux_losses = {}
        if self.latent_action_head is not None:
            latent_output = self.latent_action_head(embodied_action_tokens, actions_target)
            latent_action_tokens = latent_output.tokens
            aux_losses = {
                "latent_action_loss": latent_output.loss,
                "latent_action_vq_loss": latent_output.vq_loss,
                "latent_action_prior_loss": latent_output.prior_loss,
                "latent_action_perplexity": latent_output.perplexity,
            }
        actions_target_repeated = actions_target.repeat_interleave(self.repeated_diffusion_steps, dim=0)
        conditioning_tokens = self._conditioning_tokens(embodied_action_tokens, latent_action_tokens)
        if self.lara_moe is not None:
            with torch.no_grad():
                expert_action_losses = (
                    self._expert_action_losses(conditioning_tokens, actions_target, state)
                    if self.use_expert_loss_posterior
                    else None
                )
            moe_output = self.lara_moe(
                conditioning_tokens,
                latent_action_tokens=latent_action_tokens,
                expert_action_losses=expert_action_losses,
            )
            conditioning_tokens = moe_output.tokens
            aux_losses.update(
                {
                    "moe_router_loss": moe_output.loss,
                    "moe_route_distill_loss": moe_output.route_loss,
                    "moe_pool_distill_loss": moe_output.pool_loss,
                    "moe_router_entropy": moe_output.router_entropy,
                    "moe_posterior_entropy": moe_output.posterior_entropy,
                    "moe_pool_entropy": moe_output.pool_entropy,
                }
            )
        context_repeated = conditioning_tokens.repeat_interleave(self.repeated_diffusion_steps, dim=0)

        state_tensor = self._state_to_tensor(state, embodied_action_tokens)
        state_repeated = (
            state_tensor.repeat_interleave(self.repeated_diffusion_steps, dim=0)
            if state_tensor is not None
            else None
        )

        action_loss = self.action_model(context_repeated, actions_target_repeated, state_repeated)
        if not return_aux:
            return (
                action_loss
                + aux_losses.get("latent_action_loss", 0.0)
                + aux_losses.get("moe_router_loss", 0.0)
            )
        aux_losses["action_loss"] = action_loss
        aux_losses["total_action_loss"] = (
            action_loss
            + aux_losses.get("latent_action_loss", 0.0)
            + aux_losses.get("moe_router_loss", 0.0)
        )
        return aux_losses

    def _expert_action_losses(self, conditioning_tokens, actions_target, state=None) -> torch.Tensor:
        if self.lara_moe is None:
            raise RuntimeError("_expert_action_losses requires lara_moe")
        expert_tokens = self.lara_moe.expert_conditioning_tokens(conditioning_tokens)
        batch_size, num_experts, token_count, hidden_size = expert_tokens.shape
        flat_tokens = expert_tokens.reshape(batch_size * num_experts, token_count, hidden_size)
        flat_actions = actions_target[:, None, :, :].expand(-1, num_experts, -1, -1)
        flat_actions = flat_actions.reshape(batch_size * num_experts, actions_target.shape[1], actions_target.shape[2])

        noise = torch.randn(actions_target.shape, device=actions_target.device, dtype=actions_target.dtype)
        t = self.action_model.sample_time(actions_target.shape[0], device=actions_target.device, dtype=actions_target.dtype)
        noise = noise[:, None, :, :].expand(-1, num_experts, -1, -1).reshape_as(flat_actions)
        t = t[:, None].expand(-1, num_experts).reshape(batch_size * num_experts)

        state_tensor = self._state_to_tensor(state, conditioning_tokens)
        flat_state = None
        if state_tensor is not None:
            flat_state = state_tensor[:, None, :, :].expand(-1, num_experts, -1, -1)
            flat_state = flat_state.reshape(batch_size * num_experts, state_tensor.shape[1], state_tensor.shape[2])

        losses = self.action_model(
            flat_tokens,
            flat_actions,
            flat_state,
            noise=noise,
            t=t,
            reduction="none",
        )
        return losses.view(batch_size, num_experts)

    @torch.no_grad()
    def predict_action(
        self,
        embodied_action_tokens: torch.Tensor,
        state: Optional[np.ndarray] = None,
        latent_action_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.latent_action_head is not None:
            latent_action_tokens = self.latent_action_head.predict(embodied_action_tokens)
        conditioning_tokens = self._conditioning_tokens(embodied_action_tokens, latent_action_tokens)
        if self.lara_moe is not None:
            conditioning_tokens = self.lara_moe.predict(conditioning_tokens)
        state_tensor = self._state_to_tensor(state, embodied_action_tokens)
        return self.action_model.predict_action(conditioning_tokens, state_tensor)
