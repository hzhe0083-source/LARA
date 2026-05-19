# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

import copy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Lara.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, get_action_model
from Lara.model.modules.action_model.lara_latent import LatentActionHead, LatentActionTransitionHead
from Lara.model.modules.action_model.lara_moe import (
    ActionChunkExpertBank,
    LatentActionMoE,
    aggregate_episode_responsibilities,
    expert_diversity_loss,
    posterior_from_expert_losses,
    route_quality_metrics,
    utility_component_targets_from_expert_losses,
    utility_from_expert_losses,
)


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
        self.execution_horizon = action_cfg.get("execution_horizon", self.action_horizon)
        self.latent_action_horizon = action_cfg.get(
            "latent_action_horizon",
            self.execution_horizon,
        )
        self.router_horizon = action_cfg.get("router_horizon", self.execution_horizon)
        self.utility_horizon = action_cfg.get("utility_horizon", self.execution_horizon)
        for horizon_name, horizon_value in [
            ("execution_horizon", self.execution_horizon),
            ("latent_action_horizon", self.latent_action_horizon),
            ("router_horizon", self.router_horizon),
            ("utility_horizon", self.utility_horizon),
        ]:
            if horizon_value <= 0 or horizon_value > self.action_horizon:
                raise ValueError(f"{horizon_name} must be in [1, action_horizon]")
        self.use_latent_action_tokens = action_cfg.get("use_latent_action_tokens", True)
        self.max_latent_action_tokens = action_cfg.get("max_latent_action_tokens", None)
        self.use_latent_action_head = action_cfg.get("use_latent_action_head", True)
        self.latent_reconstruction_loss_weight = action_cfg.get("lara_latent_reconstruction_loss_weight", 1.0)
        self.use_transition_head = action_cfg.get("lara_use_transition_head", True)
        self.transition_loss_weight = action_cfg.get("lara_transition_loss_weight", 1.0)
        self.execution_transition_loss_weight = action_cfg.get("lara_execution_transition_loss_weight", 1.0)
        self.prediction_transition_loss_weight = action_cfg.get("lara_prediction_transition_loss_weight", 1.0)
        self.use_lara_moe = action_cfg.get("use_lara_moe", True)
        self.use_expert_loss_posterior = action_cfg.get("lara_use_expert_loss_posterior", True)
        self.use_action_loss_utility = action_cfg.get("lara_use_action_loss_utility", True)
        self.use_action_loss_utility_components = action_cfg.get("lara_use_action_loss_utility_components", True)
        self.action_loss_utility_temperature = action_cfg.get("lara_action_loss_utility_temperature", 1.0)
        self.action_loss_utility_normalize = action_cfg.get("lara_action_loss_utility_normalize", True)
        self.use_state_utility = action_cfg.get("lara_use_state_utility", True)
        self.use_state_utility_components = action_cfg.get("lara_use_state_utility_components", True)
        self.state_utility_temperature = action_cfg.get(
            "lara_state_utility_temperature",
            self.action_loss_utility_temperature,
        )
        self.state_utility_normalize = action_cfg.get(
            "lara_state_utility_normalize",
            self.action_loss_utility_normalize,
        )
        self.pool_critical_threshold = action_cfg.get("lara_pool_critical_threshold", 0.0)
        route_retention_fractions = action_cfg.get("lara_route_retention_fractions", [0.25, 0.5, 1.0])
        self.route_retention_fractions = (
            list(route_retention_fractions) if route_retention_fractions is not None else None
        )
        self.use_direct_action_experts = action_cfg.get("lara_use_direct_action_experts", self.use_lara_moe)
        self.use_direct_action_output = action_cfg.get(
            "lara_use_direct_action_output",
            self.use_lara_moe and self.use_direct_action_experts,
        )
        self.direct_expert_loss_weight = action_cfg.get("lara_direct_expert_loss_weight", 1.0)
        direct_expert_action_mode = action_cfg.get("lara_direct_expert_action_mode", None)
        if direct_expert_action_mode is None:
            direct_expert_action_mode = (
                "residual" if action_cfg.get("lara_use_direct_action_residual", False) else "full"
            )
        if direct_expert_action_mode not in {"full", "residual"}:
            raise ValueError("lara_direct_expert_action_mode must be 'full' or 'residual'")
        self.direct_expert_action_mode = direct_expert_action_mode
        self.direct_expert_residual_scale = action_cfg.get("lara_direct_expert_residual_scale", 1.0)
        self.direct_expert_residual_max_norm = action_cfg.get("lara_direct_expert_residual_max_norm", None)
        if self.direct_expert_residual_max_norm is not None and self.direct_expert_residual_max_norm <= 0:
            raise ValueError("lara_direct_expert_residual_max_norm must be positive when set")
        self.direct_expert_residual_warmup_steps = int(
            action_cfg.get("lara_direct_expert_residual_warmup_steps", 0) or 0
        )
        if self.direct_expert_residual_warmup_steps < 0:
            raise ValueError("lara_direct_expert_residual_warmup_steps must be non-negative")
        self.direct_expert_residual_cost_weight = action_cfg.get(
            "lara_direct_expert_residual_cost_weight",
            0.0,
        )
        self.direct_expert_improvement_posterior = action_cfg.get(
            "lara_direct_expert_improvement_posterior",
            self.direct_expert_action_mode == "residual",
        )
        self.direct_expert_hard_assignment = action_cfg.get(
            "lara_direct_expert_hard_assignment",
            False,
        )
        self.direct_expert_posterior_top_r = action_cfg.get(
            "lara_direct_expert_posterior_top_r",
            None,
        )
        if self.direct_expert_posterior_top_r is not None and self.direct_expert_posterior_top_r <= 0:
            raise ValueError("lara_direct_expert_posterior_top_r must be positive")
        self.direct_expert_improvement_margin = action_cfg.get(
            "lara_direct_expert_improvement_margin",
            0.0,
        )
        self.direct_expert_shared_only_gate = action_cfg.get(
            "lara_direct_expert_shared_only_gate",
            False,
        )
        self.direct_residual_norm_loss_weight = action_cfg.get("lara_direct_residual_norm_loss_weight", 0.0)
        self.direct_residual_diversity_loss_weight = action_cfg.get(
            "lara_direct_residual_diversity_loss_weight",
            0.0,
        )
        self.shared_action_deterministic_baseline = action_cfg.get(
            "lara_shared_action_deterministic_baseline",
            True,
        )
        if self.use_direct_action_output and (not self.use_lara_moe or not self.use_direct_action_experts):
            raise ValueError(
                "lara_use_direct_action_output requires use_lara_moe=True "
                "and lara_use_direct_action_experts=True"
            )
        self.repeated_diffusion_steps = action_cfg.get(
            "repeated_diffusion_steps",
            self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4,
        )
        self.register_buffer(
            "_direct_expert_train_steps",
            torch.zeros((), dtype=torch.long),
            persistent=False,
        )
        self.condition_norm = nn.LayerNorm(context_hidden_size)
        self.latent_norm = nn.LayerNorm(context_hidden_size)
        self.latent_type_embed = nn.Parameter(torch.zeros(1, 1, context_hidden_size))
        self.embodied_type_embed = nn.Parameter(torch.zeros(1, 1, context_hidden_size))
        self.latent_action_head = (
            LatentActionHead(
                context_dim=context_hidden_size,
                action_dim=action_cfg.action_dim,
                action_horizon=self.latent_action_horizon,
                num_latent_tokens=action_cfg.get("lara_num_latent_tokens", 4),
                codebook_size=action_cfg.get("lara_codebook_size", 128),
                hidden_dim=action_cfg.get("lara_latent_hidden_dim", action_cfg.get("hidden_size", context_hidden_size)),
                commitment_weight=action_cfg.get("lara_commitment_weight", 0.25),
                vq_loss_weight=action_cfg.get("lara_vq_loss_weight", 1.0),
                prior_loss_weight=action_cfg.get("lara_prior_loss_weight", 1.0),
                code_usage_loss_weight=action_cfg.get("lara_code_usage_loss_weight", 0.0),
                code_usage_temperature=action_cfg.get("lara_code_usage_temperature", 1.0),
                reconstruction_loss_weight=self.latent_reconstruction_loss_weight,
            )
            if self.use_latent_action_head
            else None
        )
        self.transition_head = (
            LatentActionTransitionHead(
                context_dim=context_hidden_size,
                state_dim=action_cfg.state_dim,
                hidden_dim=action_cfg.get("lara_transition_hidden_dim", action_cfg.get("hidden_size", context_hidden_size)),
                num_boundaries=2,
            )
            if self.use_transition_head
            else None
        )
        self.lara_moe = (
            LatentActionMoE(
                hidden_size=context_hidden_size,
                num_experts=action_cfg.get("lara_num_experts", 8),
                top_k=action_cfg.get("lara_top_k", 2),
                episode_pool_size=action_cfg.get("lara_episode_pool_size", None),
                episode_pool_size_min=action_cfg.get("lara_episode_pool_size_min", None),
                expert_hidden_size=action_cfg.get("lara_expert_hidden_dim", action_cfg.get("hidden_size", 1024)),
                router_hidden_size=action_cfg.get("lara_router_hidden_dim", action_cfg.get("hidden_size", 1024)),
                router_loss_weight=action_cfg.get("lara_router_loss_weight", 1.0),
                pool_loss_weight=action_cfg.get("lara_pool_loss_weight", 1.0),
                pool_coverage_loss_weight=action_cfg.get("lara_pool_coverage_loss_weight", 0.25),
                utility_loss_weight=action_cfg.get("lara_utility_loss_weight", 1.0),
                utility_rank_loss_weight=action_cfg.get("lara_utility_rank_loss_weight", 0.25),
                utility_head_loss_weight=action_cfg.get("lara_utility_head_loss_weight", 1.0),
                balance_loss_weight=action_cfg.get("lara_balance_loss_weight", 0.0),
                stickiness_loss_weight=action_cfg.get("lara_stickiness_loss_weight", 0.0),
                diversity_loss_weight=action_cfg.get("lara_diversity_loss_weight", 0.0),
                entropy_loss_weight=action_cfg.get("lara_entropy_loss_weight", 0.0),
                use_utility_head=action_cfg.get("lara_use_utility_head", True),
                utility_hidden_size=action_cfg.get("lara_utility_hidden_dim", action_cfg.get("hidden_size", 1024)),
                utility_progress_weight=action_cfg.get("lara_utility_progress_weight", 1.0),
                utility_uncertainty_weight=action_cfg.get("lara_utility_uncertainty_weight", 1.0),
                utility_cost_weight=action_cfg.get("lara_utility_cost_weight", 1.0),
                posterior_temperature=action_cfg.get("lara_posterior_temperature", 1.0),
                posterior_uniform_floor=action_cfg.get("lara_posterior_uniform_floor", 0.0),
                posterior_top_r=action_cfg.get("lara_posterior_top_r", None),
                pool_critical_threshold=action_cfg.get("lara_pool_critical_threshold", 0.0),
                inference_stickiness_weight=action_cfg.get("lara_inference_stickiness_weight", 0.0),
                residual_scale=action_cfg.get("lara_expert_residual_scale", 0.1),
            )
            if self.use_lara_moe
            else None
        )
        self.direct_action_experts = (
            ActionChunkExpertBank(
                hidden_size=context_hidden_size,
                num_experts=action_cfg.get("lara_num_experts", 8),
                expert_hidden_size=action_cfg.get("lara_direct_expert_hidden_dim", action_cfg.get("hidden_size", 1024)),
                action_horizon=self.action_horizon,
                action_dim=action_cfg.action_dim,
                state_dim=action_cfg.get("state_dim", None),
            )
            if self.use_lara_moe and self.use_direct_action_experts
            else None
        )

    def _shared_action_prediction(
        self,
        conditioning_tokens: torch.Tensor,
        state_tensor: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run the dense action head as a frozen shared baseline for residual experts."""
        was_training = self.action_model.training
        self.action_model.eval()
        initial_actions = None
        if self.shared_action_deterministic_baseline:
            initial_actions = torch.zeros(
                conditioning_tokens.shape[0],
                self.action_horizon,
                self.config.framework.action_model.action_dim,
                device=conditioning_tokens.device,
                dtype=conditioning_tokens.dtype,
            )
        try:
            shared_actions = self.action_model.predict_action(
                conditioning_tokens.detach(),
                state_tensor.detach() if state_tensor is not None else None,
                initial_actions=initial_actions,
            )
        finally:
            if was_training:
                self.action_model.train()
        return shared_actions.detach()

    def _direct_expert_action_bank(
        self,
        conditioning_tokens: torch.Tensor,
        state_tensor: Optional[torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        if self.direct_action_experts is None:
            raise RuntimeError("_direct_expert_action_bank requires direct_action_experts")
        raw_expert_actions = self.direct_action_experts(conditioning_tokens, state=state_tensor)
        if self.direct_expert_action_mode == "full":
            return raw_expert_actions, None, None, {}
        shared_actions = self._shared_action_prediction(conditioning_tokens, state_tensor)
        residual_info = self._direct_expert_residual_info(
            raw_expert_actions.device,
            raw_expert_actions.dtype,
        )
        residual_actions_pre_clamp = residual_info["effective_scale"] * raw_expert_actions
        residual_actions = self._clamp_direct_expert_residuals(residual_actions_pre_clamp)
        residual_info["pre_clamp_residual_actions"] = residual_actions_pre_clamp
        residual_info["raw_residual_norm"] = self._expert_residual_norm(residual_actions_pre_clamp)
        if self.direct_expert_residual_max_norm is None:
            residual_info["residual_clamp_rate"] = torch.zeros_like(residual_info["raw_residual_norm"])
        else:
            residual_info["residual_clamp_rate"] = (
                residual_info["raw_residual_norm"] > self.direct_expert_residual_max_norm
            ).float()
        expert_actions = shared_actions[:, None, :, :] + residual_actions
        return expert_actions, residual_actions, shared_actions, residual_info

    @staticmethod
    def _expert_residual_norm(residual_actions: torch.Tensor) -> torch.Tensor:
        return residual_actions.pow(2).mean(dim=(-1, -2)).sqrt()

    def _direct_expert_residual_info(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        if self.training:
            self._direct_expert_train_steps.add_(1)
        base_scale = torch.as_tensor(
            self.direct_expert_residual_scale,
            device=device,
            dtype=dtype,
        )
        warmup_fraction = torch.ones((), device=device, dtype=dtype)
        if self.training and self.direct_expert_residual_warmup_steps > 0:
            step = self._direct_expert_train_steps.to(device=device, dtype=dtype)
            warmup_fraction = (
                step / float(self.direct_expert_residual_warmup_steps)
            ).clamp(max=1.0)
        return {
            "base_scale": base_scale,
            "effective_scale": base_scale * warmup_fraction,
            "warmup_fraction": warmup_fraction,
        }

    def _clamp_direct_expert_residuals(self, residual_actions: torch.Tensor) -> torch.Tensor:
        if self.direct_expert_residual_max_norm is None:
            return residual_actions
        residual_norm = self._expert_residual_norm(residual_actions)
        max_norm = torch.as_tensor(
            self.direct_expert_residual_max_norm,
            device=residual_actions.device,
            dtype=residual_actions.dtype,
        )
        clamp_scale = (
            max_norm / residual_norm.clamp_min(torch.finfo(residual_actions.dtype).eps)
        ).clamp(max=1.0)
        return residual_actions * clamp_scale[:, :, None, None]

    @staticmethod
    def _as_tensor(value, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=dtype)
        if isinstance(value, (list, tuple)) and value and all(isinstance(item, torch.Tensor) for item in value):
            return torch.stack([item.to(device=device, dtype=dtype) for item in value], dim=0)
        return torch.as_tensor(np.asarray(value), device=device, dtype=dtype)

    @staticmethod
    def _trajectory_ids_to_tensor(value, device: torch.device) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=torch.long).view(-1)

        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.number):
            return torch.as_tensor(array, device=device, dtype=torch.long).view(-1)

        encoded = []
        id_to_index = {}
        for item in array.reshape(-1).tolist():
            key = item.item() if isinstance(item, np.generic) else item
            try:
                hash(key)
            except TypeError:
                key = repr(key)
            if key not in id_to_index:
                id_to_index[key] = len(id_to_index)
            encoded.append(id_to_index[key])
        return torch.tensor(encoded, device=device, dtype=torch.long)

    def _actions_to_tensor(self, actions, context_tokens: torch.Tensor) -> torch.Tensor:
        action_tensor = self._as_tensor(actions, device=context_tokens.device, dtype=torch.float32)
        if action_tensor.ndim != 3:
            raise ValueError(f"Expected actions with shape [B, T, D], got {tuple(action_tensor.shape)}")
        return action_tensor

    def _action_mask_to_tensor(
        self,
        action_mask,
        actions: torch.Tensor,
        actions_are_future: bool,
    ) -> Optional[torch.Tensor]:
        if action_mask is None:
            return None
        mask_tensor = self._as_tensor(action_mask, device=actions.device, dtype=torch.bool)
        if mask_tensor.ndim == 3 and mask_tensor.shape[-1] == 1:
            mask_tensor = mask_tensor.squeeze(-1)
        if mask_tensor.ndim != 2 or mask_tensor.shape[0] != actions.shape[0]:
            raise ValueError(
                f"Expected action_mask with shape [B, T], got {tuple(mask_tensor.shape)} "
                f"for actions {tuple(actions.shape)}"
            )
        if actions_are_future:
            if mask_tensor.shape[1] != self.action_horizon:
                raise ValueError(
                    f"future_action_mask must have exactly {self.action_horizon} steps, got {mask_tensor.shape[1]}"
                )
            return mask_tensor
        if mask_tensor.shape[1] < self.action_horizon:
            raise ValueError(f"Expected action_mask with at least {self.action_horizon} steps, got {mask_tensor.shape[1]}")
        return mask_tensor[:, -self.action_horizon :]

    def _state_to_tensor(self, state, context_tokens: torch.Tensor) -> Optional[torch.Tensor]:
        if state is None:
            return None
        state_tensor = self._as_tensor(state, device=context_tokens.device, dtype=context_tokens.dtype)
        if state_tensor.ndim == 2:
            state_tensor = state_tensor.unsqueeze(1)
        if state_tensor.ndim != 3:
            raise ValueError(f"Expected state with shape [B, D] or [B, T, D], got {tuple(state_tensor.shape)}")
        return state_tensor

    def _boundary_state_to_tensor(self, state, context_tokens: torch.Tensor) -> Optional[torch.Tensor]:
        if state is None:
            return None
        state_tensor = self._as_tensor(state, device=context_tokens.device, dtype=torch.float32)
        if state_tensor.ndim == 3 and state_tensor.shape[1] == 1:
            state_tensor = state_tensor[:, 0, :]
        if state_tensor.ndim != 2:
            raise ValueError(f"Expected boundary state with shape [B, D] or [B, 1, D], got {tuple(state_tensor.shape)}")
        return state_tensor

    def _boundary_mask_to_tensor(
        self,
        mask,
        context_tokens: torch.Tensor,
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        mask_tensor = self._as_tensor(mask, device=context_tokens.device, dtype=torch.bool).view(-1)
        if mask_tensor.shape[0] != batch_size:
            raise ValueError(f"Expected boundary mask with B={batch_size}, got {tuple(mask_tensor.shape)}")
        return mask_tensor

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

    def conditioning_tokens(
        self,
        embodied_action_tokens: torch.Tensor,
        latent_action_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target_dtype = self.condition_norm.weight.dtype
        embodied_action_tokens = embodied_action_tokens.to(dtype=target_dtype)
        if latent_action_tokens is not None:
            latent_action_tokens = latent_action_tokens.to(dtype=target_dtype)
        return self._conditioning_tokens(embodied_action_tokens, latent_action_tokens)

    def forward(
        self,
        embodied_action_tokens: torch.Tensor,
        actions,
        actions_are_future: bool = False,
        action_mask=None,
        past_actions=None,
        state=None,
        trajectory_ids=None,
        initial_context_tokens=None,
        pool_mask=None,
        utility_scores=None,
        utility_candidate_mask=None,
        utility_cost_scores=None,
        utility_value_targets=None,
        utility_progress_targets=None,
        utility_uncertainty_targets=None,
        utility_target_mask=None,
        execution_state_target=None,
        prediction_state_target=None,
        execution_state_target_mask=None,
        prediction_state_target_mask=None,
        previous_router_probs=None,
        latent_action_tokens: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        actions = self._actions_to_tensor(actions, embodied_action_tokens)
        if actions_are_future and actions.shape[1] != self.action_horizon:
            raise ValueError(
                f"future_actions must have exactly {self.action_horizon} steps, got {actions.shape[1]}"
            )
        if not actions_are_future and actions.shape[1] < self.action_horizon:
            raise ValueError(
                f"Expected at least {self.action_horizon} action steps, got {actions.shape[1]}"
            )
        if past_actions is not None:
            past_actions = self._actions_to_tensor(past_actions, embodied_action_tokens)
        # Prefer explicit future-only windows from the dataloader. The legacy
        # `action` fallback can still contain wider context, so it keeps tail slicing.
        actions_target = actions if actions_are_future else actions[:, -self.action_horizon :, :]
        action_mask_target = self._action_mask_to_tensor(action_mask, actions, actions_are_future)
        aux_losses = {}
        if self.latent_action_head is not None:
            latent_actions_target = actions_target[:, : self.latent_action_horizon, :]
            latent_action_mask = (
                action_mask_target[:, : self.latent_action_horizon] if action_mask_target is not None else None
            )
            latent_output = self.latent_action_head(
                embodied_action_tokens,
                latent_actions_target,
                future_action_mask=latent_action_mask,
            )
            latent_action_tokens = latent_output.tokens
            aux_losses = {
                "latent_action_loss": latent_output.loss,
                "latent_action_reconstruction_loss": latent_output.reconstruction_loss,
                "latent_action_reconstruction_loss_weight": torch.as_tensor(
                    self.latent_reconstruction_loss_weight,
                    device=latent_output.loss.device,
                    dtype=latent_output.loss.dtype,
                ),
                "latent_action_reconstruction_loss_weighted": (
                    self.latent_reconstruction_loss_weight * latent_output.reconstruction_loss
                ),
                "latent_action_vq_loss": latent_output.vq_loss,
                "latent_action_vq_loss_weight": torch.as_tensor(
                    self.latent_action_head.vq_loss_weight,
                    device=latent_output.loss.device,
                    dtype=latent_output.loss.dtype,
                ),
                "latent_action_vq_loss_weighted": (
                    self.latent_action_head.vq_loss_weight * latent_output.vq_loss
                ),
                "latent_action_prior_loss": latent_output.prior_loss,
                "latent_action_prior_loss_weight": torch.as_tensor(
                    self.latent_action_head.prior_loss_weight,
                    device=latent_output.loss.device,
                    dtype=latent_output.loss.dtype,
                ),
                "latent_action_prior_loss_weighted": (
                    self.latent_action_head.prior_loss_weight * latent_output.prior_loss
                ),
                "latent_action_code_usage_loss": latent_output.code_usage_loss,
                "latent_action_code_usage_loss_weight": torch.as_tensor(
                    self.latent_action_head.code_usage_loss_weight,
                    device=latent_output.loss.device,
                    dtype=latent_output.loss.dtype,
                ),
                "latent_action_code_usage_loss_weighted": (
                    self.latent_action_head.code_usage_loss_weight * latent_output.code_usage_loss
                ),
                "latent_action_perplexity": latent_output.perplexity,
            }
        if self.transition_head is not None:
            transition_loss = self._transition_loss(
                embodied_action_tokens,
                latent_action_tokens,
                execution_state_target=execution_state_target,
                prediction_state_target=prediction_state_target,
                execution_state_target_mask=execution_state_target_mask,
                prediction_state_target_mask=prediction_state_target_mask,
            )
            if transition_loss is not None:
                aux_losses["transition_state_loss_raw"] = transition_loss.detach()
                aux_losses["transition_state_loss_weight"] = torch.as_tensor(
                    self.transition_loss_weight,
                    device=transition_loss.device,
                    dtype=transition_loss.dtype,
                )
                aux_losses["transition_state_loss"] = self.transition_loss_weight * transition_loss
                aux_losses["transition_state_loss_weighted"] = self.transition_loss_weight * transition_loss.detach()
        actions_target_repeated = actions_target.repeat_interleave(self.repeated_diffusion_steps, dim=0)
        action_mask_repeated = (
            action_mask_target.repeat_interleave(self.repeated_diffusion_steps, dim=0)
            if action_mask_target is not None
            else None
        )
        conditioning_tokens = self._conditioning_tokens(embodied_action_tokens, latent_action_tokens)
        state_tensor = self._state_to_tensor(state, embodied_action_tokens)
        router_actions_target = actions_target[:, : self.router_horizon, :]
        utility_actions_target = actions_target[:, : self.utility_horizon, :]
        router_action_mask = action_mask_target[:, : self.router_horizon] if action_mask_target is not None else None
        utility_action_mask = action_mask_target[:, : self.utility_horizon] if action_mask_target is not None else None
        direct_expert_actions = None
        direct_routed_action_loss = None
        if self.lara_moe is not None:
            direct_expert_loss = None
            direct_expert_losses = None
            direct_posterior_losses = None
            direct_posterior_mask = None
            direct_assignment_support = None
            direct_assignment_active = None
            utility_expert_losses = None
            state_utility_losses = None
            state_utility_component_targets = None
            direct_expert_residuals = None
            direct_expert_residual_info = {}
            shared_actions = None
            shared_router_losses = None
            direct_residual_regularization_loss = None
            if self.direct_action_experts is not None:
                (
                    direct_expert_actions,
                    direct_expert_residuals,
                    shared_actions,
                    direct_expert_residual_info,
                ) = self._direct_expert_action_bank(
                    conditioning_tokens,
                    state_tensor,
                )
                direct_expert_loss_components = self.direct_action_experts.reconstruction_loss_components(
                    direct_expert_actions,
                    actions_target,
                    action_mask=action_mask_target,
                    execution_horizon=self.execution_horizon,
                    execution_loss_weight=self.config.framework.action_model.get("execution_loss_weight", 1.0),
                    prediction_loss_weight=self.config.framework.action_model.get("prediction_loss_weight", 1.0),
                )
                direct_expert_losses = self.direct_action_experts.reconstruction_losses(
                    direct_expert_actions[:, :, : self.router_horizon, :],
                    router_actions_target,
                    action_mask=router_action_mask,
                )
                if shared_actions is not None:
                    shared_router_losses = self.direct_action_experts.reconstruction_losses(
                        shared_actions[:, None, : self.router_horizon, :],
                        router_actions_target,
                        action_mask=router_action_mask,
                    ).squeeze(1)
                utility_expert_losses = self.direct_action_experts.reconstruction_losses(
                    direct_expert_actions[:, :, : self.utility_horizon, :],
                    utility_actions_target,
                    action_mask=utility_action_mask,
                )
                direct_utility_component_targets = None
                if self.use_action_loss_utility_components:
                    direct_utility_loss_components = self.direct_action_experts.reconstruction_loss_components(
                        direct_expert_actions,
                        actions_target,
                        action_mask=action_mask_target,
                        execution_horizon=self.utility_horizon,
                    )
                    direct_utility_component_targets = utility_component_targets_from_expert_losses(
                        value_losses=utility_expert_losses,
                        progress_losses=direct_utility_loss_components["execution"],
                        uncertainty_losses=direct_utility_loss_components["prediction"],
                        temperature=self.action_loss_utility_temperature,
                        normalize=self.action_loss_utility_normalize,
                    )
                direct_posterior_losses = direct_expert_losses.detach()
                if (
                    self.direct_expert_improvement_posterior
                    and shared_router_losses is not None
                ):
                    improvement = shared_router_losses[:, None].detach() - direct_expert_losses.detach()
                    direct_posterior_losses = direct_posterior_losses - shared_router_losses[:, None].detach()
                    if self.direct_expert_residual_cost_weight > 0 and direct_expert_residuals is not None:
                        residuals_for_cost = direct_expert_residual_info.get(
                            "pre_clamp_residual_actions",
                            direct_expert_residuals,
                        )
                        residual_cost = self._expert_residual_norm(
                            residuals_for_cost[:, :, : self.router_horizon, :]
                        ).detach()
                        direct_posterior_losses = (
                            direct_posterior_losses
                            + self.direct_expert_residual_cost_weight * residual_cost
                        )
                    if self.direct_expert_hard_assignment:
                        direct_posterior_mask = self._improvement_assignment_mask(
                            improvement,
                            top_r=self.direct_expert_posterior_top_r,
                            margin=self.direct_expert_improvement_margin,
                        )
                        direct_assignment_support = direct_posterior_mask
                        direct_assignment_active = direct_posterior_mask.any(dim=-1)
                        # posterior_from_expert_losses requires positive support. For shared-only
                        # samples, keep a top-1 fallback for a valid posterior and gate the loss below.
                        fallback_mask = self._topk_boolean_mask(improvement, top_k=1)
                        direct_posterior_mask = torch.where(
                            direct_assignment_active[:, None],
                            direct_posterior_mask,
                            fallback_mask,
                        )
                direct_posterior_top_r = self.direct_expert_posterior_top_r
                if direct_posterior_top_r is None:
                    direct_posterior_top_r = self.lara_moe.posterior_top_r
                direct_posterior = posterior_from_expert_losses(
                    direct_posterior_losses,
                    temperature=self.lara_moe.posterior_temperature,
                    mask=direct_posterior_mask,
                    uniform_floor=self.lara_moe.posterior_uniform_floor,
                    top_r=direct_posterior_top_r,
                )
                direct_expert_loss_per_sample = (
                    direct_expert_loss_components["weighted"] * direct_posterior
                ).sum(dim=-1)
                if direct_assignment_active is not None and self.direct_expert_shared_only_gate:
                    assignment_weight = direct_assignment_active.to(
                        dtype=direct_expert_loss_per_sample.dtype
                    )
                    direct_expert_loss = (
                        direct_expert_loss_per_sample * assignment_weight
                    ).sum() / assignment_weight.sum().clamp_min(1.0)
                else:
                    direct_expert_loss = direct_expert_loss_per_sample.mean()
                aux_losses.update(
                    self._posterior_diagnostics(
                        direct_posterior,
                        prefix="moe_direct_posterior",
                        support_mask=direct_posterior_mask,
                    )
                )
                if direct_expert_residuals is not None:
                    residuals_for_norm_loss = direct_expert_residual_info.get(
                        "pre_clamp_residual_actions",
                        direct_expert_residuals,
                    )
                    residual_norm = self._expert_residual_norm(direct_expert_residuals)
                    residual_norm_loss = residuals_for_norm_loss.pow(2).mean()
                    residual_diversity_loss = expert_diversity_loss(direct_expert_residuals)
                    direct_residual_regularization_loss = (
                        self.direct_residual_norm_loss_weight * residual_norm_loss
                        + self.direct_residual_diversity_loss_weight * residual_diversity_loss
                    )
                    raw_residual_norm = direct_expert_residual_info.get("raw_residual_norm")
                    residual_clamp_rate = direct_expert_residual_info.get("residual_clamp_rate")
                    residual_metrics = {
                        "moe_direct_residual_norm": residual_norm.mean().detach(),
                        "moe_direct_residual_raw_norm": (
                            raw_residual_norm.mean().detach()
                            if raw_residual_norm is not None
                            else residual_norm.mean().detach()
                        ),
                        "moe_direct_residual_clamp_rate": (
                            residual_clamp_rate.mean().detach()
                            if residual_clamp_rate is not None
                            else torch.zeros((), device=direct_expert_residuals.device)
                        ),
                        "moe_direct_residual_scale_base": direct_expert_residual_info[
                            "base_scale"
                        ].detach(),
                        "moe_direct_residual_scale_effective": direct_expert_residual_info[
                            "effective_scale"
                        ].detach(),
                        "moe_direct_residual_warmup_fraction": direct_expert_residual_info[
                            "warmup_fraction"
                        ].detach(),
                        "moe_direct_residual_max_norm": torch.as_tensor(
                            self.direct_expert_residual_max_norm or 0.0,
                            device=direct_expert_residuals.device,
                            dtype=direct_expert_residuals.dtype,
                        ),
                        "moe_direct_residual_norm_loss_raw": residual_norm_loss.detach(),
                        "moe_direct_residual_norm_loss_weight": torch.as_tensor(
                            self.direct_residual_norm_loss_weight,
                            device=residual_norm_loss.device,
                            dtype=residual_norm_loss.dtype,
                        ),
                        "moe_direct_residual_norm_loss_weighted": (
                            self.direct_residual_norm_loss_weight * residual_norm_loss.detach()
                        ),
                        "moe_direct_residual_diversity_loss_raw": residual_diversity_loss.detach(),
                        "moe_direct_residual_diversity_loss_weight": torch.as_tensor(
                            self.direct_residual_diversity_loss_weight,
                            device=residual_diversity_loss.device,
                            dtype=residual_diversity_loss.dtype,
                        ),
                        "moe_direct_residual_diversity_loss_weighted": (
                            self.direct_residual_diversity_loss_weight * residual_diversity_loss.detach()
                        ),
                        "moe_direct_residual_regularization_loss": direct_residual_regularization_loss,
                        "moe_direct_residual_regularization_loss_weighted": direct_residual_regularization_loss.detach(),
                    }
                    raw_norm_by_expert = (
                        raw_residual_norm
                        if raw_residual_norm is not None
                        else residual_norm
                    )
                    clamp_rate_by_expert = (
                        residual_clamp_rate
                        if residual_clamp_rate is not None
                        else torch.zeros_like(residual_norm)
                    )
                    for expert_idx in range(residual_norm.shape[-1]):
                        residual_metrics.update(
                            {
                                f"moe_direct_residual_norm_{expert_idx}": residual_norm[:, expert_idx].mean().detach(),
                                f"moe_direct_residual_raw_norm_{expert_idx}": raw_norm_by_expert[:, expert_idx].mean().detach(),
                                f"moe_direct_residual_clamp_rate_{expert_idx}": clamp_rate_by_expert[:, expert_idx].mean().detach(),
                            }
                        )
                    aux_losses.update(residual_metrics)
                if shared_router_losses is not None:
                    improvement = shared_router_losses[:, None].detach() - direct_expert_losses.detach()
                    improvement_metrics = {
                        "moe_shared_action_loss": shared_router_losses.mean().detach(),
                        "moe_direct_expert_improvement_mean": improvement.mean().detach(),
                        "moe_direct_expert_improvement_top1": improvement.max(dim=-1).values.mean().detach(),
                        "moe_direct_expert_improvement_positive_rate": (improvement > 0).float().mean().detach(),
                        "moe_direct_expert_improvement_margin": torch.as_tensor(
                            self.direct_expert_improvement_margin,
                            device=improvement.device,
                            dtype=improvement.dtype,
                        ),
                        "moe_direct_expert_improvement_candidate_rate": (
                            improvement > self.direct_expert_improvement_margin
                        )
                        .float()
                        .mean()
                        .detach(),
                    }
                    for expert_idx in range(improvement.shape[-1]):
                        expert_improvement = improvement[:, expert_idx]
                        improvement_metrics.update(
                            {
                                f"moe_direct_expert_improvement_mean_{expert_idx}": expert_improvement.mean().detach(),
                                f"moe_direct_expert_improvement_positive_rate_{expert_idx}": (
                                    expert_improvement > 0
                                )
                                .float()
                                .mean()
                                .detach(),
                            }
                        )
                    if direct_assignment_active is not None:
                        selected_count = direct_assignment_support.to(dtype=improvement.dtype).sum(dim=-1)
                        assignment_usage = direct_assignment_support.to(dtype=improvement.dtype).mean(dim=0)
                        improvement_metrics.update(
                            {
                                "moe_direct_assignment_active_rate": direct_assignment_active.float().mean().detach(),
                                "moe_direct_assignment_shared_only_rate": (
                                    1.0 - direct_assignment_active.float().mean()
                                ).detach(),
                                "moe_direct_assignment_selected_experts": selected_count.mean().detach(),
                            }
                        )
                        for expert_idx in range(assignment_usage.shape[-1]):
                            improvement_metrics[f"moe_direct_assignment_usage_{expert_idx}"] = assignment_usage[
                                expert_idx
                            ].detach()
                    aux_losses.update(improvement_metrics)
            else:
                direct_utility_component_targets = None
            with torch.no_grad():
                expert_action_losses = (
                    direct_posterior_losses.detach()
                    if direct_posterior_losses is not None
                    else self._expert_action_losses(
                        conditioning_tokens,
                        router_actions_target,
                        state,
                        action_mask=router_action_mask,
                    )
                    if self.use_expert_loss_posterior
                    else None
                )
                if utility_expert_losses is None and self.use_action_loss_utility:
                    utility_expert_losses = self._expert_action_losses(
                        conditioning_tokens,
                        utility_actions_target,
                        state,
                        action_mask=utility_action_mask,
                    )
                if self.use_state_utility:
                    state_utility_loss_components = self._expert_transition_loss_components(
                        conditioning_tokens,
                        latent_action_tokens,
                        execution_state_target=execution_state_target,
                        prediction_state_target=prediction_state_target,
                        execution_state_target_mask=execution_state_target_mask,
                        prediction_state_target_mask=prediction_state_target_mask,
                    )
                    if state_utility_loss_components is not None and torch.all(
                        state_utility_loss_components["valid_mask"]
                    ):
                        state_utility_losses = state_utility_loss_components["weighted"]
                        if self.use_state_utility_components:
                            state_utility_component_targets = utility_component_targets_from_expert_losses(
                                value_losses=state_utility_losses,
                                progress_losses=state_utility_loss_components["execution"],
                                uncertainty_losses=state_utility_loss_components["prediction"],
                                temperature=self.state_utility_temperature,
                                normalize=self.state_utility_normalize,
                            )
            utility_scores = (
                self._as_tensor(utility_scores, device=conditioning_tokens.device, dtype=conditioning_tokens.dtype)
                if utility_scores is not None
                else None
            )
            if utility_scores is None and self.use_state_utility and state_utility_losses is not None:
                utility_scores = utility_from_expert_losses(
                    state_utility_losses,
                    temperature=self.state_utility_temperature,
                    normalize=self.state_utility_normalize,
                ).to(dtype=conditioning_tokens.dtype)
            if utility_scores is None and self.use_action_loss_utility and utility_expert_losses is not None:
                utility_scores = utility_from_expert_losses(
                    utility_expert_losses,
                    temperature=self.action_loss_utility_temperature,
                    normalize=self.action_loss_utility_normalize,
                ).to(dtype=conditioning_tokens.dtype)
            utility_candidate_mask = (
                self._as_tensor(utility_candidate_mask, device=conditioning_tokens.device, dtype=torch.bool)
                if utility_candidate_mask is not None
                else None
            )
            utility_cost_scores = (
                self._as_tensor(utility_cost_scores, device=conditioning_tokens.device, dtype=conditioning_tokens.dtype)
                if utility_cost_scores is not None
                else None
            )
            utility_value_targets = (
                self._as_tensor(utility_value_targets, device=conditioning_tokens.device, dtype=conditioning_tokens.dtype)
                if utility_value_targets is not None
                else None
            )
            utility_progress_targets = (
                self._as_tensor(
                    utility_progress_targets,
                    device=conditioning_tokens.device,
                    dtype=conditioning_tokens.dtype,
                )
                if utility_progress_targets is not None
                else None
            )
            utility_uncertainty_targets = (
                self._as_tensor(
                    utility_uncertainty_targets,
                    device=conditioning_tokens.device,
                    dtype=conditioning_tokens.dtype,
                )
                if utility_uncertainty_targets is not None
                else None
            )
            utility_target_mask = (
                self._as_tensor(utility_target_mask, device=conditioning_tokens.device, dtype=torch.bool)
                if utility_target_mask is not None
                else None
            )
            if direct_utility_component_targets is not None and state_utility_component_targets is None:
                if utility_value_targets is None:
                    utility_value_targets = direct_utility_component_targets["value"].to(
                        device=conditioning_tokens.device,
                        dtype=conditioning_tokens.dtype,
                    )
                if utility_progress_targets is None:
                    utility_progress_targets = direct_utility_component_targets["progress"].to(
                        device=conditioning_tokens.device,
                        dtype=conditioning_tokens.dtype,
                    )
                if utility_uncertainty_targets is None:
                    utility_uncertainty_targets = direct_utility_component_targets["uncertainty"].to(
                        device=conditioning_tokens.device,
                        dtype=conditioning_tokens.dtype,
                    )
                if utility_target_mask is None:
                    utility_target_mask = torch.ones_like(utility_value_targets, dtype=torch.bool)
            if state_utility_component_targets is not None:
                if utility_value_targets is None:
                    utility_value_targets = state_utility_component_targets["value"].to(
                        device=conditioning_tokens.device,
                        dtype=conditioning_tokens.dtype,
                    )
                if utility_progress_targets is None:
                    utility_progress_targets = state_utility_component_targets["progress"].to(
                        device=conditioning_tokens.device,
                        dtype=conditioning_tokens.dtype,
                    )
                if utility_uncertainty_targets is None:
                    utility_uncertainty_targets = state_utility_component_targets["uncertainty"].to(
                        device=conditioning_tokens.device,
                        dtype=conditioning_tokens.dtype,
                    )
                if utility_target_mask is None:
                    utility_target_mask = torch.ones_like(utility_value_targets, dtype=torch.bool)
            previous_router_probs = (
                self._as_tensor(previous_router_probs, device=conditioning_tokens.device, dtype=conditioning_tokens.dtype)
                if previous_router_probs is not None
                else None
            )
            initial_context_tokens = (
                self._as_tensor(initial_context_tokens, device=conditioning_tokens.device, dtype=conditioning_tokens.dtype)
                if initial_context_tokens is not None
                else None
            )
            pool_mask = (
                self._as_tensor(pool_mask, device=conditioning_tokens.device, dtype=torch.bool)
                if pool_mask is not None
                else None
            )
            moe_output = self.lara_moe(
                conditioning_tokens,
                latent_action_tokens=latent_action_tokens,
                initial_context_tokens=initial_context_tokens,
                pool_mask=pool_mask,
                expert_action_losses=expert_action_losses,
                pool_target_probs=self._pool_target_probs(
                    expert_action_losses,
                    trajectory_ids,
                    utility_scores=utility_scores,
                    utility_candidate_mask=utility_candidate_mask,
                ),
                utility_scores=utility_scores,
                utility_candidate_mask=utility_candidate_mask,
                utility_cost_scores=utility_cost_scores,
                utility_value_targets=utility_value_targets,
                utility_progress_targets=utility_progress_targets,
                utility_uncertainty_targets=utility_uncertainty_targets,
                utility_target_mask=utility_target_mask,
                previous_router_probs=previous_router_probs,
            )
            conditioning_tokens = moe_output.tokens
            aux_losses.update(
                {
                    "moe_loss": moe_output.loss,
                    "moe_total_loss": moe_output.loss.detach(),
                    "moe_router_loss": moe_output.route_loss_weighted,
                    "moe_route_distill_loss": moe_output.route_loss,
                    "moe_route_distill_loss_raw": moe_output.route_loss,
                    "moe_route_distill_loss_weight": torch.as_tensor(
                        self.lara_moe.router_loss_weight,
                        device=moe_output.loss.device,
                        dtype=moe_output.loss.dtype,
                    ),
                    "moe_route_distill_loss_weighted": moe_output.route_loss_weighted,
                    "moe_pool_distill_loss": moe_output.pool_loss,
                    "moe_pool_distill_loss_raw": moe_output.pool_loss,
                    "moe_pool_distill_loss_weight": torch.as_tensor(
                        self.lara_moe.pool_loss_weight,
                        device=moe_output.loss.device,
                        dtype=moe_output.loss.dtype,
                    ),
                    "moe_pool_distill_loss_weighted": moe_output.pool_loss_weighted,
                    "moe_pool_coverage_loss": moe_output.pool_coverage_loss,
                    "moe_pool_coverage_loss_raw": moe_output.pool_coverage_loss,
                    "moe_pool_coverage_loss_weight": torch.as_tensor(
                        self.lara_moe.pool_coverage_loss_weight,
                        device=moe_output.loss.device,
                        dtype=moe_output.loss.dtype,
                    ),
                    "moe_pool_coverage_loss_weighted": moe_output.pool_coverage_loss_weighted,
                    "moe_utility_loss": moe_output.utility_loss,
                    "moe_utility_loss_raw": moe_output.utility_loss,
                    "moe_utility_loss_weight": torch.as_tensor(
                        self.lara_moe.utility_loss_weight,
                        device=moe_output.loss.device,
                        dtype=moe_output.loss.dtype,
                    ),
                    "moe_utility_loss_weighted": moe_output.utility_loss_weighted,
                    "moe_utility_rank_loss": moe_output.utility_rank_loss,
                    "moe_utility_rank_loss_raw": moe_output.utility_rank_loss,
                    "moe_utility_rank_loss_weight": torch.as_tensor(
                        self.lara_moe.utility_rank_loss_weight,
                        device=moe_output.loss.device,
                        dtype=moe_output.loss.dtype,
                    ),
                    "moe_utility_rank_loss_weighted": moe_output.utility_rank_loss_weighted,
                    "moe_utility_head_loss": moe_output.utility_head_loss,
                    "moe_utility_head_loss_raw": moe_output.utility_head_loss,
                    "moe_utility_head_loss_weight": torch.as_tensor(
                        self.lara_moe.utility_head_loss_weight,
                        device=moe_output.loss.device,
                        dtype=moe_output.loss.dtype,
                    ),
                    "moe_utility_head_loss_weighted": moe_output.utility_head_loss_weighted,
                    "moe_balance_loss": moe_output.balance_loss,
                    "moe_balance_loss_raw": moe_output.balance_loss,
                    "moe_balance_loss_weight": torch.as_tensor(
                        self.lara_moe.balance_loss_weight,
                        device=moe_output.loss.device,
                        dtype=moe_output.loss.dtype,
                    ),
                    "moe_balance_loss_weighted": moe_output.balance_loss_weighted,
                    "moe_stickiness_loss": moe_output.stickiness_loss,
                    "moe_stickiness_loss_raw": moe_output.stickiness_loss,
                    "moe_stickiness_loss_weight": torch.as_tensor(
                        self.lara_moe.stickiness_loss_weight,
                        device=moe_output.loss.device,
                        dtype=moe_output.loss.dtype,
                    ),
                    "moe_stickiness_loss_weighted": moe_output.stickiness_loss_weighted,
                    "moe_diversity_loss": moe_output.diversity_loss,
                    "moe_diversity_loss_raw": moe_output.diversity_loss,
                    "moe_diversity_loss_weight": torch.as_tensor(
                        self.lara_moe.diversity_loss_weight,
                        device=moe_output.loss.device,
                        dtype=moe_output.loss.dtype,
                    ),
                    "moe_diversity_loss_weighted": moe_output.diversity_loss_weighted,
                    "moe_entropy_loss": moe_output.entropy_loss,
                    "moe_entropy_loss_raw": moe_output.entropy_loss,
                    "moe_entropy_loss_weight": torch.as_tensor(
                        self.lara_moe.entropy_loss_weight,
                        device=moe_output.loss.device,
                        dtype=moe_output.loss.dtype,
                    ),
                    "moe_entropy_loss_weighted": moe_output.entropy_loss_weighted,
                    "moe_utility_calibration_error": moe_output.utility_calibration_error,
                    "moe_router_entropy": moe_output.router_entropy,
                    "moe_posterior_entropy": moe_output.posterior_entropy,
                    "moe_pool_entropy": moe_output.pool_entropy,
                    "moe_dead_expert_ratio": moe_output.dead_expert_ratio,
                    "moe_pool_dead_expert_ratio": moe_output.pool_dead_expert_ratio,
                    "moe_route_top1_match": moe_output.route_top1_match,
                    "moe_route_regret": moe_output.route_regret,
                    "moe_pool_teacher_mass": moe_output.pool_teacher_mass,
                    "moe_active_teacher_mass": moe_output.active_teacher_mass,
                    "moe_pool_teacher_top1_match": moe_output.pool_teacher_top1_match,
                    "moe_active_teacher_top1_match": moe_output.active_teacher_top1_match,
                    "moe_pool_critical_miss_rate": moe_output.pool_critical_miss_rate,
                }
            )
            if moe_output.utility_scores is not None:
                aux_losses.update(
                    {
                        "moe_utility_scores": moe_output.utility_scores,
                        "moe_utility_value_scores": moe_output.utility_value_scores,
                        "moe_utility_progress_scores": moe_output.utility_progress_scores,
                        "moe_utility_uncertainty_scores": moe_output.utility_uncertainty_scores,
                    }
                )
            for metric_name, metric_value in route_quality_metrics(
                moe_output.router_probs,
                posterior_probs=moe_output.posterior_probs,
                utility_scores=moe_output.utility_scores,
                pool_mask=moe_output.pool_mask,
                active_mask=moe_output.active_mask,
                retention_fractions=self.route_retention_fractions,
                critical_threshold=self.pool_critical_threshold,
            ).items():
                safe_metric_name = metric_name.replace("/", "_").replace(".", "_")
                aux_losses[f"moe_route_quality_{safe_metric_name}"] = metric_value
            if direct_expert_loss is not None:
                aux_losses["moe_direct_expert_loss_raw"] = direct_expert_loss.detach()
                aux_losses["moe_direct_expert_loss_weight"] = torch.as_tensor(
                    self.direct_expert_loss_weight,
                    device=direct_expert_loss.device,
                    dtype=direct_expert_loss.dtype,
                )
                aux_losses["moe_direct_expert_loss"] = self.direct_expert_loss_weight * direct_expert_loss
                aux_losses["moe_direct_expert_loss_weighted"] = (
                    self.direct_expert_loss_weight * direct_expert_loss.detach()
                )
            if state_utility_losses is not None:
                aux_losses["moe_state_utility_error"] = state_utility_losses.mean().detach()
            if self.use_direct_action_output:
                direct_routed_actions = ActionChunkExpertBank.routed_actions(
                    direct_expert_actions,
                    moe_output.posterior_probs if self.training else moe_output.router_probs,
                )
                direct_routed_action_loss = ActionChunkExpertBank.action_chunk_loss(
                    direct_routed_actions,
                    actions_target,
                    action_mask=action_mask_target,
                    execution_horizon=self.execution_horizon,
                    execution_loss_weight=self.config.framework.action_model.get("execution_loss_weight", 1.0),
                    prediction_loss_weight=self.config.framework.action_model.get("prediction_loss_weight", 1.0),
                )
                aux_losses["moe_direct_routed_action_loss"] = direct_routed_action_loss
        context_repeated = conditioning_tokens.repeat_interleave(self.repeated_diffusion_steps, dim=0)

        state_repeated = (
            state_tensor.repeat_interleave(self.repeated_diffusion_steps, dim=0)
            if state_tensor is not None
            else None
        )

        action_loss = (
            direct_routed_action_loss
            if self.use_direct_action_output
            else self.action_model(
                context_repeated,
                actions_target_repeated,
                state_repeated,
                action_mask=action_mask_repeated,
            )
        )
        if not return_aux:
            return (
                action_loss
                + aux_losses.get("latent_action_loss", 0.0)
                + aux_losses.get("transition_state_loss", 0.0)
                + aux_losses.get("moe_loss", 0.0)
                + aux_losses.get("moe_direct_expert_loss", 0.0)
                + aux_losses.get("moe_direct_residual_regularization_loss", 0.0)
            )
        aux_losses["action_loss"] = action_loss
        aux_losses["total_action_loss"] = (
            action_loss
            + aux_losses.get("latent_action_loss", 0.0)
            + aux_losses.get("transition_state_loss", 0.0)
            + aux_losses.get("moe_loss", 0.0)
            + aux_losses.get("moe_direct_expert_loss", 0.0)
            + aux_losses.get("moe_direct_residual_regularization_loss", 0.0)
        )
        return aux_losses

    def _transition_loss(
        self,
        context_tokens: torch.Tensor,
        latent_action_tokens: Optional[torch.Tensor],
        execution_state_target=None,
        prediction_state_target=None,
        execution_state_target_mask=None,
        prediction_state_target_mask=None,
    ) -> Optional[torch.Tensor]:
        if self.transition_head is None:
            return None
        if latent_action_tokens is None:
            return None
        execution_target = self._boundary_state_to_tensor(execution_state_target, context_tokens)
        prediction_target = self._boundary_state_to_tensor(prediction_state_target, context_tokens)
        if execution_target is None and prediction_target is None:
            return None
        batch_size = context_tokens.shape[0]
        execution_mask = self._boundary_mask_to_tensor(execution_state_target_mask, context_tokens, batch_size)
        prediction_mask = self._boundary_mask_to_tensor(prediction_state_target_mask, context_tokens, batch_size)

        pred_states = self.transition_head(context_tokens, latent_action_tokens).float()
        losses = []
        if execution_target is not None:
            losses.append(
                self.execution_transition_loss_weight
                * self._masked_boundary_loss(pred_states[:, 0, :], execution_target, execution_mask)
            )
        if prediction_target is not None:
            losses.append(
                self.prediction_transition_loss_weight
                * self._masked_boundary_loss(pred_states[:, 1, :], prediction_target, prediction_mask)
            )
        return torch.stack(losses).sum()

    def _expert_transition_loss_components(
        self,
        conditioning_tokens: torch.Tensor,
        latent_action_tokens: Optional[torch.Tensor],
        execution_state_target=None,
        prediction_state_target=None,
        execution_state_target_mask=None,
        prediction_state_target_mask=None,
    ) -> Optional[dict[str, torch.Tensor]]:
        if self.transition_head is None or self.lara_moe is None or latent_action_tokens is None:
            return None
        execution_target = self._boundary_state_to_tensor(execution_state_target, conditioning_tokens)
        prediction_target = self._boundary_state_to_tensor(prediction_state_target, conditioning_tokens)
        if execution_target is None and prediction_target is None:
            return None

        batch_size = conditioning_tokens.shape[0]
        execution_mask = self._boundary_mask_to_tensor(execution_state_target_mask, conditioning_tokens, batch_size)
        prediction_mask = self._boundary_mask_to_tensor(prediction_state_target_mask, conditioning_tokens, batch_size)
        if execution_mask is None and execution_target is not None:
            execution_mask = torch.ones(batch_size, device=conditioning_tokens.device, dtype=torch.bool)
        if prediction_mask is None and prediction_target is not None:
            prediction_mask = torch.ones(batch_size, device=conditioning_tokens.device, dtype=torch.bool)

        expert_tokens = self.lara_moe.expert_conditioning_tokens(conditioning_tokens)
        _, num_experts, token_count, hidden_size = expert_tokens.shape
        flat_tokens = expert_tokens.reshape(batch_size * num_experts, token_count, hidden_size)
        flat_latents = latent_action_tokens[:, None, :, :].expand(-1, num_experts, -1, -1)
        flat_latents = flat_latents.reshape(
            batch_size * num_experts,
            latent_action_tokens.shape[1],
            latent_action_tokens.shape[2],
        )
        pred_states = self.transition_head(flat_tokens, flat_latents).float()
        pred_states = pred_states.view(batch_size, num_experts, pred_states.shape[1], pred_states.shape[2])

        execution_loss = conditioning_tokens.new_zeros(batch_size, num_experts, dtype=torch.float32)
        prediction_loss = conditioning_tokens.new_zeros(batch_size, num_experts, dtype=torch.float32)
        weighted_loss = conditioning_tokens.new_zeros(batch_size, num_experts, dtype=torch.float32)
        weight_denom = conditioning_tokens.new_zeros(batch_size, 1, dtype=torch.float32)
        valid_mask = torch.zeros(batch_size, device=conditioning_tokens.device, dtype=torch.bool)

        if execution_target is not None:
            execution_loss = F.smooth_l1_loss(
                pred_states[:, :, 0, :],
                execution_target[:, None, :].expand(-1, num_experts, -1),
                reduction="none",
            ).mean(dim=-1)
            execution_weight = execution_mask.to(device=conditioning_tokens.device, dtype=torch.float32).view(-1, 1)
            weighted_loss = weighted_loss + self.execution_transition_loss_weight * execution_loss * execution_weight
            weight_denom = weight_denom + self.execution_transition_loss_weight * execution_weight
            valid_mask = valid_mask | execution_mask.to(device=conditioning_tokens.device, dtype=torch.bool)

        if prediction_target is not None:
            prediction_loss = F.smooth_l1_loss(
                pred_states[:, :, 1, :],
                prediction_target[:, None, :].expand(-1, num_experts, -1),
                reduction="none",
            ).mean(dim=-1)
            prediction_weight = prediction_mask.to(device=conditioning_tokens.device, dtype=torch.float32).view(-1, 1)
            weighted_loss = weighted_loss + self.prediction_transition_loss_weight * prediction_loss * prediction_weight
            weight_denom = weight_denom + self.prediction_transition_loss_weight * prediction_weight
            valid_mask = valid_mask | prediction_mask.to(device=conditioning_tokens.device, dtype=torch.bool)

        weighted_loss = weighted_loss / weight_denom.clamp_min(1.0)
        return {
            "weighted": weighted_loss,
            "execution": execution_loss,
            "prediction": prediction_loss,
            "valid_mask": valid_mask,
        }

    @staticmethod
    def _masked_boundary_loss(
        predicted: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        per_sample_loss = torch.nn.functional.smooth_l1_loss(
            predicted,
            target.detach(),
            reduction="none",
        ).mean(dim=-1)
        if valid_mask is None:
            return per_sample_loss.mean()
        mask = valid_mask.to(device=predicted.device, dtype=per_sample_loss.dtype)
        return (per_sample_loss * mask).sum() / mask.sum().clamp_min(1.0)

    @staticmethod
    def _topk_boolean_mask(scores: torch.Tensor, top_k: Optional[int]) -> torch.Tensor:
        if top_k is None or top_k >= scores.shape[-1]:
            return torch.ones_like(scores, dtype=torch.bool)
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        indices = scores.topk(k=top_k, dim=-1).indices
        mask = torch.zeros_like(scores, dtype=torch.bool)
        return mask.scatter(dim=-1, index=indices, value=True)

    def _improvement_assignment_mask(
        self,
        improvement: torch.Tensor,
        top_r: Optional[int],
        margin: float,
    ) -> torch.Tensor:
        support = improvement > margin
        if top_r is not None:
            support = support & self._topk_boolean_mask(improvement, top_k=top_r)
        return support

    @staticmethod
    def _percentile(values: torch.Tensor, q: float) -> torch.Tensor:
        if values.numel() == 0:
            return values.new_zeros(())
        sorted_values = values.sort().values
        index = int(round((sorted_values.numel() - 1) * q))
        index = max(0, min(sorted_values.numel() - 1, index))
        return sorted_values[index]

    @classmethod
    def _posterior_diagnostics(
        cls,
        probs: torch.Tensor,
        prefix: str,
        support_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            safe_probs = probs.detach()
            entropy_per_sample = -(safe_probs * torch.log(safe_probs.clamp_min(1e-8))).sum(dim=-1)
            sorted_probs = safe_probs.sort(dim=-1, descending=True).values
            top2_count = min(2, safe_probs.shape[-1])
            usage = safe_probs.mean(dim=0)
            metrics = {
                f"{prefix}_entropy": entropy_per_sample.mean(),
                f"{prefix}_entropy_min": entropy_per_sample.min(),
                f"{prefix}_entropy_p50": cls._percentile(entropy_per_sample, 0.5),
                f"{prefix}_entropy_p90": cls._percentile(entropy_per_sample, 0.9),
                f"{prefix}_effective_experts": torch.exp(entropy_per_sample).mean(),
                f"{prefix}_top1_prob": sorted_probs[:, 0].mean(),
                f"{prefix}_top2_mass": sorted_probs[:, :top2_count].sum(dim=-1).mean(),
            }
            if support_mask is not None:
                metrics[f"{prefix}_support_size"] = support_mask.to(dtype=safe_probs.dtype).sum(dim=-1).mean()
            for expert_idx in range(safe_probs.shape[-1]):
                metrics[f"{prefix}_usage_{expert_idx}"] = usage[expert_idx]
            return metrics

    def _pool_target_probs(self, expert_action_losses, trajectory_ids, utility_scores=None, utility_candidate_mask=None):
        if expert_action_losses is None or trajectory_ids is None:
            return None
        posterior_probs = posterior_from_expert_losses(
            expert_action_losses,
            temperature=self.lara_moe.posterior_temperature,
            uniform_floor=self.lara_moe.posterior_uniform_floor,
            top_r=self.lara_moe.posterior_top_r,
        )
        trajectory_tensor = self._trajectory_ids_to_tensor(trajectory_ids, device=expert_action_losses.device)
        return aggregate_episode_responsibilities(
            posterior_probs,
            trajectory_tensor,
            avg_weight=self.config.framework.action_model.get("lara_pool_target_avg_weight", 1.0),
            max_weight=self.config.framework.action_model.get("lara_pool_target_max_weight", 1.0),
            utility_scores=utility_scores,
            utility_weight=self.config.framework.action_model.get("lara_pool_target_utility_weight", 0.0),
            utility_candidate_mask=utility_candidate_mask,
        )

    def _expert_action_losses(self, conditioning_tokens, actions_target, state=None, action_mask=None) -> torch.Tensor:
        if self.lara_moe is None:
            raise RuntimeError("_expert_action_losses requires lara_moe")
        expert_tokens = self.lara_moe.expert_conditioning_tokens(conditioning_tokens)
        batch_size, num_experts, token_count, hidden_size = expert_tokens.shape
        flat_tokens = expert_tokens.reshape(batch_size * num_experts, token_count, hidden_size)
        flat_actions = actions_target[:, None, :, :].expand(-1, num_experts, -1, -1)
        flat_actions = flat_actions.reshape(batch_size * num_experts, actions_target.shape[1], actions_target.shape[2])
        flat_action_mask = None
        if action_mask is not None:
            flat_action_mask = action_mask[:, None, :].expand(-1, num_experts, -1)
            flat_action_mask = flat_action_mask.reshape(batch_size * num_experts, actions_target.shape[1])

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
            action_mask=flat_action_mask,
        )
        return losses.view(batch_size, num_experts)

    @torch.no_grad()
    def select_resident_pool(
        self,
        embodied_action_tokens: torch.Tensor,
        latent_action_tokens: Optional[torch.Tensor] = None,
        initial_context_tokens: Optional[torch.Tensor] = None,
    ):
        if self.lara_moe is None:
            raise RuntimeError("select_resident_pool requires use_lara_moe=True")
        if self.latent_action_head is not None and latent_action_tokens is None:
            latent_action_tokens = self.latent_action_head.predict(embodied_action_tokens)
        conditioning_tokens = self._conditioning_tokens(embodied_action_tokens, latent_action_tokens)
        initial_context_tokens = (
            self._as_tensor(initial_context_tokens, device=conditioning_tokens.device, dtype=conditioning_tokens.dtype)
            if initial_context_tokens is not None
            else None
        )
        return self.lara_moe.select_resident_pool(
            conditioning_tokens,
            initial_context_tokens=initial_context_tokens,
        )

    @torch.no_grad()
    def predict_action(
        self,
        embodied_action_tokens: torch.Tensor,
        state: Optional[np.ndarray] = None,
        latent_action_tokens: Optional[torch.Tensor] = None,
        initial_context_tokens: Optional[torch.Tensor] = None,
        pool_mask: Optional[torch.Tensor] = None,
        previous_router_probs: Optional[torch.Tensor] = None,
        forced_router_probs: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict:
        if self.latent_action_head is not None:
            latent_action_tokens = self.latent_action_head.predict(embodied_action_tokens)
        conditioning_tokens = self._conditioning_tokens(embodied_action_tokens, latent_action_tokens)
        state_tensor = self._state_to_tensor(state, embodied_action_tokens)
        moe_output = None
        if self.lara_moe is not None:
            initial_context_tokens = (
                self._as_tensor(initial_context_tokens, device=conditioning_tokens.device, dtype=conditioning_tokens.dtype)
                if initial_context_tokens is not None
                else None
            )
            pool_mask = (
                self._as_tensor(pool_mask, device=conditioning_tokens.device, dtype=torch.bool)
                if pool_mask is not None
                else None
            )
            previous_router_probs = (
                self._as_tensor(previous_router_probs, device=conditioning_tokens.device, dtype=conditioning_tokens.dtype)
                if previous_router_probs is not None
                else None
            )
            forced_router_probs = (
                self._as_tensor(forced_router_probs, device=conditioning_tokens.device, dtype=conditioning_tokens.dtype)
                if forced_router_probs is not None
                else None
            )
            moe_output = self.lara_moe(
                conditioning_tokens,
                initial_context_tokens=initial_context_tokens,
                pool_mask=pool_mask,
                previous_router_probs=previous_router_probs,
                forced_router_probs=forced_router_probs,
            )
            if self.use_direct_action_output:
                direct_expert_actions, _, _, _ = self._direct_expert_action_bank(conditioning_tokens, state_tensor)
                pred_actions = ActionChunkExpertBank.routed_actions(direct_expert_actions, moe_output.router_probs)
                if return_aux:
                    return {
                        "actions": pred_actions,
                        "router_probs": moe_output.router_probs,
                        "pool_mask": moe_output.pool_mask,
                        "active_mask": moe_output.active_mask,
                    }
                return pred_actions
            conditioning_tokens = moe_output.tokens
        pred_actions = self.action_model.predict_action(conditioning_tokens, state_tensor)
        if return_aux:
            output = {"actions": pred_actions}
            if moe_output is not None:
                output.update(
                    {
                        "router_probs": moe_output.router_probs,
                        "pool_mask": moe_output.pool_mask,
                        "active_mask": moe_output.active_mask,
                    }
                )
            return output
        return pred_actions
