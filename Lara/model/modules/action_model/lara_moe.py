from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MoEConditionerOutput:
    tokens: torch.Tensor
    loss: torch.Tensor
    route_loss: torch.Tensor
    pool_loss: torch.Tensor
    utility_loss: torch.Tensor
    utility_rank_loss: torch.Tensor
    utility_head_loss: torch.Tensor
    balance_loss: torch.Tensor
    stickiness_loss: torch.Tensor
    utility_calibration_error: torch.Tensor
    utility_scores: Optional[torch.Tensor]
    utility_value_scores: Optional[torch.Tensor]
    utility_progress_scores: Optional[torch.Tensor]
    utility_uncertainty_scores: Optional[torch.Tensor]
    router_entropy: torch.Tensor
    posterior_entropy: torch.Tensor
    pool_entropy: torch.Tensor
    router_probs: torch.Tensor
    posterior_probs: torch.Tensor
    pool_probs: torch.Tensor
    pool_mask: torch.Tensor
    active_mask: torch.Tensor
    active_usage: torch.Tensor
    pool_usage: torch.Tensor
    dead_expert_ratio: torch.Tensor
    pool_dead_expert_ratio: torch.Tensor
    route_top1_match: torch.Tensor
    route_regret: torch.Tensor


def _validate_expert_mask(mask: torch.Tensor, logits: torch.Tensor, mask_name: str) -> torch.Tensor:
    if mask.shape != logits.shape:
        raise ValueError(f"{mask_name} must have shape {tuple(logits.shape)}, got {tuple(mask.shape)}")
    mask = mask.to(device=logits.device, dtype=torch.bool)
    if not torch.all(mask.any(dim=-1)):
        raise ValueError(f"{mask_name} must select at least one expert for every sample")
    return mask


def masked_softmax(logits: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is None:
        return torch.softmax(logits, dim=-1)
    mask = _validate_expert_mask(mask, logits, "mask")
    masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    return torch.softmax(masked_logits, dim=-1)


def topk_mask(logits: torch.Tensor, top_k: int, allowed_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if allowed_mask is None:
        allowed_mask = torch.ones_like(logits, dtype=torch.bool)
    else:
        allowed_mask = _validate_expert_mask(allowed_mask, logits, "allowed_mask")

    if top_k <= 0 or top_k >= logits.shape[-1]:
        return allowed_mask

    masked_logits = logits.masked_fill(~allowed_mask, torch.finfo(logits.dtype).min)
    _, top_indices = torch.topk(masked_logits, k=top_k, dim=-1)
    mask = torch.zeros_like(allowed_mask, dtype=torch.bool)
    mask.scatter_(dim=-1, index=top_indices, value=True)
    mask = mask & allowed_mask
    return _validate_expert_mask(mask, logits, "topk_mask")


def masked_topk_softmax(
    logits: torch.Tensor,
    top_k: int,
    allowed_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    mask = topk_mask(logits, top_k=top_k, allowed_mask=allowed_mask)
    return masked_softmax(logits, mask)


def renormalize_probs(probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = _validate_expert_mask(mask, probs, "mask")
    masked_probs = probs * mask.to(dtype=probs.dtype)
    denom = masked_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return masked_probs / denom


def masked_kl_div(student_logits: torch.Tensor, teacher_probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    teacher_probs = renormalize_probs(teacher_probs, mask)
    student_probs = masked_softmax(student_logits, mask)
    return F.kl_div(
        torch.log(student_probs.clamp_min(1e-8)),
        teacher_probs.detach(),
        reduction="batchmean",
    )


def posterior_from_expert_losses(
    expert_losses: torch.Tensor,
    temperature: float = 1.0,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if expert_losses.ndim != 2:
        raise ValueError(f"Expected expert_losses [B, M], got {tuple(expert_losses.shape)}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = -expert_losses / temperature
    return masked_softmax(logits, mask)


def entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1).mean()


def route_diagnostics(
    router_probs: torch.Tensor,
    posterior_probs: torch.Tensor,
    pool_mask: torch.Tensor,
    active_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    active_usage = active_mask.float().mean(dim=0)
    pool_usage = pool_mask.float().mean(dim=0)
    dead_expert_ratio = (active_usage == 0).float().mean()
    pool_dead_expert_ratio = (pool_usage == 0).float().mean()
    route_top1_match = (router_probs.argmax(dim=-1) == posterior_probs.argmax(dim=-1)).float().mean()
    selected_value = (posterior_probs * active_mask.float()).sum(dim=-1)
    best_value = posterior_probs.max(dim=-1).values
    route_regret = (best_value - selected_value).clamp_min(0.0).mean()
    return {
        "active_usage": active_usage,
        "pool_usage": pool_usage,
        "dead_expert_ratio": dead_expert_ratio,
        "pool_dead_expert_ratio": pool_dead_expert_ratio,
        "route_top1_match": route_top1_match,
        "route_regret": route_regret,
    }


def uniform_balance_loss(probs: torch.Tensor) -> torch.Tensor:
    if probs.ndim != 2:
        raise ValueError(f"Expected probs [B, M], got {tuple(probs.shape)}")
    mean_probs = probs.mean(dim=0).clamp_min(1e-8)
    uniform = torch.full_like(mean_probs, 1.0 / probs.shape[-1])
    return F.kl_div(torch.log(mean_probs), uniform, reduction="sum")


def route_stickiness_loss(router_probs: torch.Tensor, previous_router_probs: torch.Tensor) -> torch.Tensor:
    if router_probs.shape != previous_router_probs.shape:
        raise ValueError(
            f"previous_router_probs must have shape {tuple(router_probs.shape)}, "
            f"got {tuple(previous_router_probs.shape)}"
        )
    return torch.mean(torch.abs(router_probs - previous_router_probs.to(router_probs.device, router_probs.dtype)))


def route_switch_rate(route_ids: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if route_ids.ndim != 2:
        raise ValueError(f"Expected route_ids [B, T], got {tuple(route_ids.shape)}")
    if route_ids.shape[1] < 2:
        return route_ids.new_zeros((), dtype=torch.float32)

    switches = (route_ids[:, 1:] != route_ids[:, :-1]).float()
    if valid_mask is None:
        return switches.mean()
    if valid_mask.shape != route_ids.shape:
        raise ValueError(f"valid_mask must have shape {tuple(route_ids.shape)}, got {tuple(valid_mask.shape)}")
    pair_mask = (valid_mask[:, 1:] & valid_mask[:, :-1]).to(dtype=switches.dtype)
    return (switches * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)


def retained_probability_mass(probs: torch.Tensor, retention_fractions: list[float]) -> dict[float, torch.Tensor]:
    if probs.ndim != 2:
        raise ValueError(f"Expected probs [B, M], got {tuple(probs.shape)}")
    if not retention_fractions:
        raise ValueError("retention_fractions must not be empty")

    results = {}
    num_experts = probs.shape[-1]
    for fraction in retention_fractions:
        if fraction <= 0 or fraction > 1:
            raise ValueError(f"retention fraction must be in (0, 1], got {fraction}")
        keep_count = max(1, min(num_experts, int(round(num_experts * fraction))))
        top_values, _ = torch.topk(probs, k=keep_count, dim=-1)
        results[fraction] = top_values.sum(dim=-1).mean()
    return results


def sparse_route_budget(
    total_experts: int,
    active_experts: int,
    resident_experts: Optional[int] = None,
    shared_params: int = 0,
    params_per_expert: int = 0,
) -> dict[str, float]:
    if total_experts <= 0:
        raise ValueError("total_experts must be positive")
    if active_experts <= 0 or active_experts > total_experts:
        raise ValueError("active_experts must be in [1, total_experts]")
    resident_experts = total_experts if resident_experts is None else resident_experts
    if resident_experts <= 0 or resident_experts > total_experts:
        raise ValueError("resident_experts must be in [1, total_experts]")
    if active_experts > resident_experts:
        raise ValueError("active_experts cannot exceed resident_experts")
    if shared_params < 0 or params_per_expert < 0:
        raise ValueError("parameter counts must be non-negative")

    active_params = shared_params + active_experts * params_per_expert
    resident_params = shared_params + resident_experts * params_per_expert
    total_params = shared_params + total_experts * params_per_expert
    return {
        "active_expert_fraction": active_experts / total_experts,
        "resident_expert_fraction": resident_experts / total_experts,
        "active_params": float(active_params),
        "resident_params": float(resident_params),
        "total_params": float(total_params),
        "active_param_fraction": active_params / total_params if total_params > 0 else 0.0,
        "resident_param_fraction": resident_params / total_params if total_params > 0 else 0.0,
    }


def candidate_route_utility(
    value_scores: Optional[torch.Tensor] = None,
    progress_scores: Optional[torch.Tensor] = None,
    uncertainty_scores: Optional[torch.Tensor] = None,
    cost_scores: Optional[torch.Tensor] = None,
    progress_weight: float = 1.0,
    uncertainty_weight: float = 1.0,
    cost_weight: float = 1.0,
) -> torch.Tensor:
    components = [value_scores, progress_scores, uncertainty_scores, cost_scores]
    reference = next((component for component in components if component is not None), None)
    if reference is None:
        raise ValueError("At least one utility component must be provided")
    if reference.ndim != 2:
        raise ValueError(f"Expected utility components [B, M], got {tuple(reference.shape)}")
    for component in components:
        if component is not None and component.shape != reference.shape:
            raise ValueError(
                f"All utility components must share shape {tuple(reference.shape)}, got {tuple(component.shape)}"
            )
    for name, weight in [
        ("progress_weight", progress_weight),
        ("uncertainty_weight", uncertainty_weight),
        ("cost_weight", cost_weight),
    ]:
        if weight < 0:
            raise ValueError(f"{name} must be non-negative")

    utility = torch.zeros_like(reference)
    if value_scores is not None:
        utility = utility + value_scores
    if progress_scores is not None:
        utility = utility + progress_weight * progress_scores
    if uncertainty_scores is not None:
        utility = utility - uncertainty_weight * uncertainty_scores
    if cost_scores is not None:
        utility = utility - cost_weight * cost_scores
    return utility


def aggregate_episode_responsibilities(
    posterior_probs: torch.Tensor,
    episode_ids: torch.Tensor,
) -> torch.Tensor:
    if posterior_probs.ndim != 2:
        raise ValueError(f"Expected posterior_probs [B, M], got {tuple(posterior_probs.shape)}")
    if episode_ids.ndim != 1 or episode_ids.shape[0] != posterior_probs.shape[0]:
        raise ValueError(
            f"Expected episode_ids [B] with B={posterior_probs.shape[0]}, got {tuple(episode_ids.shape)}"
        )

    episode_ids = episode_ids.to(device=posterior_probs.device)
    targets = torch.zeros_like(posterior_probs)
    for episode_id in torch.unique(episode_ids):
        mask = episode_ids == episode_id
        targets[mask] = posterior_probs[mask].mean(dim=0, keepdim=True)
    return targets / targets.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def centered_utility_targets(
    utility_scores: torch.Tensor,
    candidate_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if utility_scores.ndim != 2:
        raise ValueError(f"Expected utility_scores [B, M], got {tuple(utility_scores.shape)}")
    if candidate_mask is None:
        candidate_mask = torch.ones_like(utility_scores, dtype=torch.bool)
    else:
        candidate_mask = _validate_expert_mask(candidate_mask, utility_scores, "candidate_mask")

    mask_float = candidate_mask.to(dtype=utility_scores.dtype)
    mean_score = (utility_scores * mask_float).sum(dim=-1, keepdim=True)
    mean_score = mean_score / mask_float.sum(dim=-1, keepdim=True).clamp_min(1.0)
    targets = (utility_scores - mean_score) * mask_float
    return targets, candidate_mask


def utility_calibration_objective(
    router_logits: torch.Tensor,
    utility_scores: torch.Tensor,
    candidate_mask: Optional[torch.Tensor] = None,
    rank_loss_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if router_logits.shape != utility_scores.shape:
        raise ValueError(
            f"utility_scores must have shape {tuple(router_logits.shape)}, got {tuple(utility_scores.shape)}"
        )
    if rank_loss_weight < 0:
        raise ValueError("rank_loss_weight must be non-negative")

    targets, candidate_mask = centered_utility_targets(utility_scores, candidate_mask)
    regression_loss = F.smooth_l1_loss(router_logits[candidate_mask], targets.detach()[candidate_mask])
    calibration_error = (router_logits.detach()[candidate_mask] - targets.detach()[candidate_mask]).abs().mean()

    if rank_loss_weight == 0:
        rank_loss = router_logits.new_zeros(())
        return regression_loss, rank_loss, calibration_error

    utility_diff = targets[:, :, None] - targets[:, None, :]
    logit_diff = router_logits[:, :, None] - router_logits[:, None, :]
    pair_mask = candidate_mask[:, :, None] & candidate_mask[:, None, :] & (utility_diff.abs() > 1e-8)
    if not torch.any(pair_mask):
        rank_loss = router_logits.new_zeros(())
    else:
        rank_loss = F.softplus(-(utility_diff.detach() * logit_diff)[pair_mask]).mean()
    return regression_loss + rank_loss_weight * rank_loss, rank_loss, calibration_error


def utility_component_supervision_loss(
    value_scores: torch.Tensor,
    progress_scores: torch.Tensor,
    uncertainty_scores: torch.Tensor,
    value_targets: Optional[torch.Tensor] = None,
    progress_targets: Optional[torch.Tensor] = None,
    uncertainty_targets: Optional[torch.Tensor] = None,
    target_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    predictions_and_targets = [
        (value_scores, value_targets),
        (progress_scores, progress_targets),
        (uncertainty_scores, uncertainty_targets),
    ]
    reference = value_scores
    for prediction, target in predictions_and_targets:
        if prediction.shape != reference.shape:
            raise ValueError(f"Utility predictions must share shape {tuple(reference.shape)}")
        if target is not None and target.shape != reference.shape:
            raise ValueError(f"Utility targets must have shape {tuple(reference.shape)}, got {tuple(target.shape)}")
    if target_mask is None:
        target_mask = torch.ones_like(reference, dtype=torch.bool)
    else:
        target_mask = _validate_expert_mask(target_mask, reference, "target_mask")

    losses = []
    for prediction, target in predictions_and_targets:
        if target is not None:
            losses.append(F.smooth_l1_loss(prediction[target_mask], target.detach()[target_mask]))
    if not losses:
        return reference.new_zeros(())
    return torch.stack(losses).mean()


class ResidualActionExpert(nn.Module):
    """Small token adapter used as a lightweight action expert."""

    def __init__(self, hidden_size: int, expert_hidden_size: int, residual_scale: float):
        super().__init__()
        self.residual_scale = residual_scale
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, expert_hidden_size),
            nn.GELU(),
            nn.Linear(expert_hidden_size, hidden_size),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.residual_scale * self.net(tokens)


class ActionChunkExpert(nn.Module):
    """Expert head that directly reconstructs a future action chunk."""

    def __init__(self, hidden_size: int, expert_hidden_size: int, action_horizon: int, action_dim: int):
        super().__init__()
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.context_norm = nn.LayerNorm(hidden_size)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, expert_hidden_size),
            nn.GELU(),
            nn.Linear(expert_hidden_size, action_horizon * action_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens [B, T, D], got {tuple(tokens.shape)}")
        context_summary = self.context_norm(tokens).mean(dim=1)
        actions = self.net(context_summary)
        return actions.view(tokens.shape[0], self.action_horizon, self.action_dim)


class ActionChunkExpertBank(nn.Module):
    """Bank of direct action-chunk experts for posterior responsibility training."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        expert_hidden_size: int,
        action_horizon: int,
        action_dim: int,
    ):
        super().__init__()
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.experts = nn.ModuleList(
            [
                ActionChunkExpert(
                    hidden_size=hidden_size,
                    expert_hidden_size=expert_hidden_size,
                    action_horizon=action_horizon,
                    action_dim=action_dim,
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.stack([expert(tokens) for expert in self.experts], dim=1)

    @staticmethod
    def reconstruction_losses(
        pred_actions: torch.Tensor,
        target_actions: torch.Tensor,
        execution_horizon: Optional[int] = None,
        execution_loss_weight: float = 1.0,
        prediction_loss_weight: float = 1.0,
    ) -> torch.Tensor:
        if pred_actions.ndim != 4:
            raise ValueError(f"Expected pred_actions [B, M, H, D], got {tuple(pred_actions.shape)}")
        if target_actions.ndim != 3:
            raise ValueError(f"Expected target_actions [B, H, D], got {tuple(target_actions.shape)}")
        if pred_actions.shape[0] != target_actions.shape[0] or pred_actions.shape[2:] != target_actions.shape[1:]:
            raise ValueError(
                "pred_actions and target_actions must agree on batch, horizon, and action dim: "
                f"{tuple(pred_actions.shape)} vs {tuple(target_actions.shape)}"
            )

        loss = (pred_actions - target_actions[:, None, :, :]) ** 2
        if execution_loss_weight != prediction_loss_weight:
            horizon = loss.shape[2]
            execution_horizon = horizon if execution_horizon is None else min(execution_horizon, horizon)
            weights = torch.full(
                (1, 1, horizon, 1),
                prediction_loss_weight,
                device=loss.device,
                dtype=loss.dtype,
            )
            weights[:, :, :execution_horizon, :] = execution_loss_weight
            loss = loss * weights
        return loss.mean(dim=(2, 3))


class PosteriorResponsibilityHead(nn.Module):
    """Training-time expert responsibility estimator from context and latent action."""

    def __init__(self, hidden_size: int, num_experts: int, router_hidden_size: int):
        super().__init__()
        self.context_norm = nn.LayerNorm(hidden_size)
        self.latent_norm = nn.LayerNorm(hidden_size)
        self.net = nn.Sequential(
            nn.Linear(hidden_size * 2, router_hidden_size),
            nn.GELU(),
            nn.Linear(router_hidden_size, num_experts),
        )

    def forward(self, context_tokens: torch.Tensor, latent_action_tokens: torch.Tensor) -> torch.Tensor:
        context_summary = self.context_norm(context_tokens).mean(dim=1)
        latent_summary = self.latent_norm(latent_action_tokens).mean(dim=1)
        return self.net(torch.cat([context_summary, latent_summary], dim=-1))


class ChunkRouter(nn.Module):
    """Context-only chunk router distilled from posterior responsibility."""

    def __init__(self, hidden_size: int, num_experts: int, router_hidden_size: int):
        super().__init__()
        self.context_norm = nn.LayerNorm(hidden_size)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, router_hidden_size),
            nn.GELU(),
            nn.Linear(router_hidden_size, num_experts),
        )

    def forward(self, context_tokens: torch.Tensor) -> torch.Tensor:
        context_summary = self.context_norm(context_tokens).mean(dim=1)
        return self.net(context_summary)


class EpisodePoolRouter(nn.Module):
    """Episode-level router for selecting a resident expert pool."""

    def __init__(self, hidden_size: int, num_experts: int, router_hidden_size: int):
        super().__init__()
        self.context_norm = nn.LayerNorm(hidden_size)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, router_hidden_size),
            nn.GELU(),
            nn.Linear(router_hidden_size, num_experts),
        )

    def forward(self, initial_context_tokens: torch.Tensor) -> torch.Tensor:
        context_summary = self.context_norm(initial_context_tokens).mean(dim=1)
        return self.net(context_summary)


class RouteUtilityHead(nn.Module):
    """Context-only producer for candidate route utility components."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        utility_hidden_size: int,
        progress_weight: float = 1.0,
        uncertainty_weight: float = 1.0,
        cost_weight: float = 1.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.progress_weight = progress_weight
        self.uncertainty_weight = uncertainty_weight
        self.cost_weight = cost_weight
        self.context_norm = nn.LayerNorm(hidden_size)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, utility_hidden_size),
            nn.GELU(),
            nn.Linear(utility_hidden_size, num_experts * 3),
        )

    def forward(
        self,
        conditioning_tokens: torch.Tensor,
        cost_scores: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if conditioning_tokens.ndim != 3:
            raise ValueError(f"Expected conditioning_tokens [B, T, D], got {tuple(conditioning_tokens.shape)}")
        context_summary = self.context_norm(conditioning_tokens).mean(dim=1)
        raw = self.net(context_summary).view(conditioning_tokens.shape[0], 3, self.num_experts)
        value_scores = raw[:, 0, :]
        progress_scores = raw[:, 1, :]
        uncertainty_scores = F.softplus(raw[:, 2, :])
        utility_scores = candidate_route_utility(
            value_scores=value_scores,
            progress_scores=progress_scores,
            uncertainty_scores=uncertainty_scores,
            cost_scores=cost_scores,
            progress_weight=self.progress_weight,
            uncertainty_weight=self.uncertainty_weight,
            cost_weight=self.cost_weight,
        )
        return {
            "utility_scores": utility_scores,
            "value_scores": value_scores,
            "progress_scores": progress_scores,
            "uncertainty_scores": uncertainty_scores,
        }


class LatentActionMoE(nn.Module):
    """Optional Stage-2 MoE conditioner for latent-action routed action decoding.

    This is a lightweight implementation hook: experts adapt the conditioning
    token stream before the flow action decoder. During training, posterior
    responsibility can use latent action tokens; at deployment, the context-only
    chunk router selects sparse experts.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 8,
        top_k: int = 2,
        episode_pool_size: Optional[int] = None,
        expert_hidden_size: int = 1024,
        router_hidden_size: int = 1024,
        router_loss_weight: float = 1.0,
        pool_loss_weight: float = 1.0,
        utility_loss_weight: float = 0.0,
        utility_rank_loss_weight: float = 0.0,
        utility_head_loss_weight: float = 0.0,
        balance_loss_weight: float = 0.0,
        stickiness_loss_weight: float = 0.0,
        use_utility_head: bool = False,
        utility_hidden_size: Optional[int] = None,
        utility_progress_weight: float = 1.0,
        utility_uncertainty_weight: float = 1.0,
        utility_cost_weight: float = 1.0,
        posterior_temperature: float = 1.0,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        self.num_experts = num_experts
        self.top_k = top_k
        self.episode_pool_size = episode_pool_size if episode_pool_size is not None else num_experts
        self.router_loss_weight = router_loss_weight
        self.pool_loss_weight = pool_loss_weight
        self.utility_loss_weight = utility_loss_weight
        self.utility_rank_loss_weight = utility_rank_loss_weight
        self.utility_head_loss_weight = utility_head_loss_weight
        self.balance_loss_weight = balance_loss_weight
        self.stickiness_loss_weight = stickiness_loss_weight
        self.posterior_temperature = posterior_temperature
        self.utility_head = (
            RouteUtilityHead(
                hidden_size=hidden_size,
                num_experts=num_experts,
                utility_hidden_size=utility_hidden_size if utility_hidden_size is not None else router_hidden_size,
                progress_weight=utility_progress_weight,
                uncertainty_weight=utility_uncertainty_weight,
                cost_weight=utility_cost_weight,
            )
            if use_utility_head
            else None
        )
        self.experts = nn.ModuleList(
            [
                ResidualActionExpert(
                    hidden_size=hidden_size,
                    expert_hidden_size=expert_hidden_size,
                    residual_scale=residual_scale,
                )
                for _ in range(num_experts)
            ]
        )
        self.posterior = PosteriorResponsibilityHead(
            hidden_size=hidden_size,
            num_experts=num_experts,
            router_hidden_size=router_hidden_size,
        )
        self.chunk_router = ChunkRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            router_hidden_size=router_hidden_size,
        )
        self.pool_router = EpisodePoolRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            router_hidden_size=router_hidden_size,
        )

    def _expert_residual(self, tokens: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        expert_outputs = torch.stack([expert(tokens) for expert in self.experts], dim=1)
        return torch.sum(expert_outputs * weights[:, :, None, None], dim=1)

    def expert_conditioning_tokens(self, conditioning_tokens: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(conditioning_tokens, latent_action_tokens=None)
        expert_outputs = torch.stack([expert(conditioning_tokens) for expert in self.experts], dim=1)
        return conditioning_tokens[:, None, :, :] + expert_outputs

    def _validate_inputs(
        self,
        conditioning_tokens: torch.Tensor,
        latent_action_tokens: Optional[torch.Tensor],
    ) -> None:
        if conditioning_tokens.ndim != 3:
            raise ValueError(f"Expected conditioning_tokens [B, T, D], got {tuple(conditioning_tokens.shape)}")
        if latent_action_tokens is None:
            return
        if latent_action_tokens.ndim != 3:
            raise ValueError(f"Expected latent_action_tokens [B, T, D], got {tuple(latent_action_tokens.shape)}")
        if latent_action_tokens.shape[0] != conditioning_tokens.shape[0]:
            raise ValueError(
                "latent_action_tokens batch size must match conditioning_tokens: "
                f"{latent_action_tokens.shape[0]} != {conditioning_tokens.shape[0]}"
            )
        if latent_action_tokens.shape[-1] != conditioning_tokens.shape[-1]:
            raise ValueError(
                "latent_action_tokens hidden size must match conditioning_tokens: "
                f"{latent_action_tokens.shape[-1]} != {conditioning_tokens.shape[-1]}"
            )

    def _resident_pool(
        self,
        conditioning_tokens: torch.Tensor,
        initial_context_tokens: Optional[torch.Tensor],
        pool_mask: Optional[torch.Tensor],
    ):
        pool_context = initial_context_tokens if initial_context_tokens is not None else conditioning_tokens
        pool_logits = self.pool_router(pool_context)
        if pool_mask is None:
            pool_mask = topk_mask(pool_logits, top_k=self.episode_pool_size)
        else:
            pool_mask = _validate_expert_mask(pool_mask, pool_logits, "pool_mask")
        pool_probs = masked_softmax(pool_logits, pool_mask)
        return pool_logits, pool_probs, pool_mask

    def forward(
        self,
        conditioning_tokens: torch.Tensor,
        latent_action_tokens: Optional[torch.Tensor] = None,
        initial_context_tokens: Optional[torch.Tensor] = None,
        pool_mask: Optional[torch.Tensor] = None,
        expert_action_losses: Optional[torch.Tensor] = None,
        pool_target_probs: Optional[torch.Tensor] = None,
        utility_scores: Optional[torch.Tensor] = None,
        utility_candidate_mask: Optional[torch.Tensor] = None,
        utility_cost_scores: Optional[torch.Tensor] = None,
        utility_value_targets: Optional[torch.Tensor] = None,
        utility_progress_targets: Optional[torch.Tensor] = None,
        utility_uncertainty_targets: Optional[torch.Tensor] = None,
        utility_target_mask: Optional[torch.Tensor] = None,
        previous_router_probs: Optional[torch.Tensor] = None,
    ) -> MoEConditionerOutput:
        self._validate_inputs(conditioning_tokens, latent_action_tokens)
        if initial_context_tokens is not None and initial_context_tokens.shape[0] != conditioning_tokens.shape[0]:
            raise ValueError(
                "initial_context_tokens batch size must match conditioning_tokens: "
                f"{initial_context_tokens.shape[0]} != {conditioning_tokens.shape[0]}"
            )

        pool_logits, pool_probs, pool_mask = self._resident_pool(
            conditioning_tokens=conditioning_tokens,
            initial_context_tokens=initial_context_tokens,
            pool_mask=pool_mask,
        )
        router_logits = self.chunk_router(conditioning_tokens)
        active_mask = topk_mask(router_logits, top_k=self.top_k, allowed_mask=pool_mask)
        router_probs = masked_softmax(router_logits, active_mask)

        if expert_action_losses is not None:
            if expert_action_losses.shape != router_logits.shape:
                raise ValueError(
                    f"expert_action_losses must have shape {tuple(router_logits.shape)}, "
                    f"got {tuple(expert_action_losses.shape)}"
                )
            posterior_probs = posterior_from_expert_losses(
                expert_action_losses,
                temperature=self.posterior_temperature,
            )
            route_loss = masked_kl_div(router_logits, posterior_probs, pool_mask)
            pool_teacher = pool_target_probs if pool_target_probs is not None else posterior_probs
            pool_loss = F.kl_div(F.log_softmax(pool_logits, dim=-1), pool_teacher.detach(), reduction="batchmean")
            train_weights = posterior_probs
        elif latent_action_tokens is not None:
            posterior_logits = self.posterior(conditioning_tokens, latent_action_tokens)
            posterior_probs = torch.softmax(posterior_logits, dim=-1)
            route_loss = masked_kl_div(router_logits, posterior_probs, pool_mask)
            pool_teacher = pool_target_probs if pool_target_probs is not None else posterior_probs
            pool_loss = F.kl_div(F.log_softmax(pool_logits, dim=-1), pool_teacher.detach(), reduction="batchmean")
            train_weights = posterior_probs
        else:
            posterior_probs = router_probs.detach()
            route_loss = router_logits.new_zeros(())
            pool_loss = router_logits.new_zeros(())
            train_weights = router_probs

        has_training_teacher = expert_action_losses is not None or latent_action_tokens is not None
        weights = train_weights if self.training and has_training_teacher else router_probs
        tokens = conditioning_tokens + self._expert_residual(conditioning_tokens, weights)
        utility_value_scores = None
        utility_progress_scores = None
        utility_uncertainty_scores = None
        if utility_scores is None and self.utility_head is not None:
            utility_output = self.utility_head(conditioning_tokens, cost_scores=utility_cost_scores)
            utility_scores = utility_output["utility_scores"]
            utility_value_scores = utility_output["value_scores"]
            utility_progress_scores = utility_output["progress_scores"]
            utility_uncertainty_scores = utility_output["uncertainty_scores"]
        if utility_value_scores is not None:
            utility_head_loss = utility_component_supervision_loss(
                value_scores=utility_value_scores,
                progress_scores=utility_progress_scores,
                uncertainty_scores=utility_uncertainty_scores,
                value_targets=utility_value_targets,
                progress_targets=utility_progress_targets,
                uncertainty_targets=utility_uncertainty_targets,
                target_mask=utility_target_mask,
            )
        else:
            utility_head_loss = router_logits.new_zeros(())
        if utility_scores is None:
            utility_loss = router_logits.new_zeros(())
            utility_rank_loss = router_logits.new_zeros(())
            utility_calibration_error = router_logits.new_zeros(())
        else:
            utility_loss, utility_rank_loss, utility_calibration_error = utility_calibration_objective(
                router_logits=router_logits,
                utility_scores=utility_scores,
                candidate_mask=utility_candidate_mask,
                rank_loss_weight=self.utility_rank_loss_weight,
            )
        balance_loss = uniform_balance_loss(router_probs) + uniform_balance_loss(pool_probs)
        stickiness_loss = (
            route_stickiness_loss(router_probs, previous_router_probs)
            if previous_router_probs is not None
            else router_logits.new_zeros(())
        )
        total_loss = (
            self.router_loss_weight * route_loss
            + self.pool_loss_weight * pool_loss
            + self.utility_loss_weight * utility_loss
            + self.utility_head_loss_weight * utility_head_loss
            + self.balance_loss_weight * balance_loss
            + self.stickiness_loss_weight * stickiness_loss
        )
        diagnostics = route_diagnostics(
            router_probs=router_probs.detach(),
            posterior_probs=posterior_probs.detach(),
            pool_mask=pool_mask.detach(),
            active_mask=active_mask.detach(),
        )
        return MoEConditionerOutput(
            tokens=tokens,
            loss=total_loss,
            route_loss=route_loss.detach(),
            pool_loss=pool_loss.detach(),
            utility_loss=utility_loss.detach(),
            utility_rank_loss=utility_rank_loss.detach(),
            utility_head_loss=utility_head_loss.detach(),
            balance_loss=balance_loss.detach(),
            stickiness_loss=stickiness_loss.detach(),
            utility_calibration_error=utility_calibration_error.detach(),
            utility_scores=utility_scores.detach() if utility_scores is not None else None,
            utility_value_scores=utility_value_scores.detach() if utility_value_scores is not None else None,
            utility_progress_scores=utility_progress_scores.detach() if utility_progress_scores is not None else None,
            utility_uncertainty_scores=utility_uncertainty_scores.detach()
            if utility_uncertainty_scores is not None
            else None,
            router_entropy=entropy(router_probs).detach(),
            posterior_entropy=entropy(posterior_probs).detach(),
            pool_entropy=entropy(pool_probs).detach(),
            router_probs=router_probs.detach(),
            posterior_probs=posterior_probs.detach(),
            pool_probs=pool_probs.detach(),
            pool_mask=pool_mask.detach(),
            active_mask=active_mask.detach(),
            active_usage=diagnostics["active_usage"],
            pool_usage=diagnostics["pool_usage"],
            dead_expert_ratio=diagnostics["dead_expert_ratio"],
            pool_dead_expert_ratio=diagnostics["pool_dead_expert_ratio"],
            route_top1_match=diagnostics["route_top1_match"],
            route_regret=diagnostics["route_regret"],
        )

    @torch.no_grad()
    def predict(
        self,
        conditioning_tokens: torch.Tensor,
        initial_context_tokens: Optional[torch.Tensor] = None,
        pool_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward(
            conditioning_tokens,
            latent_action_tokens=None,
            initial_context_tokens=initial_context_tokens,
            pool_mask=pool_mask,
        ).tokens
