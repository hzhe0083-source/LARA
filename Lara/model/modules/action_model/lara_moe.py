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
    router_entropy: torch.Tensor
    posterior_entropy: torch.Tensor
    pool_entropy: torch.Tensor
    router_probs: torch.Tensor
    posterior_probs: torch.Tensor
    pool_probs: torch.Tensor
    pool_mask: torch.Tensor
    active_mask: torch.Tensor


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


def entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1).mean()


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

        if latent_action_tokens is not None:
            posterior_logits = self.posterior(conditioning_tokens, latent_action_tokens)
            posterior_probs = torch.softmax(posterior_logits, dim=-1)
            route_loss = masked_kl_div(router_logits, posterior_probs, pool_mask)
            pool_loss = F.kl_div(
                F.log_softmax(pool_logits, dim=-1),
                posterior_probs.detach(),
                reduction="batchmean",
            )
            train_weights = posterior_probs
        else:
            posterior_probs = router_probs.detach()
            route_loss = router_logits.new_zeros(())
            pool_loss = router_logits.new_zeros(())
            train_weights = router_probs

        weights = train_weights if self.training and latent_action_tokens is not None else router_probs
        tokens = conditioning_tokens + self._expert_residual(conditioning_tokens, weights)
        total_loss = self.router_loss_weight * route_loss + self.pool_loss_weight * pool_loss
        return MoEConditionerOutput(
            tokens=tokens,
            loss=total_loss,
            route_loss=route_loss.detach(),
            pool_loss=pool_loss.detach(),
            router_entropy=entropy(router_probs).detach(),
            posterior_entropy=entropy(posterior_probs).detach(),
            pool_entropy=entropy(pool_probs).detach(),
            router_probs=router_probs.detach(),
            posterior_probs=posterior_probs.detach(),
            pool_probs=pool_probs.detach(),
            pool_mask=pool_mask.detach(),
            active_mask=active_mask.detach(),
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
