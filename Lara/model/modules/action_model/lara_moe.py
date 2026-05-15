from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MoEConditionerOutput:
    tokens: torch.Tensor
    loss: torch.Tensor
    router_entropy: torch.Tensor
    posterior_entropy: torch.Tensor


def masked_topk_softmax(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.shape[-1]:
        return torch.softmax(logits, dim=-1)
    top_values, top_indices = torch.topk(logits, k=top_k, dim=-1)
    masked_logits = torch.full_like(logits, torch.finfo(logits.dtype).min)
    masked_logits.scatter_(dim=-1, index=top_indices, src=top_values)
    return torch.softmax(masked_logits, dim=-1)


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
        expert_hidden_size: int = 1024,
        router_hidden_size: int = 1024,
        router_loss_weight: float = 1.0,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_loss_weight = router_loss_weight
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

    def forward(
        self,
        conditioning_tokens: torch.Tensor,
        latent_action_tokens: Optional[torch.Tensor] = None,
    ) -> MoEConditionerOutput:
        router_logits = self.chunk_router(conditioning_tokens)
        router_probs = masked_topk_softmax(router_logits, self.top_k)

        if latent_action_tokens is not None:
            posterior_logits = self.posterior(conditioning_tokens, latent_action_tokens)
            posterior_probs = torch.softmax(posterior_logits, dim=-1)
            route_loss = F.kl_div(
                F.log_softmax(router_logits, dim=-1),
                posterior_probs.detach(),
                reduction="batchmean",
            )
            train_weights = posterior_probs
        else:
            posterior_probs = router_probs.detach()
            route_loss = router_logits.new_zeros(())
            train_weights = router_probs

        weights = train_weights if self.training and latent_action_tokens is not None else router_probs
        tokens = conditioning_tokens + self._expert_residual(conditioning_tokens, weights)
        return MoEConditionerOutput(
            tokens=tokens,
            loss=self.router_loss_weight * route_loss,
            router_entropy=entropy(router_probs).detach(),
            posterior_entropy=entropy(posterior_probs).detach(),
        )

    @torch.no_grad()
    def predict(self, conditioning_tokens: torch.Tensor) -> torch.Tensor:
        return self.forward(conditioning_tokens, latent_action_tokens=None).tokens
