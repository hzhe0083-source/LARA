from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LatentActionOutput:
    tokens: torch.Tensor
    loss: torch.Tensor
    vq_loss: torch.Tensor
    prior_loss: torch.Tensor
    perplexity: torch.Tensor


class PosteriorLatentActionEncoder(nn.Module):
    """Encodes demonstrated future actions into latent action tokens."""

    def __init__(
        self,
        context_dim: int,
        action_dim: int,
        action_horizon: int,
        num_latent_tokens: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.action_horizon = action_horizon
        self.num_latent_tokens = num_latent_tokens
        self.context_norm = nn.LayerNorm(context_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.action_pos = nn.Parameter(torch.zeros(1, action_horizon, hidden_dim))
        self.encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.to_latents = nn.Linear(hidden_dim, num_latent_tokens * context_dim)
        nn.init.normal_(self.action_pos, mean=0.0, std=0.02)

    def forward(self, context_tokens: torch.Tensor, future_actions: torch.Tensor) -> torch.Tensor:
        if context_tokens.ndim != 3:
            raise ValueError(f"Expected context_tokens [B, T, D], got {tuple(context_tokens.shape)}")
        if future_actions.ndim != 3:
            raise ValueError(f"Expected future_actions [B, H, A], got {tuple(future_actions.shape)}")
        if future_actions.shape[1] != self.action_horizon:
            raise ValueError(f"Expected future_actions horizon {self.action_horizon}, got {future_actions.shape[1]}")
        if future_actions.shape[0] != context_tokens.shape[0]:
            raise ValueError(
                f"Batch mismatch between context_tokens and future_actions: "
                f"{context_tokens.shape[0]} != {future_actions.shape[0]}"
            )

        batch_size = context_tokens.shape[0]
        context_summary = self.context_norm(context_tokens).mean(dim=1)
        context_features = self.context_proj(context_summary).unsqueeze(1).expand(-1, self.action_horizon, -1)
        action_features = self.action_proj(future_actions.to(dtype=context_tokens.dtype))
        action_features = action_features + self.action_pos[:, : self.action_horizon, :].to(
            device=action_features.device,
            dtype=action_features.dtype,
        )
        fused = self.encoder(torch.cat([context_features, action_features], dim=-1))
        pooled = fused.mean(dim=1)
        latents = self.to_latents(pooled)
        return latents.view(batch_size, self.num_latent_tokens, context_tokens.shape[-1])


class VectorQuantizer(nn.Module):
    """Straight-through VQ codebook for latent action tokens."""

    def __init__(self, codebook_size: int, code_dim: int, commitment_weight: float = 0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.commitment_weight = commitment_weight
        self.codebook = nn.Embedding(codebook_size, code_dim)
        nn.init.normal_(self.codebook.weight, mean=0.0, std=0.02)

    def forward(self, inputs: torch.Tensor):
        if inputs.ndim != 3:
            raise ValueError(f"Expected VQ inputs [B, L, D], got {tuple(inputs.shape)}")
        if inputs.shape[-1] != self.code_dim:
            raise ValueError(f"Expected VQ dim {self.code_dim}, got {inputs.shape[-1]}")

        flat_inputs = inputs.reshape(-1, self.code_dim)
        distances = (
            flat_inputs.float().pow(2).sum(dim=1, keepdim=True)
            - 2 * flat_inputs.float() @ self.codebook.weight.float().t()
            + self.codebook.weight.float().pow(2).sum(dim=1)
        )
        indices = distances.argmin(dim=1)
        quantized = self.codebook(indices).view_as(inputs).to(dtype=inputs.dtype)

        codebook_loss = F.mse_loss(quantized, inputs.detach())
        commitment_loss = F.mse_loss(inputs, quantized.detach())
        vq_loss = codebook_loss + self.commitment_weight * commitment_loss
        quantized = inputs + (quantized - inputs).detach()

        with torch.no_grad():
            encodings = F.one_hot(indices, self.codebook_size).float()
            avg_probs = encodings.mean(dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, indices.view(inputs.shape[0], inputs.shape[1]), vq_loss, perplexity


class LatentActionPrior(nn.Module):
    """Predicts latent action code indices from current context only."""

    def __init__(self, context_dim: int, num_latent_tokens: int, codebook_size: int, hidden_dim: int):
        super().__init__()
        self.num_latent_tokens = num_latent_tokens
        self.codebook_size = codebook_size
        self.context_norm = nn.LayerNorm(context_dim)
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_latent_tokens * codebook_size),
        )

    def forward(self, context_tokens: torch.Tensor) -> torch.Tensor:
        if context_tokens.ndim != 3:
            raise ValueError(f"Expected context_tokens [B, T, D], got {tuple(context_tokens.shape)}")
        summary = self.context_norm(context_tokens).mean(dim=1)
        logits = self.net(summary)
        return logits.view(context_tokens.shape[0], self.num_latent_tokens, self.codebook_size)


class LatentActionHead(nn.Module):
    """Training-time posterior plus deployable prior for LARA latent action tokens."""

    def __init__(
        self,
        context_dim: int,
        action_dim: int,
        action_horizon: int,
        num_latent_tokens: int = 4,
        codebook_size: int = 128,
        hidden_dim: int = 1024,
        commitment_weight: float = 0.25,
        vq_loss_weight: float = 1.0,
        prior_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.vq_loss_weight = vq_loss_weight
        self.prior_loss_weight = prior_loss_weight
        self.posterior = PosteriorLatentActionEncoder(
            context_dim=context_dim,
            action_dim=action_dim,
            action_horizon=action_horizon,
            num_latent_tokens=num_latent_tokens,
            hidden_dim=hidden_dim,
        )
        self.codebook = VectorQuantizer(
            codebook_size=codebook_size,
            code_dim=context_dim,
            commitment_weight=commitment_weight,
        )
        self.prior = LatentActionPrior(
            context_dim=context_dim,
            num_latent_tokens=num_latent_tokens,
            codebook_size=codebook_size,
            hidden_dim=hidden_dim,
        )

    def forward(self, context_tokens: torch.Tensor, future_actions: torch.Tensor) -> LatentActionOutput:
        posterior_tokens = self.posterior(context_tokens, future_actions)
        quantized_tokens, code_indices, vq_loss, perplexity = self.codebook(posterior_tokens)
        prior_logits = self.prior(context_tokens)
        prior_loss = F.cross_entropy(
            prior_logits.reshape(-1, prior_logits.shape[-1]),
            code_indices.reshape(-1).detach(),
        )
        loss = self.vq_loss_weight * vq_loss + self.prior_loss_weight * prior_loss
        return LatentActionOutput(
            tokens=quantized_tokens,
            loss=loss,
            vq_loss=vq_loss.detach(),
            prior_loss=prior_loss.detach(),
            perplexity=perplexity.detach(),
        )

    @torch.no_grad()
    def predict(self, context_tokens: torch.Tensor) -> torch.Tensor:
        prior_logits = self.prior(context_tokens)
        code_indices = prior_logits.argmax(dim=-1)
        return self.codebook.codebook(code_indices).to(dtype=context_tokens.dtype)
