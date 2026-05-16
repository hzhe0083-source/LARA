from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LatentActionOutput:
    tokens: torch.Tensor
    loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    vq_loss: torch.Tensor
    prior_loss: torch.Tensor
    code_usage_loss: torch.Tensor
    perplexity: torch.Tensor
    reconstructed_actions: torch.Tensor


class LatentActionTransitionHead(nn.Module):
    """Predicts chunk-boundary control state from context and latent action tokens."""

    def __init__(
        self,
        context_dim: int,
        state_dim: int,
        hidden_dim: int,
        num_boundaries: int = 2,
    ):
        super().__init__()
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if num_boundaries <= 0:
            raise ValueError("num_boundaries must be positive")
        self.state_dim = state_dim
        self.num_boundaries = num_boundaries
        self.context_norm = nn.LayerNorm(context_dim)
        self.latent_norm = nn.LayerNorm(context_dim)
        self.net = nn.Sequential(
            nn.Linear(context_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_boundaries * state_dim),
        )

    def forward(self, context_tokens: torch.Tensor, latent_action_tokens: torch.Tensor) -> torch.Tensor:
        if context_tokens.ndim != 3:
            raise ValueError(f"Expected context_tokens [B, T, D], got {tuple(context_tokens.shape)}")
        if latent_action_tokens.ndim != 3:
            raise ValueError(f"Expected latent_action_tokens [B, L, D], got {tuple(latent_action_tokens.shape)}")
        if latent_action_tokens.shape[0] != context_tokens.shape[0]:
            raise ValueError(
                "latent_action_tokens batch size must match context_tokens: "
                f"{latent_action_tokens.shape[0]} != {context_tokens.shape[0]}"
            )
        if latent_action_tokens.shape[-1] != context_tokens.shape[-1]:
            raise ValueError(
                "latent_action_tokens hidden size must match context_tokens: "
                f"{latent_action_tokens.shape[-1]} != {context_tokens.shape[-1]}"
            )

        context_summary = self.context_norm(context_tokens).mean(dim=1)
        latent_summary = self.latent_norm(latent_action_tokens).mean(dim=1)
        output = self.net(torch.cat([context_summary, latent_summary], dim=-1))
        return output.view(context_tokens.shape[0], self.num_boundaries, self.state_dim)


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

    def forward(
        self,
        context_tokens: torch.Tensor,
        future_actions: torch.Tensor,
        future_action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
        if future_action_mask is not None:
            if future_action_mask.ndim != 2 or future_action_mask.shape != future_actions.shape[:2]:
                raise ValueError(
                    f"Expected future_action_mask {tuple(future_actions.shape[:2])}, "
                    f"got {tuple(future_action_mask.shape)}"
                )
            future_action_mask = future_action_mask.to(device=future_actions.device, dtype=future_actions.dtype)
            future_actions = future_actions * future_action_mask[:, :, None]

        batch_size = context_tokens.shape[0]
        context_summary = self.context_norm(context_tokens).mean(dim=1)
        context_features = self.context_proj(context_summary).unsqueeze(1).expand(-1, self.action_horizon, -1)
        action_features = self.action_proj(future_actions.to(dtype=context_tokens.dtype))
        action_features = action_features + self.action_pos[:, : self.action_horizon, :].to(
            device=action_features.device,
            dtype=action_features.dtype,
        )
        fused = self.encoder(torch.cat([context_features, action_features], dim=-1))
        if future_action_mask is None:
            pooled = fused.mean(dim=1)
        else:
            mask = future_action_mask.to(device=fused.device, dtype=fused.dtype)
            pooled = (fused * mask[:, :, None]).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        latents = self.to_latents(pooled)
        return latents.view(batch_size, self.num_latent_tokens, context_tokens.shape[-1])


class VectorQuantizer(nn.Module):
    """Straight-through VQ codebook for latent action tokens."""

    def __init__(
        self,
        codebook_size: int,
        code_dim: int,
        commitment_weight: float = 0.25,
        usage_temperature: float = 1.0,
    ):
        super().__init__()
        if usage_temperature <= 0:
            raise ValueError("usage_temperature must be positive")
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.commitment_weight = commitment_weight
        self.usage_temperature = usage_temperature
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
        soft_assignments = torch.softmax(-distances / self.usage_temperature, dim=-1)
        avg_soft_probs = soft_assignments.mean(dim=0).clamp_min(1e-8)
        uniform_probs = torch.full_like(avg_soft_probs, 1.0 / self.codebook_size)
        code_usage_loss = F.kl_div(torch.log(avg_soft_probs), uniform_probs, reduction="sum")
        quantized = inputs + (quantized - inputs).detach()

        with torch.no_grad():
            encodings = F.one_hot(indices, self.codebook_size).float()
            avg_probs = encodings.mean(dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, indices.view(inputs.shape[0], inputs.shape[1]), vq_loss, code_usage_loss, perplexity


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


class LatentActionDecoder(nn.Module):
    """Reconstructs the demonstrated executable action chunk from latent action tokens."""

    def __init__(
        self,
        context_dim: int,
        action_dim: int,
        action_horizon: int,
        hidden_dim: int,
    ):
        super().__init__()
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.context_norm = nn.LayerNorm(context_dim)
        self.latent_norm = nn.LayerNorm(context_dim)
        self.step_pos = nn.Parameter(torch.zeros(1, action_horizon, context_dim))
        self.net = nn.Sequential(
            nn.LayerNorm(context_dim * 3),
            nn.Linear(context_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )
        nn.init.normal_(self.step_pos, mean=0.0, std=0.02)

    def forward(self, context_tokens: torch.Tensor, latent_action_tokens: torch.Tensor) -> torch.Tensor:
        if context_tokens.ndim != 3:
            raise ValueError(f"Expected context_tokens [B, T, D], got {tuple(context_tokens.shape)}")
        if latent_action_tokens.ndim != 3:
            raise ValueError(f"Expected latent_action_tokens [B, L, D], got {tuple(latent_action_tokens.shape)}")
        if latent_action_tokens.shape[0] != context_tokens.shape[0]:
            raise ValueError(
                "latent_action_tokens batch size must match context_tokens: "
                f"{latent_action_tokens.shape[0]} != {context_tokens.shape[0]}"
            )
        if latent_action_tokens.shape[-1] != context_tokens.shape[-1]:
            raise ValueError(
                "latent_action_tokens hidden size must match context_tokens: "
                f"{latent_action_tokens.shape[-1]} != {context_tokens.shape[-1]}"
            )

        context_summary = self.context_norm(context_tokens).mean(dim=1)
        latent_summary = self.latent_norm(latent_action_tokens).mean(dim=1)
        context_features = context_summary[:, None, :].expand(-1, self.action_horizon, -1)
        latent_features = latent_summary[:, None, :].expand(-1, self.action_horizon, -1)
        step_features = self.step_pos.to(device=context_tokens.device, dtype=context_tokens.dtype).expand(
            context_tokens.shape[0],
            -1,
            -1,
        )
        return self.net(torch.cat([context_features, latent_features, step_features], dim=-1))


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
        code_usage_loss_weight: float = 0.0,
        code_usage_temperature: float = 1.0,
        reconstruction_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.vq_loss_weight = vq_loss_weight
        self.prior_loss_weight = prior_loss_weight
        self.code_usage_loss_weight = code_usage_loss_weight
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
            usage_temperature=code_usage_temperature,
        )
        self.prior = LatentActionPrior(
            context_dim=context_dim,
            num_latent_tokens=num_latent_tokens,
            codebook_size=codebook_size,
            hidden_dim=hidden_dim,
        )
        self.decoder = LatentActionDecoder(
            context_dim=context_dim,
            action_dim=action_dim,
            action_horizon=action_horizon,
            hidden_dim=hidden_dim,
        )

    def forward(
        self,
        context_tokens: torch.Tensor,
        future_actions: torch.Tensor,
        future_action_mask: torch.Tensor | None = None,
    ) -> LatentActionOutput:
        posterior_tokens = self.posterior(context_tokens, future_actions, future_action_mask=future_action_mask)
        quantized_tokens, code_indices, vq_loss, code_usage_loss, perplexity = self.codebook(posterior_tokens)
        prior_logits = self.prior(context_tokens)
        prior_loss = F.cross_entropy(
            prior_logits.reshape(-1, prior_logits.shape[-1]),
            code_indices.reshape(-1).detach(),
        )
        reconstructed_actions = self.decoder(context_tokens, quantized_tokens)
        reconstruction_loss = self._reconstruction_loss(
            reconstructed_actions,
            future_actions,
            future_action_mask=future_action_mask,
        )
        loss = (
            self.reconstruction_loss_weight * reconstruction_loss
            + self.vq_loss_weight * vq_loss
            + self.prior_loss_weight * prior_loss
            + self.code_usage_loss_weight * code_usage_loss
        )
        return LatentActionOutput(
            tokens=quantized_tokens,
            loss=loss,
            reconstruction_loss=reconstruction_loss.detach(),
            vq_loss=vq_loss.detach(),
            prior_loss=prior_loss.detach(),
            code_usage_loss=code_usage_loss.detach(),
            perplexity=perplexity.detach(),
            reconstructed_actions=reconstructed_actions.detach(),
        )

    @torch.no_grad()
    def predict(self, context_tokens: torch.Tensor) -> torch.Tensor:
        prior_logits = self.prior(context_tokens)
        code_indices = prior_logits.argmax(dim=-1)
        return self.codebook.codebook(code_indices).to(dtype=context_tokens.dtype)

    @staticmethod
    def _reconstruction_loss(
        predicted_actions: torch.Tensor,
        target_actions: torch.Tensor,
        future_action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if predicted_actions.ndim != 3:
            raise ValueError(f"Expected predicted_actions [B, H, A], got {tuple(predicted_actions.shape)}")
        if target_actions.ndim != 3:
            raise ValueError(f"Expected target_actions [B, H, A], got {tuple(target_actions.shape)}")
        if predicted_actions.shape != target_actions.shape:
            raise ValueError(
                "predicted_actions and target_actions must share shape: "
                f"{tuple(predicted_actions.shape)} != {tuple(target_actions.shape)}"
            )

        per_step_loss = (predicted_actions.float() - target_actions.float()).pow(2).mean(dim=-1)
        if future_action_mask is None:
            return per_step_loss.mean()
        if future_action_mask.ndim == 3 and future_action_mask.shape[-1] == 1:
            future_action_mask = future_action_mask.squeeze(-1)
        if future_action_mask.shape != per_step_loss.shape:
            raise ValueError(
                f"Expected future_action_mask shape {tuple(per_step_loss.shape)}, "
                f"got {tuple(future_action_mask.shape)}"
            )
        mask = future_action_mask.to(device=per_step_loss.device, dtype=per_step_loss.dtype)
        return (per_step_loss * mask).sum() / mask.sum().clamp_min(1.0)
