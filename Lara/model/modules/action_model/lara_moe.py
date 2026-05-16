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
    diversity_loss: torch.Tensor
    entropy_loss: torch.Tensor
    pool_coverage_loss: torch.Tensor
    route_loss_weighted: torch.Tensor
    pool_loss_weighted: torch.Tensor
    pool_coverage_loss_weighted: torch.Tensor
    utility_loss_weighted: torch.Tensor
    utility_rank_loss_weighted: torch.Tensor
    utility_head_loss_weighted: torch.Tensor
    balance_loss_weighted: torch.Tensor
    stickiness_loss_weighted: torch.Tensor
    diversity_loss_weighted: torch.Tensor
    entropy_loss_weighted: torch.Tensor
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
    pool_teacher_mass: torch.Tensor
    active_teacher_mass: torch.Tensor
    pool_teacher_top1_match: torch.Tensor
    active_teacher_top1_match: torch.Tensor
    pool_critical_miss_rate: torch.Tensor


@dataclass
class ResidentPoolOutput:
    logits: torch.Tensor
    probs: torch.Tensor
    mask: torch.Tensor


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


def forced_router_probs_from_scores(
    forced_router_probs: torch.Tensor,
    pool_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if forced_router_probs.ndim != 2:
        raise ValueError(f"forced_router_probs must have shape [B, M], got {tuple(forced_router_probs.shape)}")
    if not torch.isfinite(forced_router_probs).all() or torch.any(forced_router_probs < 0):
        raise ValueError("forced_router_probs must contain finite non-negative values")
    active_mask = forced_router_probs > 0
    if not torch.all(active_mask.any(dim=-1)):
        raise ValueError("forced_router_probs must select at least one expert per sample")
    if pool_mask is not None:
        pool_mask = _validate_expert_mask(pool_mask, forced_router_probs, "pool_mask")
        if torch.any(active_mask & ~pool_mask):
            raise ValueError("forced_router_probs selects experts outside pool_mask")
        forced_router_probs = forced_router_probs.masked_fill(~pool_mask, 0.0)
        active_mask = forced_router_probs > 0
    router_probs = forced_router_probs / forced_router_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return router_probs, active_mask


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
    uniform_floor: float = 0.0,
    top_r: Optional[int] = None,
) -> torch.Tensor:
    if expert_losses.ndim != 2:
        raise ValueError(f"Expected expert_losses [B, M], got {tuple(expert_losses.shape)}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if uniform_floor < 0 or uniform_floor >= 1:
        raise ValueError("uniform_floor must be in [0, 1)")
    if top_r is not None and top_r <= 0:
        raise ValueError("top_r must be positive")
    logits = -expert_losses / temperature
    if top_r is not None:
        top_r_mask = topk_mask(logits, top_k=top_r, allowed_mask=mask)
        probs = masked_softmax(logits, top_r_mask)
        support_mask = top_r_mask
    else:
        probs = masked_softmax(logits, mask)
        support_mask = torch.ones_like(probs, dtype=torch.bool) if mask is None else _validate_expert_mask(mask, probs, "mask")
    if uniform_floor == 0:
        return probs

    uniform = support_mask.to(dtype=probs.dtype)
    uniform = uniform / uniform.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return (1.0 - uniform_floor) * probs + uniform_floor * uniform


def entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1).mean()


def normalize_probability_targets(
    probs: torch.Tensor,
    reference: torch.Tensor,
    target_name: str,
) -> torch.Tensor:
    if probs.shape != reference.shape:
        raise ValueError(f"{target_name} must have shape {tuple(reference.shape)}, got {tuple(probs.shape)}")
    probs = probs.to(device=reference.device, dtype=reference.dtype)
    if not torch.isfinite(probs).all() or torch.any(probs < 0):
        raise ValueError(f"{target_name} must contain finite non-negative values")
    denom = probs.sum(dim=-1, keepdim=True)
    if torch.any(denom <= 0):
        raise ValueError(f"{target_name} must have positive mass for every sample")
    return probs / denom.clamp_min(1e-8)


def pool_coverage_objective(pool_logits: torch.Tensor, teacher_probs: torch.Tensor) -> torch.Tensor:
    teacher_probs = normalize_probability_targets(teacher_probs, pool_logits, "teacher_probs")
    soft_pool_probs = torch.softmax(pool_logits, dim=-1)
    coverage = (teacher_probs.detach() * soft_pool_probs).sum(dim=-1).clamp_min(1e-8)
    return -torch.log(coverage).mean()


def pool_coverage_diagnostics(
    pool_mask: torch.Tensor,
    active_mask: torch.Tensor,
    teacher_probs: torch.Tensor,
    critical_threshold: float = 0.0,
) -> dict[str, torch.Tensor]:
    teacher_probs = normalize_probability_targets(teacher_probs, teacher_probs, "teacher_probs")
    pool_mask = _validate_expert_mask(pool_mask, teacher_probs, "pool_mask")
    active_mask = _validate_expert_mask(active_mask, teacher_probs, "active_mask")
    if torch.any(active_mask & ~pool_mask):
        raise ValueError("active_mask cannot select experts outside pool_mask")
    if critical_threshold < 0 or critical_threshold > 1:
        raise ValueError("critical_threshold must be in [0, 1]")

    pool_teacher_mass = (teacher_probs * pool_mask.to(dtype=teacher_probs.dtype)).sum(dim=-1).mean()
    active_teacher_mass = (teacher_probs * active_mask.to(dtype=teacher_probs.dtype)).sum(dim=-1).mean()
    teacher_top1 = teacher_probs.argmax(dim=-1, keepdim=True)
    pool_teacher_top1_match = pool_mask.gather(dim=-1, index=teacher_top1).to(dtype=teacher_probs.dtype).mean()
    active_teacher_top1_match = active_mask.gather(dim=-1, index=teacher_top1).to(dtype=teacher_probs.dtype).mean()

    if critical_threshold > 0:
        critical_mask = teacher_probs >= critical_threshold
        no_critical = ~critical_mask.any(dim=-1, keepdim=True)
        critical_mask = torch.where(
            no_critical,
            torch.zeros_like(critical_mask).scatter(dim=-1, index=teacher_top1, value=True),
            critical_mask,
        )
    else:
        critical_mask = torch.zeros_like(pool_mask, dtype=torch.bool).scatter(
            dim=-1,
            index=teacher_top1,
            value=True,
        )
    missed_critical = critical_mask & ~pool_mask
    pool_critical_miss_rate = (
        missed_critical.to(dtype=teacher_probs.dtype).sum()
        / critical_mask.to(dtype=teacher_probs.dtype).sum().clamp_min(1.0)
    )
    return {
        "pool_teacher_mass": pool_teacher_mass,
        "active_teacher_mass": active_teacher_mass,
        "pool_teacher_top1_match": pool_teacher_top1_match,
        "active_teacher_top1_match": active_teacher_top1_match,
        "pool_critical_miss_rate": pool_critical_miss_rate,
    }


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


def expert_diversity_loss(expert_outputs: torch.Tensor) -> torch.Tensor:
    if expert_outputs.ndim != 4:
        raise ValueError(f"Expected expert_outputs [B, M, T, D], got {tuple(expert_outputs.shape)}")
    num_experts = expert_outputs.shape[1]
    if num_experts < 2:
        return expert_outputs.new_zeros(())

    vectors = expert_outputs.permute(1, 0, 2, 3).reshape(num_experts, -1).float()
    vectors = F.normalize(vectors, dim=-1)
    similarity = vectors @ vectors.t()
    off_diagonal = ~torch.eye(num_experts, dtype=torch.bool, device=expert_outputs.device)
    return similarity[off_diagonal].pow(2).mean().to(dtype=expert_outputs.dtype)


def route_entropy_regularization_loss(router_probs: torch.Tensor, pool_probs: torch.Tensor) -> torch.Tensor:
    return -(entropy(router_probs) + entropy(pool_probs))


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


def _validate_score_pair(
    predicted_scores: torch.Tensor,
    target_scores: torch.Tensor,
    candidate_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if predicted_scores.ndim != 2:
        raise ValueError(f"Expected predicted_scores [B, M], got {tuple(predicted_scores.shape)}")
    if target_scores.shape != predicted_scores.shape:
        raise ValueError(
            f"target_scores must have shape {tuple(predicted_scores.shape)}, got {tuple(target_scores.shape)}"
        )
    if candidate_mask is None:
        return torch.ones_like(predicted_scores, dtype=torch.bool)
    return _validate_expert_mask(candidate_mask, predicted_scores, "candidate_mask")


def _rank_vector(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), device=values.device, dtype=torch.float32)
    return ranks


def _pearson_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.float() - x.float().mean()
    y = y.float() - y.float().mean()
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denom.detach()) <= 1e-8:
        return x.new_zeros(())
    return (x * y).sum() / denom


def spearman_rank_correlation(
    predicted_scores: torch.Tensor,
    target_scores: torch.Tensor,
    candidate_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    candidate_mask = _validate_score_pair(predicted_scores, target_scores, candidate_mask)
    correlations = []
    for pred_row, target_row, mask_row in zip(predicted_scores, target_scores, candidate_mask):
        if int(mask_row.sum().item()) < 2:
            continue
        correlations.append(_pearson_correlation(_rank_vector(pred_row[mask_row]), _rank_vector(target_row[mask_row])))
    if not correlations:
        return predicted_scores.new_zeros(())
    return torch.stack(correlations).mean().to(dtype=predicted_scores.dtype)


def kendall_rank_correlation(
    predicted_scores: torch.Tensor,
    target_scores: torch.Tensor,
    candidate_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    candidate_mask = _validate_score_pair(predicted_scores, target_scores, candidate_mask)
    correlations = []
    for pred_row, target_row, mask_row in zip(predicted_scores, target_scores, candidate_mask):
        if int(mask_row.sum().item()) < 2:
            continue
        pred_values = pred_row[mask_row].float()
        target_values = target_row[mask_row].float()
        pred_diff = pred_values[:, None] - pred_values[None, :]
        target_diff = target_values[:, None] - target_values[None, :]
        pair_mask = torch.triu(
            torch.ones_like(pred_diff, dtype=torch.bool),
            diagonal=1,
        ) & (pred_diff != 0) & (target_diff != 0)
        if not torch.any(pair_mask):
            continue
        correlations.append(torch.sign(pred_diff[pair_mask] * target_diff[pair_mask]).mean())
    if not correlations:
        return predicted_scores.new_zeros(())
    return torch.stack(correlations).mean().to(dtype=predicted_scores.dtype)


def topk_route_consistency(
    router_scores: torch.Tensor,
    target_scores: torch.Tensor,
    top_k: int,
    candidate_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    candidate_mask = _validate_score_pair(router_scores, target_scores, candidate_mask)
    overlaps = []
    for router_row, target_row, mask_row in zip(router_scores, target_scores, candidate_mask):
        candidate_indices = torch.where(mask_row)[0]
        if candidate_indices.numel() == 0:
            continue
        k = min(top_k, int(candidate_indices.numel()))
        router_top = candidate_indices[torch.topk(router_row[candidate_indices], k=k).indices]
        target_top = candidate_indices[torch.topk(target_row[candidate_indices], k=k).indices]
        overlaps.append(torch.isin(router_top, target_top).to(dtype=router_scores.dtype).mean())
    if not overlaps:
        return router_scores.new_zeros(())
    return torch.stack(overlaps).mean()


def route_regret_from_scores(
    target_scores: torch.Tensor,
    active_mask: torch.Tensor,
    candidate_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if target_scores.ndim != 2:
        raise ValueError(f"Expected target_scores [B, M], got {tuple(target_scores.shape)}")
    active_mask = _validate_expert_mask(active_mask, target_scores, "active_mask")
    if candidate_mask is None:
        candidate_mask = torch.ones_like(active_mask, dtype=torch.bool)
    else:
        candidate_mask = _validate_expert_mask(candidate_mask, target_scores, "candidate_mask")
    if torch.any(active_mask & ~candidate_mask):
        raise ValueError("active_mask cannot select experts outside candidate_mask")

    min_value = torch.finfo(target_scores.dtype).min
    best_score = target_scores.masked_fill(~candidate_mask, min_value).max(dim=-1).values
    selected_score = target_scores.masked_fill(~active_mask, min_value).max(dim=-1).values
    return (best_score - selected_score).clamp_min(0.0).mean()


def posterior_router_kl(
    router_probs: torch.Tensor,
    posterior_probs: torch.Tensor,
    candidate_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    candidate_mask = _validate_score_pair(router_probs, posterior_probs, candidate_mask)
    posterior_target = renormalize_probs(posterior_probs, candidate_mask)
    router_candidate = renormalize_probs(router_probs.clamp_min(1e-8), candidate_mask)
    return F.kl_div(torch.log(router_candidate.clamp_min(1e-8)), posterior_target.detach(), reduction="batchmean")


def route_quality_metrics(
    router_probs: torch.Tensor,
    posterior_probs: Optional[torch.Tensor] = None,
    utility_scores: Optional[torch.Tensor] = None,
    pool_mask: Optional[torch.Tensor] = None,
    active_mask: Optional[torch.Tensor] = None,
    route_ids: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    retention_fractions: Optional[list[float]] = None,
    top_k: Optional[int] = None,
    critical_threshold: float = 0.0,
) -> dict[str, torch.Tensor]:
    if router_probs.ndim != 2:
        raise ValueError(f"Expected router_probs [B, M], got {tuple(router_probs.shape)}")
    candidate_mask = (
        torch.ones_like(router_probs, dtype=torch.bool)
        if pool_mask is None
        else _validate_expert_mask(pool_mask, router_probs, "pool_mask")
    )
    metrics = {"router_entropy": entropy(router_probs)}
    if active_mask is not None:
        active_mask = _validate_expert_mask(active_mask, router_probs, "active_mask")
        if torch.any(active_mask & ~candidate_mask):
            raise ValueError("active_mask cannot select experts outside pool_mask")
        inferred_top_k = int(active_mask.sum(dim=-1).max().item())
        top_k = inferred_top_k if top_k is None else top_k

    if posterior_probs is not None:
        _validate_score_pair(router_probs, posterior_probs, candidate_mask)
        metrics["posterior_spearman"] = spearman_rank_correlation(router_probs, posterior_probs, candidate_mask)
        metrics["posterior_kendall"] = kendall_rank_correlation(router_probs, posterior_probs, candidate_mask)
        metrics["posterior_router_kl"] = posterior_router_kl(router_probs, posterior_probs, candidate_mask)
        if top_k is not None:
            metrics["posterior_topk_consistency"] = topk_route_consistency(
                router_probs,
                posterior_probs,
                top_k=top_k,
                candidate_mask=candidate_mask,
            )
        if active_mask is not None:
            metrics["posterior_route_regret"] = route_regret_from_scores(posterior_probs, active_mask, candidate_mask)
            coverage = pool_coverage_diagnostics(
                pool_mask=candidate_mask,
                active_mask=active_mask,
                teacher_probs=posterior_probs,
                critical_threshold=critical_threshold,
            )
            metrics["posterior_pool_mass"] = coverage["pool_teacher_mass"]
            metrics["posterior_active_mass"] = coverage["active_teacher_mass"]
            metrics["posterior_pool_top1_match"] = coverage["pool_teacher_top1_match"]
            metrics["posterior_active_top1_match"] = coverage["active_teacher_top1_match"]
            metrics["posterior_pool_critical_miss_rate"] = coverage["pool_critical_miss_rate"]

    if utility_scores is not None:
        _validate_score_pair(router_probs, utility_scores, candidate_mask)
        metrics["utility_spearman"] = spearman_rank_correlation(router_probs, utility_scores, candidate_mask)
        metrics["utility_kendall"] = kendall_rank_correlation(router_probs, utility_scores, candidate_mask)
        if top_k is not None:
            metrics["utility_topk_consistency"] = topk_route_consistency(
                router_probs,
                utility_scores,
                top_k=top_k,
                candidate_mask=candidate_mask,
            )
        if active_mask is not None:
            metrics["utility_route_regret"] = route_regret_from_scores(utility_scores, active_mask, candidate_mask)

    if route_ids is not None:
        metrics["route_switch_rate"] = route_switch_rate(route_ids, valid_mask)
    if retention_fractions is not None:
        retention_source = posterior_probs if posterior_probs is not None else router_probs
        for fraction, retained_mass in retained_probability_mass(retention_source, retention_fractions).items():
            metrics[f"retained_probability_mass/{fraction:g}"] = retained_mass
    return metrics


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


def utility_from_expert_losses(
    expert_losses: torch.Tensor,
    temperature: float = 1.0,
    normalize: bool = True,
) -> torch.Tensor:
    if expert_losses.ndim != 2:
        raise ValueError(f"Expected expert_losses [B, M], got {tuple(expert_losses.shape)}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    utilities = -expert_losses.detach() / temperature
    if not normalize:
        return utilities

    mean = utilities.mean(dim=-1, keepdim=True)
    std = utilities.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (utilities - mean) / std


def uncertainty_from_expert_losses(
    expert_losses: torch.Tensor,
    temperature: float = 1.0,
    normalize: bool = True,
) -> torch.Tensor:
    if expert_losses.ndim != 2:
        raise ValueError(f"Expected expert_losses [B, M], got {tuple(expert_losses.shape)}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    uncertainty = expert_losses.detach().clamp_min(0.0) / temperature
    if not normalize:
        return uncertainty

    mean = uncertainty.mean(dim=-1, keepdim=True).clamp_min(1e-6)
    return uncertainty / mean


def utility_component_targets_from_expert_losses(
    value_losses: torch.Tensor,
    progress_losses: Optional[torch.Tensor] = None,
    uncertainty_losses: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    normalize: bool = True,
) -> dict[str, torch.Tensor]:
    """Build utility-head component labels from counterfactual expert losses.

    Higher value/progress targets mean the expert better reconstructs the
    demonstrated action chunk. Higher uncertainty targets mean the expert has
    larger long-horizon reconstruction error and should be penalized by the
    route utility composition.
    """

    progress_losses = value_losses if progress_losses is None else progress_losses
    uncertainty_losses = value_losses if uncertainty_losses is None else uncertainty_losses
    if progress_losses.shape != value_losses.shape or uncertainty_losses.shape != value_losses.shape:
        raise ValueError(
            "utility component losses must share shape "
            f"{tuple(value_losses.shape)}; got progress={tuple(progress_losses.shape)}, "
            f"uncertainty={tuple(uncertainty_losses.shape)}"
        )
    return {
        "value": utility_from_expert_losses(value_losses, temperature=temperature, normalize=normalize),
        "progress": utility_from_expert_losses(progress_losses, temperature=temperature, normalize=normalize),
        "uncertainty": uncertainty_from_expert_losses(
            uncertainty_losses,
            temperature=temperature,
            normalize=normalize,
        ),
    }


def aggregate_episode_responsibilities(
    posterior_probs: torch.Tensor,
    episode_ids: torch.Tensor,
    avg_weight: float = 1.0,
    max_weight: float = 0.0,
    utility_scores: Optional[torch.Tensor] = None,
    utility_weight: float = 0.0,
    utility_candidate_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if posterior_probs.ndim != 2:
        raise ValueError(f"Expected posterior_probs [B, M], got {tuple(posterior_probs.shape)}")
    if episode_ids.ndim != 1 or episode_ids.shape[0] != posterior_probs.shape[0]:
        raise ValueError(
            f"Expected episode_ids [B] with B={posterior_probs.shape[0]}, got {tuple(episode_ids.shape)}"
        )
    if avg_weight < 0 or max_weight < 0 or utility_weight < 0:
        raise ValueError("pool target weights must be non-negative")
    if avg_weight == 0 and max_weight == 0 and utility_weight == 0:
        raise ValueError("at least one pool target weight must be positive")
    posterior_probs = normalize_probability_targets(posterior_probs, posterior_probs, "posterior_probs")
    if utility_scores is not None:
        if utility_scores.shape != posterior_probs.shape:
            raise ValueError(
                f"utility_scores must have shape {tuple(posterior_probs.shape)}, got {tuple(utility_scores.shape)}"
            )
        utility_scores = utility_scores.to(device=posterior_probs.device, dtype=posterior_probs.dtype)
        if not torch.isfinite(utility_scores).all():
            raise ValueError("utility_scores must contain finite values")
        if utility_candidate_mask is None:
            utility_candidate_mask = torch.ones_like(posterior_probs, dtype=torch.bool)
        else:
            utility_candidate_mask = _validate_expert_mask(
                utility_candidate_mask.to(device=posterior_probs.device),
                posterior_probs,
                "utility_candidate_mask",
            )
    elif utility_weight > 0:
        raise ValueError("utility_scores are required when utility_weight is positive")

    episode_ids = episode_ids.to(device=posterior_probs.device)
    targets = torch.zeros_like(posterior_probs)
    for episode_id in torch.unique(episode_ids):
        mask = episode_ids == episode_id
        episode_score = posterior_probs.new_zeros((1, posterior_probs.shape[-1]))
        if avg_weight > 0:
            episode_score = episode_score + avg_weight * posterior_probs[mask].mean(dim=0, keepdim=True)
        if max_weight > 0:
            episode_score = episode_score + max_weight * posterior_probs[mask].max(dim=0, keepdim=True).values
        if utility_scores is not None and utility_weight > 0:
            utility_rows = utility_scores[mask]
            utility_mask_rows = utility_candidate_mask[mask]
            utility_probs = masked_softmax(utility_rows, utility_mask_rows)
            episode_score = episode_score + utility_weight * utility_probs.mean(dim=0, keepdim=True)
        targets[mask] = episode_score
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

    def __init__(
        self,
        hidden_size: int,
        expert_hidden_size: int,
        action_horizon: int,
        action_dim: int,
        state_dim: Optional[int] = None,
    ):
        super().__init__()
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.context_norm = nn.LayerNorm(hidden_size)
        self.state_proj = nn.Linear(state_dim, hidden_size) if state_dim is not None else None
        self.net = nn.Sequential(
            nn.Linear(hidden_size, expert_hidden_size),
            nn.GELU(),
            nn.Linear(expert_hidden_size, action_horizon * action_dim),
        )

    def forward(self, tokens: torch.Tensor, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens [B, T, D], got {tuple(tokens.shape)}")
        context_summary = self.context_norm(tokens).mean(dim=1)
        if state is not None and self.state_proj is not None:
            if state.ndim == 2:
                state = state.unsqueeze(1)
            if state.ndim != 3 or state.shape[0] != tokens.shape[0]:
                raise ValueError(f"Expected state [B, D] or [B, T, D], got {tuple(state.shape)}")
            state_summary = state.to(device=tokens.device, dtype=tokens.dtype).mean(dim=1)
            context_summary = context_summary + self.state_proj(state_summary)
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
        state_dim: Optional[int] = None,
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
                    state_dim=state_dim,
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, tokens: torch.Tensor, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        return torch.stack([expert(tokens, state=state) for expert in self.experts], dim=1)

    @staticmethod
    def routed_actions(pred_actions: torch.Tensor, route_weights: torch.Tensor) -> torch.Tensor:
        if pred_actions.ndim != 4:
            raise ValueError(f"Expected pred_actions [B, M, H, D], got {tuple(pred_actions.shape)}")
        if route_weights.ndim != 2 or route_weights.shape != pred_actions.shape[:2]:
            raise ValueError(
                f"Expected route_weights [B, M] matching pred_actions, got {tuple(route_weights.shape)}"
            )
        return torch.sum(pred_actions * route_weights[:, :, None, None].to(pred_actions.dtype), dim=1)

    @staticmethod
    def action_chunk_loss(
        pred_actions: torch.Tensor,
        target_actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        execution_horizon: Optional[int] = None,
        execution_loss_weight: float = 1.0,
        prediction_loss_weight: float = 1.0,
    ) -> torch.Tensor:
        if pred_actions.ndim != 3:
            raise ValueError(f"Expected pred_actions [B, H, D], got {tuple(pred_actions.shape)}")
        if target_actions.ndim != 3:
            raise ValueError(f"Expected target_actions [B, H, D], got {tuple(target_actions.shape)}")
        if pred_actions.shape != target_actions.shape:
            raise ValueError(
                f"pred_actions and target_actions must share shape, got "
                f"{tuple(pred_actions.shape)} vs {tuple(target_actions.shape)}"
            )

        loss = (pred_actions - target_actions) ** 2
        if action_mask is not None:
            if action_mask.ndim == 3 and action_mask.shape[-1] == 1:
                action_mask = action_mask.squeeze(-1)
            if action_mask.shape != loss.shape[:2]:
                raise ValueError(f"Expected action_mask shape {tuple(loss.shape[:2])}, got {tuple(action_mask.shape)}")
            mask = action_mask.to(device=loss.device, dtype=loss.dtype).unsqueeze(-1)
        else:
            mask = torch.ones(loss.shape[:2], device=loss.device, dtype=loss.dtype).unsqueeze(-1)
        weights = None
        if execution_loss_weight != prediction_loss_weight:
            horizon = loss.shape[1]
            execution_horizon = horizon if execution_horizon is None else min(execution_horizon, horizon)
            weights = torch.full(
                (1, horizon, 1),
                prediction_loss_weight,
                device=loss.device,
                dtype=loss.dtype,
            )
            weights[:, :execution_horizon, :] = execution_loss_weight
        if weights is None:
            return (loss * mask).sum() / (mask.sum() * loss.shape[-1]).clamp_min(1.0)
        weighted_mask = mask * weights
        return (loss * weighted_mask).sum() / (weighted_mask.sum() * loss.shape[-1]).clamp_min(1.0)

    @staticmethod
    def reconstruction_losses(
        pred_actions: torch.Tensor,
        target_actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        execution_horizon: Optional[int] = None,
        execution_loss_weight: float = 1.0,
        prediction_loss_weight: float = 1.0,
    ) -> torch.Tensor:
        return ActionChunkExpertBank.reconstruction_loss_components(
            pred_actions,
            target_actions,
            action_mask=action_mask,
            execution_horizon=execution_horizon,
            execution_loss_weight=execution_loss_weight,
            prediction_loss_weight=prediction_loss_weight,
        )["weighted"]

    @staticmethod
    def reconstruction_loss_components(
        pred_actions: torch.Tensor,
        target_actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        execution_horizon: Optional[int] = None,
        execution_loss_weight: float = 1.0,
        prediction_loss_weight: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        if pred_actions.ndim != 4:
            raise ValueError(f"Expected pred_actions [B, M, H, D], got {tuple(pred_actions.shape)}")
        if target_actions.ndim != 3:
            raise ValueError(f"Expected target_actions [B, H, D], got {tuple(target_actions.shape)}")
        if pred_actions.shape[0] != target_actions.shape[0] or pred_actions.shape[2:] != target_actions.shape[1:]:
            raise ValueError(
                "pred_actions and target_actions must agree on batch, horizon, and action dim: "
                f"{tuple(pred_actions.shape)} vs {tuple(target_actions.shape)}"
            )

        per_step_loss = (pred_actions - target_actions[:, None, :, :]).pow(2).mean(dim=-1)
        horizon = per_step_loss.shape[-1]
        if action_mask is not None:
            if action_mask.ndim == 3 and action_mask.shape[-1] == 1:
                action_mask = action_mask.squeeze(-1)
            if action_mask.shape != target_actions.shape[:2]:
                raise ValueError(
                    f"Expected action_mask shape {tuple(target_actions.shape[:2])}, got {tuple(action_mask.shape)}"
                )
            mask = action_mask.to(device=per_step_loss.device, dtype=per_step_loss.dtype)[:, None, :]
        else:
            mask = torch.ones(
                per_step_loss.shape[0],
                1,
                horizon,
                device=per_step_loss.device,
                dtype=per_step_loss.dtype,
            )
        execution_horizon = horizon if execution_horizon is None else min(max(int(execution_horizon), 0), horizon)
        execution_mask = mask[:, :, :execution_horizon]
        prediction_mask = mask[:, :, execution_horizon:]
        execution_loss = (
            (per_step_loss[:, :, :execution_horizon] * execution_mask).sum(dim=-1)
            / execution_mask.sum(dim=-1).clamp_min(1.0)
            if execution_horizon > 0
            else per_step_loss.new_zeros(per_step_loss.shape[:2])
        )
        prediction_loss = (
            (per_step_loss[:, :, execution_horizon:] * prediction_mask).sum(dim=-1)
            / prediction_mask.sum(dim=-1).clamp_min(1.0)
            if execution_horizon < horizon
            else per_step_loss.new_zeros(per_step_loss.shape[:2])
        )
        full_loss = (per_step_loss * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
        weighted_step_loss = per_step_loss
        if execution_loss_weight != prediction_loss_weight:
            weights = torch.full(
                (1, 1, horizon),
                prediction_loss_weight,
                device=per_step_loss.device,
                dtype=per_step_loss.dtype,
            )
            weights[:, :, :execution_horizon] = execution_loss_weight
            weighted_step_loss = per_step_loss * weights
        return {
            "full": full_loss,
            "execution": execution_loss,
            "prediction": prediction_loss,
            "weighted": (weighted_step_loss * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0),
        }


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
        self.budget_proj = nn.Linear(2, hidden_size, bias=False)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, router_hidden_size),
            nn.GELU(),
            nn.Linear(router_hidden_size, num_experts),
        )
        nn.init.zeros_(self.budget_proj.weight)

    def forward(
        self,
        initial_context_tokens: torch.Tensor,
        budget_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if initial_context_tokens.ndim != 3:
            raise ValueError(f"Expected initial_context_tokens [B, T, D], got {tuple(initial_context_tokens.shape)}")
        context_summary = self.context_norm(initial_context_tokens).mean(dim=1)
        if budget_features is None:
            budget_features = context_summary.new_zeros(context_summary.shape[0], 2)
        if budget_features.ndim != 2 or budget_features.shape != (context_summary.shape[0], 2):
            raise ValueError(
                f"budget_features must have shape ({context_summary.shape[0]}, 2), "
                f"got {tuple(budget_features.shape)}"
            )
        budget_summary = self.budget_proj(budget_features.to(device=context_summary.device, dtype=context_summary.dtype))
        context_summary = context_summary + budget_summary
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
        episode_pool_size_min: Optional[int] = None,
        expert_hidden_size: int = 1024,
        router_hidden_size: int = 1024,
        router_loss_weight: float = 1.0,
        pool_loss_weight: float = 1.0,
        pool_coverage_loss_weight: float = 0.0,
        utility_loss_weight: float = 0.0,
        utility_rank_loss_weight: float = 0.0,
        utility_head_loss_weight: float = 0.0,
        balance_loss_weight: float = 0.0,
        stickiness_loss_weight: float = 0.0,
        diversity_loss_weight: float = 0.0,
        entropy_loss_weight: float = 0.0,
        use_utility_head: bool = False,
        utility_hidden_size: Optional[int] = None,
        utility_progress_weight: float = 1.0,
        utility_uncertainty_weight: float = 1.0,
        utility_cost_weight: float = 1.0,
        posterior_temperature: float = 1.0,
        posterior_uniform_floor: float = 0.0,
        posterior_top_r: Optional[int] = None,
        pool_critical_threshold: float = 0.0,
        inference_stickiness_weight: float = 0.0,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        self.num_experts = num_experts
        self.top_k = top_k
        self.episode_pool_size = episode_pool_size if episode_pool_size is not None else num_experts
        if self.episode_pool_size <= 0 or self.episode_pool_size > num_experts:
            raise ValueError("episode_pool_size must be in [1, num_experts]")
        self.episode_pool_size_min = (
            episode_pool_size_min if episode_pool_size_min is not None else self.episode_pool_size
        )
        if self.episode_pool_size_min <= 0 or self.episode_pool_size_min > self.episode_pool_size:
            raise ValueError("episode_pool_size_min must be in [1, episode_pool_size]")
        self.router_loss_weight = router_loss_weight
        self.pool_loss_weight = pool_loss_weight
        if pool_coverage_loss_weight < 0:
            raise ValueError("pool_coverage_loss_weight must be non-negative")
        self.pool_coverage_loss_weight = pool_coverage_loss_weight
        self.utility_loss_weight = utility_loss_weight
        self.utility_rank_loss_weight = utility_rank_loss_weight
        self.utility_head_loss_weight = utility_head_loss_weight
        self.balance_loss_weight = balance_loss_weight
        self.stickiness_loss_weight = stickiness_loss_weight
        self.diversity_loss_weight = diversity_loss_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.posterior_temperature = posterior_temperature
        self.posterior_uniform_floor = posterior_uniform_floor
        self.posterior_top_r = posterior_top_r
        if pool_critical_threshold < 0 or pool_critical_threshold > 1:
            raise ValueError("pool_critical_threshold must be in [0, 1]")
        self.pool_critical_threshold = pool_critical_threshold
        if inference_stickiness_weight < 0:
            raise ValueError("inference_stickiness_weight must be non-negative")
        self.inference_stickiness_weight = inference_stickiness_weight
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

    def _episode_pool_top_k(self, device: torch.device) -> int:
        if self.training and self.episode_pool_size_min < self.episode_pool_size:
            return int(
                torch.randint(
                    low=self.episode_pool_size_min,
                    high=self.episode_pool_size + 1,
                    size=(1,),
                    device=device,
                ).item()
            )
        return self.episode_pool_size

    def _pool_budget_features(
        self,
        pool_top_k: int | torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if isinstance(pool_top_k, torch.Tensor):
            pool_sizes = pool_top_k.to(device=device, dtype=dtype).view(-1)
            if pool_sizes.numel() == 1:
                pool_sizes = pool_sizes.expand(batch_size)
            if pool_sizes.shape[0] != batch_size:
                raise ValueError(f"pool_top_k must have {batch_size} values, got {pool_sizes.shape[0]}")
        else:
            pool_sizes = torch.full((batch_size,), float(pool_top_k), device=device, dtype=dtype)

        if torch.any(pool_sizes <= 0) or torch.any(pool_sizes > self.num_experts):
            raise ValueError(f"pool_top_k must be in [1, {self.num_experts}]")

        active_sizes = torch.minimum(
            torch.full_like(pool_sizes, float(self.top_k)),
            pool_sizes,
        )
        resident_fraction = pool_sizes / float(self.num_experts)
        active_within_resident = active_sizes / pool_sizes.clamp_min(1.0)
        return torch.stack([resident_fraction, active_within_resident], dim=-1)

    def _pool_size_from_mask(self, pool_mask: torch.Tensor, batch_size: int) -> torch.Tensor:
        if pool_mask.ndim != 2 or pool_mask.shape != (batch_size, self.num_experts):
            raise ValueError(f"pool_mask must have shape ({batch_size}, {self.num_experts}), got {tuple(pool_mask.shape)}")
        pool_mask = pool_mask.to(dtype=torch.bool)
        if not torch.all(pool_mask.any(dim=-1)):
            raise ValueError("pool_mask must select at least one expert for every sample")
        return pool_mask.sum(dim=-1)

    def _expert_outputs(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.stack([expert(tokens) for expert in self.experts], dim=1)

    def _expert_residual(self, expert_outputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.sum(expert_outputs * weights[:, :, None, None], dim=1)

    def expert_conditioning_tokens(self, conditioning_tokens: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(conditioning_tokens, latent_action_tokens=None)
        expert_outputs = self._expert_outputs(conditioning_tokens)
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
        if pool_mask is None:
            pool_top_k = self._episode_pool_top_k(pool_context.device)
        else:
            pool_top_k = self._pool_size_from_mask(pool_mask, pool_context.shape[0])
        budget_features = self._pool_budget_features(
            pool_top_k,
            batch_size=pool_context.shape[0],
            device=pool_context.device,
            dtype=pool_context.dtype,
        )
        pool_logits = self.pool_router(pool_context, budget_features=budget_features)
        if pool_mask is None:
            pool_mask = topk_mask(pool_logits, top_k=int(pool_top_k))
        else:
            pool_mask = _validate_expert_mask(pool_mask, pool_logits, "pool_mask")
        pool_probs = masked_softmax(pool_logits, pool_mask)
        return pool_logits, pool_probs, pool_mask

    @torch.no_grad()
    def select_resident_pool(
        self,
        conditioning_tokens: torch.Tensor,
        initial_context_tokens: Optional[torch.Tensor] = None,
        pool_mask: Optional[torch.Tensor] = None,
    ) -> ResidentPoolOutput:
        """Select the episode-level resident expert pool for reuse across chunks."""
        self._validate_inputs(conditioning_tokens, latent_action_tokens=None)
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
        return ResidentPoolOutput(logits=pool_logits, probs=pool_probs, mask=pool_mask)

    def forward(
        self,
        conditioning_tokens: torch.Tensor,
        latent_action_tokens: Optional[torch.Tensor] = None,
        initial_context_tokens: Optional[torch.Tensor] = None,
        pool_mask: Optional[torch.Tensor] = None,
        expert_action_losses: Optional[torch.Tensor] = None,
        pool_target_probs: Optional[torch.Tensor] = None,
        forced_router_probs: Optional[torch.Tensor] = None,
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

        if forced_router_probs is not None:
            if forced_router_probs.shape != (conditioning_tokens.shape[0], self.num_experts):
                raise ValueError(
                    "forced_router_probs must have shape "
                    f"({conditioning_tokens.shape[0]}, {self.num_experts}), got {tuple(forced_router_probs.shape)}"
                )
            forced_router_probs = forced_router_probs.to(
                device=conditioning_tokens.device,
                dtype=conditioning_tokens.dtype,
            )
            if pool_mask is None:
                pool_mask = forced_router_probs > 0

        pool_logits, pool_probs, pool_mask = self._resident_pool(
            conditioning_tokens=conditioning_tokens,
            initial_context_tokens=initial_context_tokens,
            pool_mask=pool_mask,
        )
        router_logits = self.chunk_router(conditioning_tokens)
        if (not self.training) and previous_router_probs is not None and self.inference_stickiness_weight > 0:
            if previous_router_probs.shape != router_logits.shape:
                raise ValueError(
                    f"previous_router_probs must have shape {tuple(router_logits.shape)}, "
                    f"got {tuple(previous_router_probs.shape)}"
                )
            previous_log_probs = torch.log(
                previous_router_probs.to(device=router_logits.device, dtype=router_logits.dtype).clamp_min(1e-8)
            )
            router_logits = router_logits + self.inference_stickiness_weight * previous_log_probs
        if forced_router_probs is None:
            active_mask = topk_mask(router_logits, top_k=self.top_k, allowed_mask=pool_mask)
            router_probs = masked_softmax(router_logits, active_mask)
        else:
            router_probs, active_mask = forced_router_probs_from_scores(forced_router_probs, pool_mask=pool_mask)

        pool_teacher = None
        if expert_action_losses is not None:
            if expert_action_losses.shape != router_logits.shape:
                raise ValueError(
                    f"expert_action_losses must have shape {tuple(router_logits.shape)}, "
                    f"got {tuple(expert_action_losses.shape)}"
                )
            posterior_probs = posterior_from_expert_losses(
                expert_action_losses,
                temperature=self.posterior_temperature,
                uniform_floor=self.posterior_uniform_floor,
                top_r=self.posterior_top_r,
            )
            route_loss = masked_kl_div(router_logits, posterior_probs, pool_mask)
            pool_teacher = pool_target_probs if pool_target_probs is not None else posterior_probs
            pool_teacher = normalize_probability_targets(pool_teacher, pool_logits, "pool_target_probs")
            pool_loss = F.kl_div(F.log_softmax(pool_logits, dim=-1), pool_teacher.detach(), reduction="batchmean")
            train_weights = posterior_probs
        elif latent_action_tokens is not None:
            posterior_logits = self.posterior(conditioning_tokens, latent_action_tokens)
            posterior_probs = torch.softmax(posterior_logits, dim=-1)
            route_loss = masked_kl_div(router_logits, posterior_probs, pool_mask)
            pool_teacher = pool_target_probs if pool_target_probs is not None else posterior_probs
            pool_teacher = normalize_probability_targets(pool_teacher, pool_logits, "pool_target_probs")
            pool_loss = F.kl_div(F.log_softmax(pool_logits, dim=-1), pool_teacher.detach(), reduction="batchmean")
            train_weights = posterior_probs
        else:
            posterior_probs = router_probs.detach()
            route_loss = router_logits.new_zeros(())
            pool_loss = router_logits.new_zeros(())
            train_weights = router_probs

        has_training_teacher = expert_action_losses is not None or latent_action_tokens is not None
        if pool_teacher is None:
            pool_teacher = posterior_probs.detach()
            pool_coverage_loss = router_logits.new_zeros(())
        else:
            pool_coverage_loss = pool_coverage_objective(pool_logits, pool_teacher)
        weights = train_weights if self.training and has_training_teacher else router_probs
        expert_outputs = self._expert_outputs(conditioning_tokens)
        tokens = conditioning_tokens + self._expert_residual(expert_outputs, weights)
        utility_value_scores = None
        utility_progress_scores = None
        utility_uncertainty_scores = None
        has_utility_component_targets = any(
            target is not None
            for target in [
                utility_value_targets,
                utility_progress_targets,
                utility_uncertainty_targets,
            ]
        )
        if self.utility_head is not None and (utility_scores is None or has_utility_component_targets):
            utility_output = self.utility_head(conditioning_tokens, cost_scores=utility_cost_scores)
            if utility_scores is None:
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
                candidate_mask=utility_candidate_mask if utility_candidate_mask is not None else pool_mask,
                rank_loss_weight=self.utility_rank_loss_weight,
            )
        balance_loss = uniform_balance_loss(router_probs) + uniform_balance_loss(pool_probs)
        stickiness_loss = (
            route_stickiness_loss(router_probs, previous_router_probs)
            if previous_router_probs is not None
            else router_logits.new_zeros(())
        )
        diversity_loss = expert_diversity_loss(expert_outputs)
        entropy_loss = route_entropy_regularization_loss(router_probs, pool_probs)
        route_loss_weighted = self.router_loss_weight * route_loss
        pool_loss_weighted = self.pool_loss_weight * pool_loss
        pool_coverage_loss_weighted = self.pool_coverage_loss_weight * pool_coverage_loss
        utility_loss_weighted = self.utility_loss_weight * utility_loss
        utility_rank_loss_weighted = self.utility_loss_weight * self.utility_rank_loss_weight * utility_rank_loss
        utility_head_loss_weighted = self.utility_head_loss_weight * utility_head_loss
        balance_loss_weighted = self.balance_loss_weight * balance_loss
        stickiness_loss_weighted = self.stickiness_loss_weight * stickiness_loss
        diversity_loss_weighted = self.diversity_loss_weight * diversity_loss
        entropy_loss_weighted = self.entropy_loss_weight * entropy_loss
        total_loss = (
            route_loss_weighted
            + pool_loss_weighted
            + pool_coverage_loss_weighted
            + utility_loss_weighted
            + utility_head_loss_weighted
            + balance_loss_weighted
            + stickiness_loss_weighted
            + diversity_loss_weighted
            + entropy_loss_weighted
        )
        diagnostics = route_diagnostics(
            router_probs=router_probs.detach(),
            posterior_probs=posterior_probs.detach(),
            pool_mask=pool_mask.detach(),
            active_mask=active_mask.detach(),
        )
        pool_diagnostics = pool_coverage_diagnostics(
            pool_mask=pool_mask.detach(),
            active_mask=active_mask.detach(),
            teacher_probs=pool_teacher.detach(),
            critical_threshold=self.pool_critical_threshold,
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
            diversity_loss=diversity_loss.detach(),
            entropy_loss=entropy_loss.detach(),
            pool_coverage_loss=pool_coverage_loss.detach(),
            route_loss_weighted=route_loss_weighted.detach(),
            pool_loss_weighted=pool_loss_weighted.detach(),
            pool_coverage_loss_weighted=pool_coverage_loss_weighted.detach(),
            utility_loss_weighted=utility_loss_weighted.detach(),
            utility_rank_loss_weighted=utility_rank_loss_weighted.detach(),
            utility_head_loss_weighted=utility_head_loss_weighted.detach(),
            balance_loss_weighted=balance_loss_weighted.detach(),
            stickiness_loss_weighted=stickiness_loss_weighted.detach(),
            diversity_loss_weighted=diversity_loss_weighted.detach(),
            entropy_loss_weighted=entropy_loss_weighted.detach(),
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
            pool_teacher_mass=pool_diagnostics["pool_teacher_mass"],
            active_teacher_mass=pool_diagnostics["active_teacher_mass"],
            pool_teacher_top1_match=pool_diagnostics["pool_teacher_top1_match"],
            active_teacher_top1_match=pool_diagnostics["active_teacher_top1_match"],
            pool_critical_miss_rate=pool_diagnostics["pool_critical_miss_rate"],
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
