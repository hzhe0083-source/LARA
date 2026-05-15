import math
from collections.abc import Mapping, Sequence
from typing import Optional

import torch

from Lara.model.modules.action_model.lara_moe import sparse_route_budget

ROUTE_SEQUENCE_DIAGNOSTIC_KEYS = (
    "route_switch_rate",
    "active_set_switch_rate",
    "active_set_jaccard",
    "pool_switch_rate",
    "pool_reuse_rate",
    "pool_jaccard",
    "resident_expert_fraction_mean",
    "mean_router_entropy",
)

PAPER_REQUIRED_METRIC_KEYS = (
    "success",
    "flops",
    "latency_ms",
    "vram_mb",
    "route_switch_rate",
    "pool_reuse_rate",
)

METRIC_KEY_ALIASES = {
    "success": ("success", "success_rate"),
    "return_score": ("return_score", "return"),
}


def _as_mean_float(values, name: str) -> float:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain finite values")
    return float(tensor.mean().item())


def _optional_float(value):
    return None if value is None else float(value)


def _optional_mean_by_fraction(values_by_fraction, fraction: float, name: str) -> Optional[float]:
    if values_by_fraction is None:
        return None
    for key, values in values_by_fraction.items():
        if float(key) == fraction:
            return _as_mean_float(values, f"{name}[{fraction:g}]")
    raise ValueError(f"{name} must define resident fraction {fraction:g}")


def _record_value(record: Mapping[str, object], *keys: str):
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _metric_value(record: Mapping[str, object], metric_key: str):
    return _record_value(record, *METRIC_KEY_ALIASES.get(metric_key, (metric_key,)))


def _key_aliases(keys: str | Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(keys, str):
        aliases = (keys,)
    else:
        aliases = tuple(keys)
    if not aliases:
        raise ValueError(f"{name} must not be empty")
    return aliases


def _finite_record_float(
    record: Mapping[str, object],
    *,
    record_index: int,
    keys: Sequence[str],
    label: str,
    required: bool,
) -> Optional[float]:
    value = _record_value(record, *keys)
    if value is None:
        if required:
            raise ValueError(f"record {record_index} must define {label}")
        return None
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"record {record_index} has non-numeric {label}: {value!r}") from exc
    if not math.isfinite(scalar):
        raise ValueError(f"record {record_index} must define finite {label}")
    return scalar


def _record_expert_id(
    record: Mapping[str, object],
    *,
    record_index: int,
    expert_keys: Sequence[str],
    num_experts: int,
) -> int:
    value = _record_value(record, *expert_keys)
    if value is None:
        raise ValueError(f"record {record_index} must define expert_id")
    try:
        expert_float = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"record {record_index} has non-numeric expert_id: {value!r}") from exc
    if not math.isfinite(expert_float) or not expert_float.is_integer():
        raise ValueError(f"record {record_index} expert_id must be an integer, got {value!r}")
    expert_id = int(expert_float)
    if expert_id < 0 or expert_id >= num_experts:
        raise ValueError(f"record {record_index} expert_id must be in [0, {num_experts}), got {expert_id}")
    return expert_id


def counterfactual_utility_matrix_from_records(
    records: Sequence[Mapping[str, object]],
    *,
    num_experts: int,
    context_key: str = "context_id",
    expert_keys: str | Sequence[str] = ("expert_id", "candidate_expert_id"),
    value_keys: str | Sequence[str] = (
        "utility_score",
        "utility",
        "return_score",
        "return",
        "success",
        "success_rate",
    ),
    cost_keys: str | Sequence[str] = ("utility_cost", "cost", "flops", "latency_ms"),
    cost_weight: float = 0.0,
    require_all_experts: bool = False,
    min_candidates_per_context: int = 2,
) -> dict[str, object]:
    """Build route-utility supervision tensors from counterfactual rollout records.

    Each record must describe one candidate expert evaluated for one fixed
    context. Records with the same `(context_id, expert_id)` are averaged, which
    lets repeated rollouts reduce label noise. Missing experts are marked false
    in `utility_candidate_mask`; they are not silently treated as zero utility.
    """

    if not records:
        raise ValueError("records must not be empty")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if cost_weight < 0:
        raise ValueError("cost_weight must be non-negative")
    if min_candidates_per_context <= 0:
        raise ValueError("min_candidates_per_context must be positive")

    expert_keys = _key_aliases(expert_keys, "expert_keys")
    value_keys = _key_aliases(value_keys, "value_keys")
    cost_keys = _key_aliases(cost_keys, "cost_keys")
    context_ids = []
    context_to_row = {}
    utility_sums = []
    candidate_counts = []

    for record_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError("records must contain mapping objects")
        context_id = _record_value(record, context_key)
        if context_id is None:
            raise ValueError(f"record {record_index} must define {context_key}")
        try:
            row_index = context_to_row.get(context_id)
        except TypeError as exc:
            raise ValueError(f"record {record_index} {context_key} must be hashable") from exc
        if row_index is None:
            row_index = len(context_ids)
            context_to_row[context_id] = row_index
            context_ids.append(context_id)
            utility_sums.append(torch.zeros(num_experts, dtype=torch.float32))
            candidate_counts.append(torch.zeros(num_experts, dtype=torch.long))

        expert_id = _record_expert_id(
            record,
            record_index=record_index,
            expert_keys=expert_keys,
            num_experts=num_experts,
        )
        value = _finite_record_float(
            record,
            record_index=record_index,
            keys=value_keys,
            label="utility value",
            required=True,
        )
        cost = _finite_record_float(
            record,
            record_index=record_index,
            keys=cost_keys,
            label="utility cost",
            required=cost_weight > 0,
        )
        utility = value - cost_weight * (0.0 if cost is None else cost)
        utility_sums[row_index][expert_id] += float(utility)
        candidate_counts[row_index][expert_id] += 1

    counts = torch.stack(candidate_counts)
    candidate_mask = counts > 0
    candidates_per_context = candidate_mask.sum(dim=-1)
    sparse_context_rows = torch.nonzero(candidates_per_context < min_candidates_per_context, as_tuple=False).view(-1)
    if sparse_context_rows.numel() > 0:
        sparse_contexts = [context_ids[int(index)] for index in sparse_context_rows.tolist()]
        raise ValueError(
            "counterfactual utility records must evaluate at least "
            f"{min_candidates_per_context} candidates per context; sparse contexts: {sparse_contexts}"
        )

    missing_candidates = {
        str(context_ids[row_index]): torch.nonzero(~candidate_mask[row_index], as_tuple=False).view(-1).tolist()
        for row_index in range(len(context_ids))
        if not bool(candidate_mask[row_index].all().item())
    }
    if require_all_experts and missing_candidates:
        raise ValueError(f"missing counterfactual candidates: {missing_candidates}")

    utility_scores = torch.stack(utility_sums)
    utility_scores[candidate_mask] = (
        utility_scores[candidate_mask] / counts.to(dtype=utility_scores.dtype)[candidate_mask]
    )
    utility_scores[~candidate_mask] = 0.0
    return {
        "context_ids": context_ids,
        "utility_scores": utility_scores,
        "utility_candidate_mask": candidate_mask,
        "candidate_counts": counts,
        "num_contexts": len(context_ids),
        "num_candidates": int(candidate_mask.sum().item()),
        "missing_candidates": missing_candidates,
    }


def _group_record_values_by_fraction(
    records: Sequence[Mapping[str, object]],
    *,
    resident_fraction_key: str,
    value_keys: Sequence[str],
    required: bool,
    metric_name: str,
) -> dict[float, list[float]]:
    grouped: dict[float, list[float]] = {}
    for index, record in enumerate(records):
        raw_fraction = _record_value(record, resident_fraction_key, "resident_fraction_requested")
        if raw_fraction is None:
            raise ValueError(f"record {index} must define {resident_fraction_key}")
        fraction = float(raw_fraction)
        if fraction <= 0 or fraction > 1:
            raise ValueError(f"resident fraction must be in (0, 1], got {fraction}")
        value = _record_value(record, *value_keys)
        if value is None:
            if required:
                raise ValueError(f"record {index} must define {metric_name}")
            continue
        grouped.setdefault(fraction, []).append(float(value))
    if required and not grouped:
        raise ValueError(f"{metric_name} values must not be empty")
    return grouped


def _complete_optional_group(
    values_by_fraction: dict[float, list[float]],
    required_fractions: Sequence[float],
) -> Optional[dict[float, list[float]]]:
    if not values_by_fraction:
        return None
    missing = [fraction for fraction in required_fractions if fraction not in values_by_fraction]
    if missing:
        return None
    return values_by_fraction


def _summarize_optional_record_metrics(
    records: Sequence[Mapping[str, object]],
    *,
    resident_fraction_key: str,
    fractions: Sequence[float],
    metric_keys: Sequence[str],
) -> dict[str, dict[str, float]]:
    summaries = {}
    for metric_key in metric_keys:
        values_by_fraction = _complete_optional_group(
            _group_record_values_by_fraction(
                records,
                resident_fraction_key=resident_fraction_key,
                value_keys=(metric_key,),
                required=False,
                metric_name=metric_key,
            ),
            fractions,
        )
        if values_by_fraction is None:
            continue
        summaries[metric_key] = {
            f"{fraction:g}": _as_mean_float(values_by_fraction[fraction], f"{metric_key}[{fraction:g}]")
            for fraction in fractions
        }
    return summaries


def _validate_route_sequence_tensor(tensor: torch.Tensor, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(tensor)
    if tensor.ndim != 3:
        raise ValueError(f"{name} must have shape [B, T, M], got {tuple(tensor.shape)}")
    if tensor.shape[1] == 0 or tensor.shape[2] == 0:
        raise ValueError(f"{name} must have non-empty time and expert dimensions")
    return tensor


def _route_record_sequence(record: Mapping[str, object], key: str, *, has_expert_dim: bool) -> Optional[torch.Tensor]:
    value = record.get(key)
    if value is None:
        return None
    tensor = torch.as_tensor(value)
    if has_expert_dim and tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if not has_expert_dim and tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def _sequence_valid_mask(
    valid_mask: Optional[torch.Tensor],
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    if valid_mask is None:
        return torch.ones((batch_size, sequence_length), device=device, dtype=torch.bool)
    valid_mask = torch.as_tensor(valid_mask, device=device, dtype=torch.bool)
    if valid_mask.shape != (batch_size, sequence_length):
        raise ValueError(f"valid_mask must have shape ({batch_size}, {sequence_length}), got {tuple(valid_mask.shape)}")
    return valid_mask


def _masked_mean_float(values: torch.Tensor, mask: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    mask = mask.to(device=values.device, dtype=values.dtype)
    denom = mask.sum()
    if float(denom.item()) <= 0:
        return 0.0
    return float((values * mask).sum().item() / denom.item())


def _mask_set_jaccard(current_mask: torch.Tensor, next_mask: torch.Tensor) -> torch.Tensor:
    intersection = (current_mask & next_mask).sum(dim=-1).float()
    union = (current_mask | next_mask).sum(dim=-1).float().clamp_min(1.0)
    return intersection / union


def route_sequence_diagnostics(
    router_probs: torch.Tensor,
    *,
    active_mask: Optional[torch.Tensor] = None,
    pool_mask: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    """Summarize router behavior across receding-horizon chunks.

    The inputs are rollout-level sequences, not independent training batches:
    `[B, T, M]` means B episodes, T replanning chunks per episode, and M
    experts. These diagnostics help check whether the two-level route pool is
    stable across closed-loop chunks before treating subset-retention results as
    meaningful.
    """

    router_probs = _validate_route_sequence_tensor(router_probs, "router_probs").float()
    if not torch.isfinite(router_probs).all() or torch.any(router_probs < 0):
        raise ValueError("router_probs must contain finite non-negative values")
    row_sums = router_probs.sum(dim=-1, keepdim=True)
    if torch.any(row_sums <= 0):
        raise ValueError("router_probs must have positive probability mass for every chunk")
    router_probs = router_probs / row_sums.clamp_min(1e-8)

    batch_size, sequence_length, _ = router_probs.shape
    valid_mask = _sequence_valid_mask(
        valid_mask,
        batch_size=batch_size,
        sequence_length=sequence_length,
        device=router_probs.device,
    )
    pair_mask = valid_mask[:, 1:] & valid_mask[:, :-1] if sequence_length > 1 else valid_mask[:, :0]

    route_ids = router_probs.argmax(dim=-1)
    route_switches = (route_ids[:, 1:] != route_ids[:, :-1]).float() if sequence_length > 1 else router_probs.new_zeros((batch_size, 0))
    entropy_values = -(router_probs * torch.log(router_probs.clamp_min(1e-8))).sum(dim=-1)
    diagnostics = {
        "valid_chunks": float(valid_mask.sum().item()),
        "valid_transitions": float(pair_mask.sum().item()),
        "route_switch_rate": _masked_mean_float(route_switches, pair_mask),
        "mean_router_entropy": _masked_mean_float(entropy_values, valid_mask),
    }

    if active_mask is not None:
        active_mask = _validate_route_sequence_tensor(active_mask, "active_mask").to(
            device=router_probs.device,
            dtype=torch.bool,
        )
        if active_mask.shape != router_probs.shape:
            raise ValueError(f"active_mask must have shape {tuple(router_probs.shape)}, got {tuple(active_mask.shape)}")
        if not torch.all(active_mask.any(dim=-1)):
            raise ValueError("active_mask must select at least one expert for every chunk")
        active_changes = torch.any(active_mask[:, 1:] != active_mask[:, :-1], dim=-1).float()
        active_jaccard = _mask_set_jaccard(active_mask[:, :-1], active_mask[:, 1:])
        diagnostics["active_set_switch_rate"] = _masked_mean_float(active_changes, pair_mask)
        diagnostics["active_set_jaccard"] = _masked_mean_float(active_jaccard, pair_mask)

    if pool_mask is not None:
        pool_mask = _validate_route_sequence_tensor(pool_mask, "pool_mask").to(
            device=router_probs.device,
            dtype=torch.bool,
        )
        if pool_mask.shape != router_probs.shape:
            raise ValueError(f"pool_mask must have shape {tuple(router_probs.shape)}, got {tuple(pool_mask.shape)}")
        if not torch.all(pool_mask.any(dim=-1)):
            raise ValueError("pool_mask must select at least one expert for every chunk")
        pool_changes = torch.any(pool_mask[:, 1:] != pool_mask[:, :-1], dim=-1).float()
        pool_jaccard = _mask_set_jaccard(pool_mask[:, :-1], pool_mask[:, 1:])
        pool_switch_rate = _masked_mean_float(pool_changes, pair_mask)
        diagnostics["pool_switch_rate"] = pool_switch_rate
        diagnostics["pool_reuse_rate"] = 1.0 - pool_switch_rate
        diagnostics["pool_jaccard"] = _masked_mean_float(pool_jaccard, pair_mask)
        diagnostics["resident_expert_fraction_mean"] = _masked_mean_float(
            pool_mask.float().mean(dim=-1),
            valid_mask,
        )

    return diagnostics


def rollout_record_with_route_diagnostics(
    record: Mapping[str, object],
    *,
    router_probs_key: str = "router_probs_sequence",
    active_mask_key: str = "active_mask_sequence",
    pool_mask_key: str = "pool_mask_sequence",
    valid_mask_key: str = "valid_mask_sequence",
    overwrite: bool = False,
) -> dict[str, object]:
    """Return a JSON-style rollout record with route-sequence diagnostics added.

    A single record may store one episode as `[T, M]` route arrays or a batched
    collection as `[B, T, M]`. Existing scalar diagnostic fields are preserved by
    default so hand-verified rollout logs are not silently rewritten.
    """

    enriched = dict(record)
    router_probs = _route_record_sequence(record, router_probs_key, has_expert_dim=True)
    if router_probs is None:
        return enriched

    diagnostics = route_sequence_diagnostics(
        router_probs,
        active_mask=_route_record_sequence(record, active_mask_key, has_expert_dim=True),
        pool_mask=_route_record_sequence(record, pool_mask_key, has_expert_dim=True),
        valid_mask=_route_record_sequence(record, valid_mask_key, has_expert_dim=False),
    )
    for key, value in diagnostics.items():
        if overwrite or key not in enriched:
            enriched[key] = value
    return enriched


def normalize_protocol_records(
    records: Sequence[Mapping[str, object]],
    *,
    add_route_diagnostics: bool = True,
) -> list[dict[str, object]]:
    """Normalize raw rollout records before paper-protocol summarization."""

    normalized = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("records must contain mapping objects")
        if add_route_diagnostics:
            normalized.append(rollout_record_with_route_diagnostics(record))
        else:
            normalized.append(dict(record))
    return normalized


def protocol_evidence_audit(
    records: Sequence[Mapping[str, object]],
    *,
    resident_fraction_key: str = "resident_fraction",
    required_fractions: Optional[Sequence[float]] = None,
    required_metric_keys: Sequence[str] = PAPER_REQUIRED_METRIC_KEYS,
    add_route_diagnostics: bool = True,
) -> dict[str, object]:
    """Audit whether rollout records are strong enough for paper-protocol claims."""

    records = normalize_protocol_records(records, add_route_diagnostics=add_route_diagnostics)
    fractions = []
    missing_fraction_records = []
    for index, record in enumerate(records):
        raw_fraction = _record_value(record, resident_fraction_key, "resident_fraction_requested")
        if raw_fraction is None:
            missing_fraction_records.append(index)
            continue
        fraction = float(raw_fraction)
        if fraction <= 0 or fraction > 1:
            raise ValueError(f"resident fraction must be in (0, 1], got {fraction}")
        fractions.append(fraction)

    present_fractions = sorted(set(fractions))
    required_fractions = (
        present_fractions if required_fractions is None else sorted({float(fraction) for fraction in required_fractions})
    )
    for fraction in required_fractions:
        if fraction <= 0 or fraction > 1:
            raise ValueError(f"required resident fraction must be in (0, 1], got {fraction}")

    num_records_by_fraction = {
        f"{fraction:g}": sum(1 for observed_fraction in fractions if observed_fraction == fraction)
        for fraction in present_fractions
    }
    missing_required_fractions = [
        f"{fraction:g}" for fraction in required_fractions if fraction not in present_fractions
    ]
    missing_metrics_by_fraction = {}
    for fraction in required_fractions:
        fraction_records = [
            record
            for record in records
            if _record_value(record, resident_fraction_key, "resident_fraction_requested") is not None
            and float(_record_value(record, resident_fraction_key, "resident_fraction_requested")) == fraction
        ]
        if not fraction_records:
            continue
        missing_metrics = [
            metric_key
            for metric_key in required_metric_keys
            if not all(_metric_value(record, metric_key) is not None for record in fraction_records)
        ]
        if missing_metrics:
            missing_metrics_by_fraction[f"{fraction:g}"] = missing_metrics

    ok = not missing_fraction_records and not missing_required_fractions and not missing_metrics_by_fraction
    return {
        "ok": ok,
        "num_records": len(records),
        "num_records_by_fraction": num_records_by_fraction,
        "required_fractions": [f"{fraction:g}" for fraction in required_fractions],
        "required_metric_keys": list(required_metric_keys),
        "missing_fraction_records": missing_fraction_records,
        "missing_required_fractions": missing_required_fractions,
        "missing_metrics_by_fraction": missing_metrics_by_fraction,
    }


def resident_experts_for_fraction(total_experts: int, resident_fraction: float) -> int:
    if total_experts <= 0:
        raise ValueError("total_experts must be positive")
    resident_fraction = float(resident_fraction)
    if resident_fraction <= 0 or resident_fraction > 1:
        raise ValueError(f"resident fraction must be in (0, 1], got {resident_fraction}")
    return max(1, min(total_experts, int(math.ceil(total_experts * resident_fraction))))


def subset_retention_success_curve(
    success_by_fraction: Mapping[float, Sequence[float] | torch.Tensor],
) -> dict[str, float]:
    if not success_by_fraction:
        raise ValueError("success_by_fraction must not be empty")

    points = []
    summary = {}
    for fraction, successes in sorted(success_by_fraction.items()):
        fraction = float(fraction)
        if fraction <= 0 or fraction > 1:
            raise ValueError(f"resident fraction must be in (0, 1], got {fraction}")
        success_rate = _as_mean_float(successes, f"successes[{fraction:g}]")
        points.append((fraction, success_rate))
        summary[f"success_at_resident_{fraction:g}"] = success_rate

    if len(points) == 1:
        summary["subset_retention_auc"] = points[0][1]
    else:
        area = 0.0
        for (left_fraction, left_success), (right_fraction, right_success) in zip(points, points[1:]):
            area += 0.5 * (left_success + right_success) * (right_fraction - left_fraction)
        summary["subset_retention_auc"] = area / max(points[-1][0] - points[0][0], 1e-8)

    if 1.0 in success_by_fraction:
        full_success = summary["success_at_resident_1"]
        min_fraction, min_success = points[0]
        summary[f"success_drop_1_to_{min_fraction:g}"] = full_success - min_success
    return summary


def matched_compute_row(
    benchmark: str,
    method: str,
    *,
    success_rate: Optional[float] = None,
    return_score: Optional[float] = None,
    route_regret: Optional[float] = None,
    flops: Optional[float] = None,
    latency_ms: Optional[float] = None,
    vram_mb: Optional[float] = None,
    total_experts: Optional[int] = None,
    active_experts: Optional[int] = None,
    resident_experts: Optional[int] = None,
    shared_params: int = 0,
    params_per_expert: int = 0,
    total_params: Optional[float] = None,
    active_params: Optional[float] = None,
    resident_params: Optional[float] = None,
) -> dict[str, float | str | None]:
    active_expert_fraction = None
    resident_expert_fraction = None
    if total_experts is not None or active_experts is not None:
        if total_experts is None or active_experts is None:
            raise ValueError("total_experts and active_experts must be provided together")
        budget = sparse_route_budget(
            total_experts=total_experts,
            active_experts=active_experts,
            resident_experts=resident_experts,
            shared_params=shared_params,
            params_per_expert=params_per_expert,
        )
        total_params = budget["total_params"]
        active_params = budget["active_params"]
        resident_params = budget["resident_params"]
        active_param_fraction = budget["active_param_fraction"]
        resident_param_fraction = budget["resident_param_fraction"]
        active_expert_fraction = budget["active_expert_fraction"]
        resident_expert_fraction = budget["resident_expert_fraction"]
        resident_experts = total_experts if resident_experts is None else resident_experts
    else:
        active_param_fraction = (
            None
            if active_params is None or total_params is None or total_params == 0
            else float(active_params) / float(total_params)
        )
        resident_param_fraction = (
            None
            if resident_params is None or total_params is None or total_params == 0
            else float(resident_params) / float(total_params)
        )

    return {
        "benchmark": benchmark,
        "method": method,
        "success_rate": _optional_float(success_rate),
        "return_score": _optional_float(return_score),
        "route_regret": _optional_float(route_regret),
        "total_experts": None if total_experts is None else int(total_experts),
        "active_experts": None if active_experts is None else int(active_experts),
        "resident_experts": None if resident_experts is None else int(resident_experts),
        "active_expert_fraction": _optional_float(active_expert_fraction),
        "resident_expert_fraction": _optional_float(resident_expert_fraction),
        "total_params": _optional_float(total_params),
        "active_params": _optional_float(active_params),
        "resident_params": _optional_float(resident_params),
        "active_param_fraction": _optional_float(active_param_fraction),
        "resident_param_fraction": _optional_float(resident_param_fraction),
        "flops": _optional_float(flops),
        "latency_ms": _optional_float(latency_ms),
        "vram_mb": _optional_float(vram_mb),
    }


def subset_retention_rows(
    benchmark: str,
    method: str,
    *,
    success_by_fraction: Mapping[float, Sequence[float] | torch.Tensor],
    total_experts: int,
    active_experts: int,
    shared_params: int = 0,
    params_per_expert: int = 0,
    return_by_fraction: Optional[Mapping[float, Sequence[float] | torch.Tensor]] = None,
    route_regret_by_fraction: Optional[Mapping[float, Sequence[float] | torch.Tensor]] = None,
    flops_by_fraction: Optional[Mapping[float, Sequence[float] | torch.Tensor]] = None,
    latency_ms_by_fraction: Optional[Mapping[float, Sequence[float] | torch.Tensor]] = None,
    vram_mb_by_fraction: Optional[Mapping[float, Sequence[float] | torch.Tensor]] = None,
) -> list[dict[str, float | str | None]]:
    if not success_by_fraction:
        raise ValueError("success_by_fraction must not be empty")

    rows = []
    for raw_fraction, successes in sorted(success_by_fraction.items(), key=lambda item: float(item[0])):
        fraction = float(raw_fraction)
        resident_experts = resident_experts_for_fraction(total_experts, fraction)
        row = matched_compute_row(
            benchmark=benchmark,
            method=f"{method} resident={fraction:g}",
            success_rate=_as_mean_float(successes, f"successes[{fraction:g}]"),
            return_score=_optional_mean_by_fraction(return_by_fraction, fraction, "returns"),
            route_regret=_optional_mean_by_fraction(route_regret_by_fraction, fraction, "route_regret"),
            flops=_optional_mean_by_fraction(flops_by_fraction, fraction, "flops"),
            latency_ms=_optional_mean_by_fraction(latency_ms_by_fraction, fraction, "latency_ms"),
            vram_mb=_optional_mean_by_fraction(vram_mb_by_fraction, fraction, "vram_mb"),
            total_experts=total_experts,
            active_experts=active_experts,
            resident_experts=resident_experts,
            shared_params=shared_params,
            params_per_expert=params_per_expert,
        )
        row["resident_fraction_requested"] = fraction
        rows.append(row)
    return rows


def protocol_summary_from_records(
    records: Sequence[Mapping[str, object]],
    *,
    benchmark: str,
    method: str,
    total_experts: int,
    active_experts: int,
    shared_params: int = 0,
    params_per_expert: int = 0,
    resident_fraction_key: str = "resident_fraction",
    success_keys: Sequence[str] = ("success", "success_rate"),
    add_route_diagnostics: bool = True,
) -> dict[str, object]:
    if not records:
        raise ValueError("records must not be empty")

    records = normalize_protocol_records(records, add_route_diagnostics=add_route_diagnostics)
    success_by_fraction = _group_record_values_by_fraction(
        records,
        resident_fraction_key=resident_fraction_key,
        value_keys=success_keys,
        required=True,
        metric_name="success",
    )
    fractions = sorted(success_by_fraction)
    optional_groups = {
        "return_by_fraction": _complete_optional_group(
            _group_record_values_by_fraction(
                records,
                resident_fraction_key=resident_fraction_key,
                value_keys=("return_score", "return"),
                required=False,
                metric_name="return_score",
            ),
            fractions,
        ),
        "route_regret_by_fraction": _complete_optional_group(
            _group_record_values_by_fraction(
                records,
                resident_fraction_key=resident_fraction_key,
                value_keys=("route_regret",),
                required=False,
                metric_name="route_regret",
            ),
            fractions,
        ),
        "flops_by_fraction": _complete_optional_group(
            _group_record_values_by_fraction(
                records,
                resident_fraction_key=resident_fraction_key,
                value_keys=("flops",),
                required=False,
                metric_name="flops",
            ),
            fractions,
        ),
        "latency_ms_by_fraction": _complete_optional_group(
            _group_record_values_by_fraction(
                records,
                resident_fraction_key=resident_fraction_key,
                value_keys=("latency_ms",),
                required=False,
                metric_name="latency_ms",
            ),
            fractions,
        ),
        "vram_mb_by_fraction": _complete_optional_group(
            _group_record_values_by_fraction(
                records,
                resident_fraction_key=resident_fraction_key,
                value_keys=("vram_mb",),
                required=False,
                metric_name="vram_mb",
            ),
            fractions,
        ),
    }
    rows = subset_retention_rows(
        benchmark=benchmark,
        method=method,
        success_by_fraction=success_by_fraction,
        total_experts=total_experts,
        active_experts=active_experts,
        shared_params=shared_params,
        params_per_expert=params_per_expert,
        **optional_groups,
    )
    if all(row.get("flops") is not None for row in rows):
        for row, frontier in zip(rows, pareto_frontier_flags(rows), strict=True):
            row["compute_success_pareto"] = frontier
    curve = subset_retention_success_curve(success_by_fraction)
    summary = {
        "benchmark": benchmark,
        "method": method,
        "num_records": len(records),
        "num_records_by_fraction": {f"{fraction:g}": len(success_by_fraction[fraction]) for fraction in fractions},
        "curve": curve,
        "rows": rows,
    }
    route_diagnostics = _summarize_optional_record_metrics(
        records,
        resident_fraction_key=resident_fraction_key,
        fractions=fractions,
        metric_keys=ROUTE_SEQUENCE_DIAGNOSTIC_KEYS,
    )
    if route_diagnostics:
        summary["route_diagnostics_by_fraction"] = route_diagnostics
    return summary


def _relative_match(reference: float, candidate: float, tolerance: float) -> bool:
    denominator = max(abs(reference), 1e-8)
    return abs(candidate - reference) / denominator <= tolerance


def matched_budget_flags(
    reference_row: Mapping[str, float | str | None],
    candidate_row: Mapping[str, float | str | None],
    *,
    active_tolerance: float = 0.05,
    resident_tolerance: float = 0.05,
) -> dict[str, bool]:
    if active_tolerance < 0 or resident_tolerance < 0:
        raise ValueError("tolerances must be non-negative")
    for key in ["active_params", "resident_params"]:
        if reference_row.get(key) is None or candidate_row.get(key) is None:
            raise ValueError(f"Both rows must define {key}")

    active_params_matched = _relative_match(
        float(reference_row["active_params"]),
        float(candidate_row["active_params"]),
        active_tolerance,
    )
    resident_params_matched = _relative_match(
        float(reference_row["resident_params"]),
        float(candidate_row["resident_params"]),
        resident_tolerance,
    )
    return {
        "active_params_matched": active_params_matched,
        "resident_params_matched": resident_params_matched,
        "matched_compute": active_params_matched and resident_params_matched,
    }


def matched_expert_budget_flags(
    reference_row: Mapping[str, float | str | None],
    candidate_row: Mapping[str, float | str | None],
    *,
    active_tolerance: float = 0.0,
    resident_tolerance: float = 0.0,
) -> dict[str, bool]:
    if active_tolerance < 0 or resident_tolerance < 0:
        raise ValueError("tolerances must be non-negative")
    for key in ["active_experts", "resident_experts"]:
        if reference_row.get(key) is None or candidate_row.get(key) is None:
            raise ValueError(f"Both rows must define {key}")

    active_experts_matched = _relative_match(
        float(reference_row["active_experts"]),
        float(candidate_row["active_experts"]),
        active_tolerance,
    )
    resident_experts_matched = _relative_match(
        float(reference_row["resident_experts"]),
        float(candidate_row["resident_experts"]),
        resident_tolerance,
    )
    return {
        "active_experts_matched": active_experts_matched,
        "resident_experts_matched": resident_experts_matched,
        "matched_expert_budget": active_experts_matched and resident_experts_matched,
    }


def pareto_frontier_flags(
    rows: Sequence[Mapping[str, float | str | None]],
    *,
    success_key: str = "success_rate",
    cost_key: str = "flops",
) -> list[bool]:
    if not rows:
        raise ValueError("rows must not be empty")

    values = []
    for row in rows:
        if row.get(success_key) is None or row.get(cost_key) is None:
            raise ValueError(f"Every row must define {success_key} and {cost_key}")
        values.append((float(row[success_key]), float(row[cost_key])))

    flags = []
    for idx, (success, cost) in enumerate(values):
        dominated = False
        for other_idx, (other_success, other_cost) in enumerate(values):
            if other_idx == idx:
                continue
            no_worse = other_success >= success and other_cost <= cost
            strictly_better = other_success > success or other_cost < cost
            if no_worse and strictly_better:
                dominated = True
                break
        flags.append(not dominated)
    return flags
