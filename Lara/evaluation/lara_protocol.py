import math
from collections.abc import Mapping, Sequence
from typing import Optional

import torch

from Lara.model.modules.action_model.lara_moe import sparse_route_budget


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
