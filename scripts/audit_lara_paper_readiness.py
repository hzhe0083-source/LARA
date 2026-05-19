#!/usr/bin/env python3
"""Audit whether local artifacts justify claiming the LARA paper method is ready."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

try:
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover - exercised in lightweight runtime environments.
    OmegaConf = None

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Lara.evaluation import counterfactual_utility_matrix_from_records  # noqa: E402


def _repo_path(path: str | Path | None) -> Path | None:
    if path is None or str(path) == "":
        return None
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _value(cfg: Any, dotted_key: str, default: Any = None) -> Any:
    current = cfg
    for key in dotted_key.split("."):
        if current is None or key not in current:
            return default
        current = current[key]
    return current


def _check(name: str, ok: bool, *, detail: str, severity: str = "required", **extra: Any) -> dict[str, Any]:
    payload = {
        "name": name,
        "ok": bool(ok),
        "severity": severity,
        "detail": detail,
    }
    payload.update(extra)
    return payload


def _load_json_object(path: Path, artifact_name: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{artifact_name} must not be empty")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"{artifact_name} must be a JSON object")
    return payload


def _artifact_value(payload: dict[str, Any], dotted_keys: str | tuple[str, ...], default: Any = None) -> Any:
    if isinstance(dotted_keys, str):
        dotted_keys = (dotted_keys,)
    for dotted_key in dotted_keys:
        value = _value(payload, dotted_key, None)
        if value is not None:
            return value
    return default


def _artifact_bool(payload: dict[str, Any], dotted_keys: str | tuple[str, ...], default: bool = False) -> bool:
    value = _artifact_value(payload, dotted_keys, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "ok", "completed"}
    return bool(value)


def _finite_float(value: Any) -> float | None:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    return scalar if math.isfinite(scalar) else None


def load_records(path: str | Path) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("input records file must not be empty")
    if text[0] == "[":
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError("input must be a JSON list of objects or JSONL object records")
    return records


PROTOCOL_METRIC_KEY_ALIASES = {
    "success": ("success", "success_rate"),
    "teacher_mass_at_resident_pool": ("teacher_mass_at_resident_pool", "pool_teacher_mass", "posterior_pool_mass"),
    "critical_expert_miss_rate": (
        "critical_expert_miss_rate",
        "pool_critical_miss_rate",
        "posterior_pool_critical_miss_rate",
    ),
}

PROTOCOL_REQUIRED_METRIC_KEYS = (
    "success",
    "flops",
    "latency_ms",
    "vram_mb",
    "route_switch_rate",
    "pool_reuse_rate",
    "posterior_router_kl",
    "teacher_mass_at_resident_pool",
    "critical_expert_miss_rate",
)


def _record_value(record: Mapping[str, object], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _metric_value(record: Mapping[str, object], metric_key: str) -> Any:
    return _record_value(record, *PROTOCOL_METRIC_KEY_ALIASES.get(metric_key, (metric_key,)))


def _sidecar_metadata(
    records: list[dict[str, Any]],
    *,
    num_experts: int,
    cost_weight: float,
    require_all_experts: bool,
    min_candidates_per_context: int,
) -> dict[str, Any]:
    if not records:
        raise ValueError("records must not be empty")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if cost_weight < 0:
        raise ValueError("cost_weight must be non-negative")
    context_candidates: dict[object, set[int]] = {}
    for record_index, record in enumerate(records):
        context_id = _record_value(record, "context_id")
        if context_id is None:
            raise ValueError(f"record {record_index} must define context_id")
        expert_id = _record_value(record, "expert_id", "candidate_expert_id")
        if expert_id is None:
            raise ValueError(f"record {record_index} must define expert_id")
        expert_float = _finite_float(expert_id)
        if expert_float is None or not expert_float.is_integer():
            raise ValueError(f"record {record_index} expert_id must be an integer, got {expert_id!r}")
        expert_id = int(expert_float)
        if expert_id < 0 or expert_id >= num_experts:
            raise ValueError(f"record {record_index} expert_id must be in [0, {num_experts}), got {expert_id}")
        if _finite_float(_record_value(record, "utility_score", "utility", "return_score", "return", "success")) is None:
            raise ValueError(f"record {record_index} must define finite utility value")
        if (
            cost_weight > 0
            and _finite_float(_record_value(record, "utility_cost", "cost", "flops", "latency_ms")) is None
        ):
            raise ValueError(f"record {record_index} must define finite utility cost")
        context_candidates.setdefault(context_id, set()).add(expert_id)

    sparse_contexts = [
        str(context_id)
        for context_id, candidates in context_candidates.items()
        if len(candidates) < min_candidates_per_context
    ]
    if sparse_contexts:
        raise ValueError(
            "counterfactual utility records must evaluate at least "
            f"{min_candidates_per_context} candidates per context; sparse contexts: {sparse_contexts}"
        )
    missing_candidates = {
        str(context_id): sorted(set(range(num_experts)) - candidates)
        for context_id, candidates in context_candidates.items()
        if len(candidates) < num_experts
    }
    if require_all_experts and missing_candidates:
        raise ValueError(f"missing counterfactual candidates: {missing_candidates}")
    return {
        "num_contexts": len(context_candidates),
        "num_candidates": sum(len(candidates) for candidates in context_candidates.values()),
        "missing_candidates": missing_candidates,
    }


def _argmax(values: Sequence[object]) -> int:
    return max(range(len(values)), key=lambda index: float(values[index]))


def _route_sequence_diagnostics(record: dict[str, Any]) -> dict[str, float]:
    diagnostics = {}
    router_probs = record.get("router_probs_sequence")
    if isinstance(router_probs, Sequence) and len(router_probs) > 0:
        route_ids = [_argmax(row) for row in router_probs]
        transitions = max(len(route_ids) - 1, 0)
        if transitions > 0:
            switches = sum(1 for left, right in pairwise(route_ids) if left != right)
            diagnostics["route_switch_rate"] = switches / transitions
        else:
            diagnostics["route_switch_rate"] = 0.0

    pool_mask = record.get("pool_mask_sequence")
    if isinstance(pool_mask, Sequence) and len(pool_mask) > 0:
        pool_rows = [tuple(bool(value) for value in row) for row in pool_mask]
        transitions = max(len(pool_rows) - 1, 0)
        if transitions > 0:
            switches = sum(1 for left, right in pairwise(pool_rows) if left != right)
            diagnostics["pool_reuse_rate"] = 1.0 - switches / transitions
        else:
            diagnostics["pool_reuse_rate"] = 1.0
    return diagnostics


def _protocol_evidence_audit(
    records: list[dict[str, Any]],
    *,
    required_fractions: list[float] | None,
    add_route_diagnostics: bool,
) -> dict[str, Any]:
    if add_route_diagnostics:
        records = [{**record, **_route_sequence_diagnostics(record)} for record in records]
    fractions = []
    missing_fraction_records = []
    for index, record in enumerate(records):
        raw_fraction = _record_value(record, "resident_fraction", "resident_fraction_requested")
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
            if _record_value(record, "resident_fraction", "resident_fraction_requested") is not None
            and float(_record_value(record, "resident_fraction", "resident_fraction_requested")) == fraction
        ]
        if not fraction_records:
            continue
        missing_metrics = [
            metric_key
            for metric_key in PROTOCOL_REQUIRED_METRIC_KEYS
            if not all(_metric_value(record, metric_key) is not None for record in fraction_records)
        ]
        if missing_metrics:
            missing_metrics_by_fraction[f"{fraction:g}"] = missing_metrics

    return {
        "ok": not missing_fraction_records and not missing_required_fractions and not missing_metrics_by_fraction,
        "num_records": len(records),
        "num_records_by_fraction": num_records_by_fraction,
        "required_fractions": [f"{fraction:g}" for fraction in required_fractions],
        "required_metric_keys": list(PROTOCOL_REQUIRED_METRIC_KEYS),
        "missing_fraction_records": missing_fraction_records,
        "missing_required_fractions": missing_required_fractions,
        "missing_metrics_by_fraction": missing_metrics_by_fraction,
    }


def parse_resident_fractions(value: str | None) -> list[float] | None:
    if value is None:
        return None
    fractions = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not fractions:
        raise ValueError("required resident fractions must not be empty")
    return fractions


def _load_config(path: Path) -> Any:
    if OmegaConf is not None:
        return OmegaConf.load(path)
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _config_checks(cfg: Any, config_path: Path) -> list[dict[str, Any]]:
    action_cfg = _value(cfg, "framework.action_model", {})
    dataset_cfg = _value(cfg, "datasets.vla_data", {})
    trainer_cfg = _value(cfg, "trainer", {})
    pretrained = str(_value(trainer_cfg, "pretrained_checkpoint", ""))
    required_lara_flags = (
        "use_latent_action_head",
        "lara_use_transition_head",
        "use_lara_moe",
        "lara_use_action_loss_utility",
        "lara_use_action_loss_utility_components",
        "lara_use_state_utility",
        "lara_use_state_utility_components",
        "lara_use_utility_head",
        "lara_use_direct_action_experts",
        "lara_use_direct_action_output",
    )
    disabled_lara_flags = {
        key: _value(action_cfg, key)
        for key in required_lara_flags
        if _value(action_cfg, key) is not True
    }
    required_positive_weight_keys = (
        "lara_transition_loss_weight",
        "lara_pool_coverage_loss_weight",
        "lara_utility_loss_weight",
        "lara_utility_rank_loss_weight",
        "lara_utility_head_loss_weight",
    )
    nonpositive_lara_weights = {
        key: float(_value(action_cfg, key, 0.0) or 0.0)
        for key in required_positive_weight_keys
        if float(_value(action_cfg, key, 0.0) or 0.0) <= 0.0
    }
    optional_zero_weight_keys = (
        "lara_balance_loss_weight",
        "lara_stickiness_loss_weight",
        "lara_diversity_loss_weight",
        "lara_entropy_loss_weight",
        "lara_inference_stickiness_weight",
    )
    nonzero_optional_weights = {
        key: float(_value(action_cfg, key, 0.0))
        for key in optional_zero_weight_keys
        if float(_value(action_cfg, key, 0.0) or 0.0) != 0.0
    }

    expected_horizons = {
        "future_action_window_size": 59,
        "action_horizon": 60,
        "execution_horizon": 10,
        "latent_action_horizon": 10,
        "router_horizon": 10,
        "utility_horizon": 10,
        "long_prediction_aux_horizon": 60,
    }
    horizon_values = {key: _value(action_cfg, key) for key in expected_horizons}
    horizon_mismatches = {
        key: {"expected": expected, "actual": horizon_values[key]}
        for key, expected in expected_horizons.items()
        if horizon_values[key] != expected
    }

    return [
        _check(
            "config_file_exists",
            config_path.exists(),
            detail=f"Config path: {config_path}",
        ),
        _check(
            "so101_pretrain_not_realworld",
            "Pretrain" in pretrained and "Real" not in pretrained and "real" not in pretrained,
            detail="SO101 should reuse VLA-JEPA Pretrain, not a Real-world embodiment checkpoint.",
            pretrained_checkpoint=pretrained,
        ),
        _check(
            "lara_defaults_enabled",
            not disabled_lara_flags and not nonpositive_lara_weights,
            detail=(
                "Default configs should instantiate the latent-action, MoE/router, direct-expert, "
                "transition, and utility paths; paper readiness still requires real training and rollout evidence."
            ),
            disabled_flags=disabled_lara_flags,
            nonpositive_weights=nonpositive_lara_weights,
            required_flags=list(required_lara_flags),
            required_positive_weights=list(required_positive_weight_keys),
            nonzero_optional_weights=nonzero_optional_weights,
        ),
        _check(
            "so101_horizon_contract",
            not horizon_mismatches,
            detail="Expected 30Hz long-prediction/short-execution horizons are H_p=60 and H_e=10.",
            expected=expected_horizons,
            actual=horizon_values,
            mismatches=horizon_mismatches,
        ),
        _check(
            "so101_dataset_contract",
            _value(dataset_cfg, "dataset_py") == "lerobot_datasets"
            and _value(dataset_cfg, "data_mix") == "so101_single_arm"
            and _value(action_cfg, "action_dim") == 7
            and _value(action_cfg, "state_dim") == 8,
            detail="SO101 path should train follower-arm action/state from the local LeRobot dataset.",
            dataset_py=_value(dataset_cfg, "dataset_py"),
            data_mix=_value(dataset_cfg, "data_mix"),
            action_dim=_value(action_cfg, "action_dim"),
            state_dim=_value(action_cfg, "state_dim"),
        ),
    ]


def _sidecar_check(path: Path | None, cfg: Any) -> dict[str, Any]:
    if path is None:
        return _check(
            "counterfactual_utility_sidecar",
            False,
            detail="No real counterfactual utility sidecar was provided.",
            missing="Provide --counterfactual-utility-labels or datasets.vla_data.counterfactual_utility_labels_path.",
        )
    if not path.exists():
        return _check(
            "counterfactual_utility_sidecar",
            False,
            detail=f"Counterfactual utility sidecar does not exist: {path}",
            path=str(path),
        )

    action_cfg = _value(cfg, "framework.action_model", {})
    dataset_cfg = _value(cfg, "datasets.vla_data", {})
    records = load_records(path)
    matrix = counterfactual_utility_matrix_from_records(
        records,
        num_experts=int(_value(action_cfg, "lara_num_experts", 0)),
        cost_weight=float(_value(dataset_cfg, "counterfactual_utility_cost_weight", 0.0) or 0.0),
        require_all_experts=bool(_value(dataset_cfg, "counterfactual_utility_require_all_experts", False)),
        min_candidates_per_context=int(_value(dataset_cfg, "counterfactual_utility_min_candidates_per_context", 2)),
    )
    return _check(
        "counterfactual_utility_sidecar",
        True,
        detail="Counterfactual utility sidecar validates with multi-candidate contexts.",
        path=str(path),
        num_records=len(records),
        num_contexts=matrix["num_contexts"],
        num_candidates=matrix["num_candidates"],
        missing_candidates=matrix["missing_candidates"],
    )


def _rollout_protocol_check(
    path: Path | None,
    *,
    required_fractions: list[float] | None,
    add_route_diagnostics: bool,
) -> dict[str, Any]:
    if path is None:
        return _check(
            "closed_loop_protocol_records",
            False,
            detail="No closed-loop rollout records were provided for matched-resident protocol evidence.",
            missing="Provide --rollout-records with success, FLOPs, latency, VRAM, and route sequences.",
        )
    if not path.exists():
        return _check(
            "closed_loop_protocol_records",
            False,
            detail=f"Closed-loop rollout records do not exist: {path}",
            path=str(path),
        )
    records = load_records(path)
    audit = _protocol_evidence_audit(
        records,
        required_fractions=required_fractions,
        add_route_diagnostics=add_route_diagnostics,
    )
    return _check(
        "closed_loop_protocol_records",
        bool(audit["ok"]),
        detail="Closed-loop rollout records must support paper protocol claims.",
        path=str(path),
        evidence_audit=audit,
    )


def _existing_path(path_value: Any, *, relative_to: Path) -> str | None:
    if path_value is None or str(path_value) == "":
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = (relative_to / path).resolve()
    return str(path) if path.exists() else None


def _training_artifact_check(path: Path | None, cfg: Any, *, min_steps: int) -> dict[str, Any]:
    name = "full_so101_training_artifact"
    detail = "Full SO101 training evidence for latent/MoE paths is required before paper-complete claims."
    if path is None:
        return _check(name, False, detail=detail, missing=f"Provide --{name.replace('_', '-')}.")
    if not path.exists():
        return _check(name, False, detail=f"{detail} Path does not exist: {path}", path=str(path))

    payload = _load_json_object(path, name)
    artifact_dir = path.parent
    status = str(_artifact_value(payload, ("status", "training_status"), "")).lower()
    data_mix = _artifact_value(payload, ("config.datasets.vla_data.data_mix", "dataset.data_mix", "data_mix"))
    train_steps = int(_artifact_value(payload, ("train_steps_completed", "global_step", "trainer.global_step"), 0) or 0)
    final_metrics = _artifact_value(payload, ("final_metrics", "metrics"), {})
    if not isinstance(final_metrics, dict):
        final_metrics = {}

    required_flags = {
        "use_latent_action_head": (
            "config.framework.action_model.use_latent_action_head",
            "features.use_latent_action_head",
            "use_latent_action_head",
        ),
        "lara_use_transition_head": (
            "config.framework.action_model.lara_use_transition_head",
            "features.lara_use_transition_head",
            "lara_use_transition_head",
        ),
        "use_lara_moe": (
            "config.framework.action_model.use_lara_moe",
            "features.use_lara_moe",
            "use_lara_moe",
        ),
        "lara_use_direct_action_experts": (
            "config.framework.action_model.lara_use_direct_action_experts",
            "features.lara_use_direct_action_experts",
            "lara_use_direct_action_experts",
        ),
        "lara_use_expert_loss_posterior": (
            "config.framework.action_model.lara_use_expert_loss_posterior",
            "features.lara_use_expert_loss_posterior",
            "lara_use_expert_loss_posterior",
        ),
    }
    missing_flags = [
        flag_name for flag_name, keys in required_flags.items() if not _artifact_bool(payload, keys)
    ]
    uses_utility_labels = _artifact_bool(
        payload,
        (
            "uses_counterfactual_utility_labels",
            "features.uses_counterfactual_utility_labels",
        ),
    ) or bool(_artifact_value(payload, "config.datasets.vla_data.counterfactual_utility_labels_path"))
    utility_loss_weight = _finite_float(
        _artifact_value(payload, "config.framework.action_model.lara_utility_loss_weight", 0.0)
    )
    uses_utility_training = uses_utility_labels and utility_loss_weight is not None and utility_loss_weight > 0
    transition_target_type = str(
        _artifact_value(
            payload,
            (
                "transition_target_type",
                "features.transition_target_type",
                "config.framework.action_model.lara_transition_target_type",
            ),
            "proprio_state",
        )
    )
    uses_latent_transition_targets = _artifact_bool(
        payload,
        (
            "uses_latent_transition_targets",
            "features.uses_latent_transition_targets",
        ),
    ) or transition_target_type == "latent_state"

    required_metric_keys = (
        "action_loss",
        "transition_state_loss",
        "moe_loss",
        "moe_route_distill_loss_raw",
        "moe_route_distill_loss_weighted",
        "moe_pool_distill_loss_weighted",
        "moe_pool_coverage_loss_weighted",
        "moe_pool_teacher_mass",
        "moe_pool_critical_miss_rate",
        "moe_utility_loss_weighted",
    )
    missing_metrics = [
        metric_key for metric_key in required_metric_keys if _finite_float(final_metrics.get(metric_key)) is None
    ]
    checkpoint_path = _existing_path(
        _artifact_value(payload, ("checkpoint_path", "best_checkpoint_path", "final_checkpoint_path")),
        relative_to=artifact_dir,
    )

    failures = []
    if status not in {"ok", "success", "completed", "complete"}:
        failures.append("status must be completed/ok")
    if not _artifact_bool(payload, ("uses_real_so101_data", "dataset.uses_real_so101_data")):
        failures.append("uses_real_so101_data must be true")
    if data_mix != _value(cfg, "datasets.vla_data.data_mix", "so101_single_arm"):
        failures.append("data_mix must match the SO101 config")
    if train_steps < min_steps:
        failures.append(f"train_steps_completed must be >= {min_steps}")
    if missing_flags:
        failures.append(f"required paper-stage flags are missing/false: {missing_flags}")
    if not uses_utility_training:
        failures.append("counterfactual utility labels and positive lara_utility_loss_weight are required")
    if not uses_latent_transition_targets:
        failures.append(
            "transition targets must be encoded latent_state targets, not raw proprio_state boundary targets"
        )
    if missing_metrics:
        failures.append(f"required finite final metrics are missing: {missing_metrics}")
    if checkpoint_path is None:
        failures.append("checkpoint_path must point to an existing checkpoint")

    return _check(
        name,
        not failures,
        detail=detail,
        path=str(path),
        failures=failures,
        train_steps_completed=train_steps,
        min_training_steps=min_steps,
        data_mix=data_mix,
        checkpoint_path=checkpoint_path,
        missing_required_flags=missing_flags,
        missing_required_metrics=missing_metrics,
        uses_counterfactual_utility_labels=uses_utility_labels,
        lara_utility_loss_weight=utility_loss_weight,
        transition_target_type=transition_target_type,
        uses_latent_transition_targets=uses_latent_transition_targets,
    )


def _robot_eval_artifact_check(
    path: Path | None,
    *,
    required_fractions: list[float] | None,
    min_episodes: int,
    cfg: Any,
) -> dict[str, Any]:
    name = "closed_loop_robot_eval_artifact"
    detail = "Real closed-loop SO101 evaluation evidence is required beyond unit/smoke tests."
    if path is None:
        return _check(name, False, detail=detail, missing=f"Provide --{name.replace('_', '-')}.")
    if not path.exists():
        return _check(name, False, detail=f"{detail} Path does not exist: {path}", path=str(path))

    payload = _load_json_object(path, name)
    status = str(_artifact_value(payload, ("status", "eval_status"), "")).lower()
    robot = str(_artifact_value(payload, ("robot", "robot_type", "benchmark"), ""))
    num_episodes = int(_artifact_value(payload, ("num_episodes", "episodes", "rollout_episodes"), 0) or 0)
    success_rate = _finite_float(_artifact_value(payload, ("success_rate", "success")))
    observed_fractions = [
        float(fraction)
        for fraction in _artifact_value(payload, ("resident_fractions", "route_retention_fractions"), [])
    ]
    required_fractions = required_fractions or []
    missing_fractions = [
        f"{fraction:g}" for fraction in required_fractions if fraction not in observed_fractions
    ]

    expected_action_horizon = int(_value(cfg, "framework.action_model.action_horizon", 60))
    expected_execution_horizon = int(_value(cfg, "framework.action_model.execution_horizon", 10))
    prediction_horizon = int(_artifact_value(payload, ("prediction_horizon", "action_horizon"), 0) or 0)
    execution_horizon = int(_artifact_value(payload, "execution_horizon", 0) or 0)

    failures = []
    if status not in {"ok", "success", "completed", "complete"}:
        failures.append("status must be completed/ok")
    if "so101" not in robot.lower():
        failures.append("robot must identify SO101")
    if not _artifact_bool(payload, ("uses_real_robot", "real_robot")):
        failures.append("uses_real_robot must be true")
    if not _artifact_bool(payload, ("closed_loop", "is_closed_loop")):
        failures.append("closed_loop must be true")
    if num_episodes < min_episodes:
        failures.append(f"num_episodes must be >= {min_episodes}")
    if success_rate is None or not 0.0 <= success_rate <= 1.0:
        failures.append("success_rate must be finite and in [0, 1]")
    if prediction_horizon != expected_action_horizon or execution_horizon != expected_execution_horizon:
        failures.append(
            "horizons must match "
            f"action_horizon={expected_action_horizon}, execution_horizon={expected_execution_horizon}"
        )
    if missing_fractions:
        failures.append(f"missing resident fractions: {missing_fractions}")
    if not _artifact_bool(payload, ("has_route_diagnostics", "route_diagnostics.ok")):
        failures.append("route diagnostics must be present")
    if not _artifact_bool(payload, ("has_matched_compute_metrics", "matched_compute.ok")):
        failures.append("matched-compute metrics must be present")
    if not _artifact_bool(payload, ("has_counterfactual_utility_eval", "counterfactual_utility_eval.ok")):
        failures.append("counterfactual utility evaluation must be present")

    return _check(
        name,
        not failures,
        detail=detail,
        path=str(path),
        failures=failures,
        num_episodes=num_episodes,
        min_robot_eval_episodes=min_episodes,
        success_rate=success_rate,
        robot=robot,
        resident_fractions=[f"{fraction:g}" for fraction in observed_fractions],
        missing_required_fractions=missing_fractions,
    )


def audit_lara_paper_readiness(args: argparse.Namespace) -> dict[str, Any]:
    config_path = _repo_path(args.config)
    if config_path is None:
        raise ValueError("--config is required")
    cfg = _load_config(config_path)

    sidecar_path = _repo_path(args.counterfactual_utility_labels)
    if sidecar_path is None:
        sidecar_path = _repo_path(_value(cfg, "datasets.vla_data.counterfactual_utility_labels_path"))

    required_fractions = parse_resident_fractions(args.required_resident_fractions)
    if required_fractions is None:
        fractions = _value(cfg, "framework.action_model.lara_route_retention_fractions", None)
        required_fractions = [float(fraction) for fraction in fractions] if fractions is not None else None

    checks = []
    checks.extend(_config_checks(cfg, config_path))
    checks.append(_sidecar_check(sidecar_path, cfg))
    checks.append(
        _rollout_protocol_check(
            _repo_path(args.rollout_records),
            required_fractions=required_fractions,
            add_route_diagnostics=not args.no_route_sequence_diagnostics,
        )
    )
    checks.append(
        _training_artifact_check(
            _repo_path(args.full_so101_training_artifact),
            cfg,
            min_steps=args.min_training_steps,
        )
    )
    checks.append(
        _robot_eval_artifact_check(
            _repo_path(args.closed_loop_robot_eval_artifact),
            required_fractions=required_fractions,
            min_episodes=args.min_robot_eval_episodes,
            cfg=cfg,
        )
    )

    failed_required = [check for check in checks if check["severity"] == "required" and not check["ok"]]
    return {
        "ok": not failed_required,
        "config": str(config_path),
        "required_resident_fractions": (
            None if required_fractions is None else [f"{fraction:g}" for fraction in required_fractions]
        ),
        "checks": checks,
        "missing": [check["name"] for check in failed_required],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="scripts/config/lara_so101_ft.yaml")
    parser.add_argument("--counterfactual-utility-labels", help="Validated utility sidecar JSON/JSONL.")
    parser.add_argument("--rollout-records", help="Closed-loop rollout JSON/JSONL for paper protocol auditing.")
    parser.add_argument("--full-so101-training-artifact", help="Path to full SO101 latent/MoE training evidence.")
    parser.add_argument("--closed-loop-robot-eval-artifact", help="Path to real SO101 closed-loop evaluation evidence.")
    parser.add_argument("--min-training-steps", type=int, default=1000)
    parser.add_argument("--min-robot-eval-episodes", type=int, default=1)
    parser.add_argument(
        "--required-resident-fractions",
        help=(
            "Comma-separated resident fractions required in rollout records; defaults to config "
            "lara_route_retention_fractions."
        ),
    )
    parser.add_argument(
        "--no-route-sequence-diagnostics",
        action="store_true",
        help="Do not derive route diagnostics from raw router_probs_sequence fields.",
    )
    parser.add_argument("--output", help="Optional output JSON path. Defaults to stdout.")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Return exit code 0 even when required readiness evidence is missing.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    payload = audit_lara_paper_readiness(args)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0 if payload["ok"] or args.allow_incomplete else 2


if __name__ == "__main__":
    raise SystemExit(main())
