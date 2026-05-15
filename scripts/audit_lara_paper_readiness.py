#!/usr/bin/env python3
"""Audit whether local artifacts justify claiming the LARA paper method is ready."""

from __future__ import annotations

import argparse
import json
import sys
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

    baseline_default_keys = {
        "use_latent_action_head": False,
        "lara_use_transition_head": False,
        "use_lara_moe": False,
        "lara_use_action_loss_utility": False,
        "lara_use_action_loss_utility_components": False,
        "lara_use_state_utility": False,
        "lara_use_state_utility_components": False,
        "lara_use_utility_head": False,
        "lara_use_direct_action_experts": False,
        "lara_use_direct_action_output": False,
    }
    unsafe_defaults = {
        key: _value(action_cfg, key)
        for key, expected in baseline_default_keys.items()
        if _value(action_cfg, key) != expected
    }
    zero_weight_keys = (
        "lara_utility_loss_weight",
        "lara_utility_rank_loss_weight",
        "lara_utility_head_loss_weight",
        "lara_transition_loss_weight",
        "lara_balance_loss_weight",
        "lara_stickiness_loss_weight",
        "lara_diversity_loss_weight",
        "lara_entropy_loss_weight",
        "lara_inference_stickiness_weight",
    )
    nonzero_default_weights = {
        key: float(_value(action_cfg, key, 0.0))
        for key in zero_weight_keys
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
            "baseline_defaults_safe",
            not unsafe_defaults and not nonzero_default_weights,
            detail="Latent/MoE/utility research paths must stay default-off for the baseline config.",
            unsafe_defaults=unsafe_defaults,
            nonzero_default_weights=nonzero_default_weights,
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

    from Lara.evaluation import counterfactual_utility_matrix_from_records

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

    from Lara.evaluation import protocol_evidence_audit

    records = load_records(path)
    audit = protocol_evidence_audit(
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


def _artifact_check(name: str, path: Path | None, detail: str) -> dict[str, Any]:
    if path is None:
        return _check(name, False, detail=detail, missing=f"Provide --{name.replace('_', '-')}.")
    return _check(
        name,
        path.exists(),
        detail=f"{detail} Path: {path}",
        path=str(path),
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
        _artifact_check(
            "full_so101_training_artifact",
            _repo_path(args.full_so101_training_artifact),
            "Full SO101 training evidence for latent/MoE paths is required before paper-complete claims.",
        )
    )
    checks.append(
        _artifact_check(
            "closed_loop_robot_eval_artifact",
            _repo_path(args.closed_loop_robot_eval_artifact),
            "Real closed-loop SO101 evaluation evidence is required beyond unit/smoke tests.",
        )
    )

    failed_required = [check for check in checks if check["severity"] == "required" and not check["ok"]]
    return {
        "ok": not failed_required,
        "config": str(config_path),
        "required_resident_fractions": None if required_fractions is None else [f"{fraction:g}" for fraction in required_fractions],
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
    parser.add_argument(
        "--required-resident-fractions",
        help="Comma-separated resident fractions required in rollout records; defaults to config lara_route_retention_fractions.",
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
