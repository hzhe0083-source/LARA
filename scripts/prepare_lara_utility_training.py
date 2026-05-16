#!/usr/bin/env python3
"""Prepare utility-router training inputs from real forced-expert rollouts.

This script does not run robot evaluation or invent utility labels. It converts
already-collected forced-expert rollout records into the sidecar format consumed
by the SO101 dataloader, then writes a training config that points at that
sidecar and enables the utility-router loss.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Lara.evaluation import (  # noqa: E402
    counterfactual_utility_matrix_from_records,
    counterfactual_utility_records_from_rollouts,
)
from scripts.build_counterfactual_utility_labels import load_records, write_jsonl  # noqa: E402


def _nested_get(mapping: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _nested_set(mapping: dict[str, Any], keys: tuple[str, ...], value: Any) -> None:
    current = mapping
    for key in keys[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            child = {}
            current[key] = child
        current = child
    current[keys[-1]] = value


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a YAML mapping: {path}")
    return payload


def write_yaml_config(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def prepare_utility_training_config(
    *,
    rollout_records_path: str | Path,
    base_config_path: str | Path,
    sidecar_output_path: str | Path,
    config_output_path: str | Path,
    summary_output_path: str | Path | None = None,
    num_experts: int | None = None,
    cost_weight: float | None = None,
    require_all_experts: bool = False,
    min_candidates_per_context: int | None = None,
    utility_loss_weight: float = 1.0,
    utility_rank_loss_weight: float | None = None,
    sample_labeled_only: bool = True,
    run_id: str | None = None,
    deepspeed_config: str = "./Lara/config/deepseeds/deepspeed_zero2.yaml",
    num_processes: int = 8,
) -> dict[str, Any]:
    if utility_loss_weight <= 0:
        raise ValueError("utility_loss_weight must be positive for utility-router training")

    base_cfg = load_yaml_config(base_config_path)
    data_cfg = _nested_get(base_cfg, ("datasets", "vla_data"), {})

    resolved_num_experts = num_experts or _nested_get(base_cfg, ("framework", "action_model", "lara_num_experts"))
    if resolved_num_experts is None:
        raise ValueError("--num-experts is required when the base config has no lara_num_experts")
    resolved_num_experts = int(resolved_num_experts)

    resolved_cost_weight = (
        float(cost_weight)
        if cost_weight is not None
        else float(data_cfg.get("counterfactual_utility_cost_weight", 0.0) or 0.0)
    )
    resolved_min_candidates = (
        int(min_candidates_per_context)
        if min_candidates_per_context is not None
        else int(data_cfg.get("counterfactual_utility_min_candidates_per_context", 2) or 2)
    )

    rollout_records = load_records(rollout_records_path)
    sidecar_records = counterfactual_utility_records_from_rollouts(
        rollout_records,
        num_experts=resolved_num_experts,
    )
    labels = counterfactual_utility_matrix_from_records(
        sidecar_records,
        num_experts=resolved_num_experts,
        cost_weight=resolved_cost_weight,
        require_all_experts=require_all_experts,
        min_candidates_per_context=resolved_min_candidates,
    )

    sidecar_path = Path(sidecar_output_path).expanduser().resolve()
    config_path = Path(config_output_path).expanduser().resolve()
    write_jsonl(sidecar_path, sidecar_records)

    train_cfg = copy.deepcopy(base_cfg)
    _nested_set(train_cfg, ("run_id",), run_id or f"{base_cfg.get('run_id', 'lara')}_utility_sidecar")
    _nested_set(train_cfg, ("framework", "action_model", "use_latent_action_head"), True)
    _nested_set(train_cfg, ("framework", "action_model", "use_lara_moe"), True)
    _nested_set(train_cfg, ("framework", "action_model", "lara_use_direct_action_experts"), True)
    _nested_set(train_cfg, ("framework", "action_model", "lara_use_direct_action_output"), True)
    _nested_set(train_cfg, ("framework", "action_model", "lara_utility_loss_weight"), float(utility_loss_weight))
    if utility_rank_loss_weight is not None:
        _nested_set(
            train_cfg,
            ("framework", "action_model", "lara_utility_rank_loss_weight"),
            float(utility_rank_loss_weight),
        )
    _nested_set(train_cfg, ("datasets", "vla_data", "counterfactual_utility_labels_path"), str(sidecar_path))
    _nested_set(train_cfg, ("datasets", "vla_data", "counterfactual_utility_cost_weight"), resolved_cost_weight)
    _nested_set(
        train_cfg,
        ("datasets", "vla_data", "counterfactual_utility_require_all_experts"),
        bool(require_all_experts),
    )
    _nested_set(
        train_cfg,
        ("datasets", "vla_data", "counterfactual_utility_min_candidates_per_context"),
        resolved_min_candidates,
    )
    _nested_set(
        train_cfg,
        ("datasets", "vla_data", "counterfactual_utility_sample_labeled_only"),
        bool(sample_labeled_only),
    )
    write_yaml_config(config_path, train_cfg)

    train_command = [
        "accelerate",
        "launch",
        "--config_file",
        deepspeed_config,
        "--num_processes",
        str(num_processes),
        "./Lara/training/train_lara.py",
        "--config_yaml",
        str(config_path),
    ]
    summary = {
        "status": "prepared",
        "base_config": str(Path(base_config_path).expanduser().resolve()),
        "config_output": str(config_path),
        "sidecar_output": str(sidecar_path),
        "input_rollout_records": len(rollout_records),
        "output_sidecar_records": len(sidecar_records),
        "num_experts": resolved_num_experts,
        "num_contexts": labels["num_contexts"],
        "num_candidates": labels["num_candidates"],
        "missing_candidates": labels["missing_candidates"],
        "utility_loss_weight": float(utility_loss_weight),
        "utility_rank_loss_weight": _nested_get(
            train_cfg,
            ("framework", "action_model", "lara_utility_rank_loss_weight"),
        ),
        "counterfactual_utility_sample_labeled_only": bool(sample_labeled_only),
        "train_command": train_command,
        "train_command_string": " ".join(train_command),
        "paper_ready": False,
        "paper_ready_note": (
            "This prepares utility-training inputs from provided rollout records; "
            "paper-ready claims still require full SO101 training and real closed-loop robot evaluation artifacts."
        ),
    }
    if summary_output_path:
        summary_path = Path(summary_output_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-records", required=True, help="Forced-expert rollout JSON/JSONL records.")
    parser.add_argument("--base-config", default="scripts/config/lara_so101_utility_pool.yaml")
    parser.add_argument("--sidecar-output", required=True, help="Output counterfactual utility sidecar JSONL.")
    parser.add_argument("--config-output", required=True, help="Output YAML config for utility-router training.")
    parser.add_argument("--summary-output", help="Optional JSON summary path. Defaults to stdout.")
    parser.add_argument("--num-experts", type=int)
    parser.add_argument("--cost-weight", type=float)
    parser.add_argument("--require-all-experts", action="store_true")
    parser.add_argument("--min-candidates-per-context", type=int)
    parser.add_argument("--utility-loss-weight", type=float, default=1.0)
    parser.add_argument("--utility-rank-loss-weight", type=float)
    parser.add_argument(
        "--sample-all-steps",
        action="store_true",
        help="Do not restrict utility training batches to sidecar-labeled steps.",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--deepspeed-config", default="./Lara/config/deepseeds/deepspeed_zero2.yaml")
    parser.add_argument("--num-processes", type=int, default=8)
    args = parser.parse_args()

    summary = prepare_utility_training_config(
        rollout_records_path=args.rollout_records,
        base_config_path=args.base_config,
        sidecar_output_path=args.sidecar_output,
        config_output_path=args.config_output,
        summary_output_path=args.summary_output,
        num_experts=args.num_experts,
        cost_weight=args.cost_weight,
        require_all_experts=args.require_all_experts,
        min_candidates_per_context=args.min_candidates_per_context,
        utility_loss_weight=args.utility_loss_weight,
        utility_rank_loss_weight=args.utility_rank_loss_weight,
        sample_labeled_only=not args.sample_all_steps,
        run_id=args.run_id,
        deepspeed_config=args.deepspeed_config,
        num_processes=args.num_processes,
    )
    if not args.summary_output:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
