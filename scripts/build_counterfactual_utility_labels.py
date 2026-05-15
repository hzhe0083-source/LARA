#!/usr/bin/env python3
"""Build utility-calibration sidecar labels from forced-expert rollout records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Lara.evaluation import (  # noqa: E402
    counterfactual_utility_matrix_from_records,
    counterfactual_utility_records_from_rollouts,
)


def load_records(path: str | Path) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("input rollout records file must not be empty")
    if text[0] == "[":
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError("input must be a JSON list of objects or JSONL object records")
    return records


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to forced-expert rollout JSON/JSONL records.")
    parser.add_argument("--output", required=True, help="Output JSONL sidecar path for the LeRobot dataloader.")
    parser.add_argument("--summary-output", help="Optional JSON summary path. Defaults to stdout.")
    parser.add_argument("--num-experts", type=int, required=True)
    parser.add_argument("--cost-weight", type=float, default=0.0)
    parser.add_argument("--require-all-experts", action="store_true")
    parser.add_argument("--min-candidates-per-context", type=int, default=2)
    args = parser.parse_args()

    records = load_records(args.input)
    sidecar_records = counterfactual_utility_records_from_rollouts(
        records,
        num_experts=args.num_experts,
    )
    labels = counterfactual_utility_matrix_from_records(
        sidecar_records,
        num_experts=args.num_experts,
        cost_weight=args.cost_weight,
        require_all_experts=args.require_all_experts,
        min_candidates_per_context=args.min_candidates_per_context,
    )
    write_jsonl(args.output, sidecar_records)
    summary = {
        "input_records": len(records),
        "output_records": len(sidecar_records),
        "num_contexts": labels["num_contexts"],
        "num_candidates": labels["num_candidates"],
        "missing_candidates": labels["missing_candidates"],
        "output": str(args.output),
    }
    payload = json.dumps(summary, indent=2, sort_keys=True)
    if args.summary_output:
        Path(args.summary_output).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
