#!/usr/bin/env python3
"""Summarize LARA rollout records into paper protocol rows.

Input can be either a JSON list of records or newline-delimited JSON. Each
record must contain a resident fraction plus a success value. If a record
contains `router_probs_sequence`, `active_mask_sequence`, `pool_mask_sequence`,
or `valid_mask_sequence`, route-sequence diagnostics are computed before
aggregation.

{"resident_fraction": 0.5, "success": 1, "flops": 2.1, "latency_ms": 18.4}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Lara.evaluation import protocol_evidence_audit, protocol_summary_from_records


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to JSON or JSONL rollout records.")
    parser.add_argument("--output", help="Optional output JSON path. Defaults to stdout.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--total-experts", type=int, required=True)
    parser.add_argument("--active-experts", type=int, required=True)
    parser.add_argument("--shared-params", type=int, default=0)
    parser.add_argument("--params-per-expert", type=int, default=0)
    parser.add_argument("--resident-fraction-key", default="resident_fraction")
    parser.add_argument(
        "--no-route-sequence-diagnostics",
        action="store_true",
        help="Do not derive route diagnostics from raw router_probs_sequence fields.",
    )
    parser.add_argument(
        "--require-paper-metrics",
        action="store_true",
        help="Fail unless rollout records contain success, FLOPs, latency, VRAM, and route diagnostics.",
    )
    parser.add_argument(
        "--required-resident-fractions",
        help="Comma-separated resident fractions required by --require-paper-metrics, for example 0.25,0.5,1.0.",
    )
    args = parser.parse_args()
    records = load_records(args.input)

    summary = protocol_summary_from_records(
        records,
        benchmark=args.benchmark,
        method=args.method,
        total_experts=args.total_experts,
        active_experts=args.active_experts,
        shared_params=args.shared_params,
        params_per_expert=args.params_per_expert,
        resident_fraction_key=args.resident_fraction_key,
        add_route_diagnostics=not args.no_route_sequence_diagnostics,
    )
    if args.require_paper_metrics:
        audit = protocol_evidence_audit(
            records,
            resident_fraction_key=args.resident_fraction_key,
            required_fractions=parse_resident_fractions(args.required_resident_fractions),
            add_route_diagnostics=not args.no_route_sequence_diagnostics,
        )
        summary["evidence_audit"] = audit
    payload = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 2 if args.require_paper_metrics and not summary["evidence_audit"]["ok"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
