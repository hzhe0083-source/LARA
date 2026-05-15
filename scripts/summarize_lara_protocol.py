#!/usr/bin/env python3
"""Summarize LARA rollout records into paper protocol rows.

Input can be either a JSON list of records or newline-delimited JSON. Each
record must contain a resident fraction plus a success value, for example:

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

from Lara.evaluation import protocol_summary_from_records


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
    args = parser.parse_args()

    summary = protocol_summary_from_records(
        load_records(args.input),
        benchmark=args.benchmark,
        method=args.method,
        total_experts=args.total_experts,
        active_experts=args.active_experts,
        shared_params=args.shared_params,
        params_per_expert=args.params_per_expert,
        resident_fraction_key=args.resident_fraction_key,
    )
    payload = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
