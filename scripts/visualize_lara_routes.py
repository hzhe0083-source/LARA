#!/usr/bin/env python3
"""Create lightweight LARA route/expert visualizations from JSONL or CSV logs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if text[0] == "[":
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"JSON input must be a list: {path}")
        return [record for record in payload if isinstance(record, dict)]
    records = []
    for line in text.splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _step(record: dict[str, Any], fallback: int) -> int:
    for key in ("step", "steps", "global_step", "completed_steps"):
        value = _as_float(record.get(key))
        if value is not None:
            return int(value)
    return fallback


def _extract_series(records: list[dict[str, Any]], keys: list[str]) -> tuple[list[int], list[float], str | None]:
    for key in keys:
        xs, ys = [], []
        for idx, record in enumerate(records):
            value = _as_float(record.get(key))
            if value is not None:
                xs.append(_step(record, idx))
                ys.append(value)
        if ys:
            return xs, ys, key
    return [], [], None


def _flatten_route_sequence(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    if arr.dtype == object:
        try:
            arr = np.asarray(value, dtype=np.float32)
        except Exception:
            return None
    arr = arr.astype(np.float32)
    # Accepted shapes include [T, E], [B, T, E], [B, T, 1, E].
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim == 2:
        return arr
    if arr.ndim >= 3:
        return arr.reshape(-1, arr.shape[-1])
    return None


def _route_matrices(records: list[dict[str, Any]], key: str) -> list[np.ndarray]:
    matrices = []
    for record in records:
        matrix = _flatten_route_sequence(record.get(key))
        if matrix is not None:
            matrices.append(matrix)
    return matrices


def _save_bar(path: Path, values: np.ndarray, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(np.arange(values.shape[0]), values)
    ax.set_title(title)
    ax.set_xlabel("expert")
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(values.shape[0]))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_line(path: Path, xs: list[int], ys: list[float], title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, ys)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_heatmap(path: Path, matrix: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(matrix.T, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("route chunk")
    ax.set_ylabel("expert")
    fig.colorbar(im, ax=ax, label="probability / mask")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def visualize(records: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    skipped: list[str] = []

    router_mats = _route_matrices(records, "router_probs_sequence")
    active_mats = _route_matrices(records, "active_mask_sequence")
    pool_mats = _route_matrices(records, "pool_mask_sequence")

    if router_mats:
        router_all = np.concatenate(router_mats, axis=0)
        expert_usage = router_all.mean(axis=0)
        path = figures_dir / "expert_usage_router_probs.png"
        _save_bar(path, expert_usage, "Expert Usage", "mean router probability")
        generated.append(str(path))

        path = figures_dir / "route_heatmap_episode_000.png"
        _save_heatmap(path, router_mats[0], "Route Heatmap")
        generated.append(str(path))
    elif active_mats:
        active_all = np.concatenate(active_mats, axis=0)
        expert_usage = active_all.mean(axis=0)
        path = figures_dir / "expert_usage_active_mask.png"
        _save_bar(path, expert_usage, "Expert Usage", "active fraction")
        generated.append(str(path))
    else:
        skipped.append("expert usage and route heatmap: no router_probs_sequence or active_mask_sequence")

    if router_mats and active_mats:
        router_mean = np.concatenate(router_mats, axis=0).mean(axis=0)
        active_mean = np.concatenate(active_mats, axis=0).mean(axis=0)
        width = 0.4
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(router_mean.shape[0])
        ax.bar(x - width / 2, router_mean, width=width, label="router")
        ax.bar(x + width / 2, active_mean, width=width, label="active")
        ax.set_title("Posterior/Router Mass Proxy")
        ax.set_xlabel("expert")
        ax.set_ylabel("mean mass")
        ax.legend()
        fig.tight_layout()
        path = figures_dir / "posterior_vs_router_step_latest.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        generated.append(str(path))
    else:
        skipped.append("posterior vs router mass: need router and active route sequences")

    if pool_mats:
        fractions = [float(matrix.mean()) for matrix in pool_mats]
        path = figures_dir / "pool_coverage_curve.png"
        _save_line(path, list(range(len(fractions))), fractions, "Pool Coverage", "resident fraction")
        generated.append(str(path))
    else:
        xs, ys, key = _extract_series(records, ["moe_pool_teacher_mass", "pool_teacher_mass"])
        if ys:
            path = figures_dir / "pool_coverage_curve.png"
            _save_line(path, xs, ys, "Pool Teacher Mass", key or "pool mass")
            generated.append(str(path))
        else:
            skipped.append("pool coverage: no pool_mask_sequence or pool mass metric")

    for output_name, keys, title in [
        ("router_topk_teacher_mass_curve.png", ["moe_active_teacher_mass", "active_teacher_mass"], "Router Top-k Teacher Mass"),
        ("dead_expert_count_curve.png", ["moe_dead_expert_ratio", "dead_expert_ratio"], "Dead Expert Ratio"),
        ("latent_code_perplexity_curve.png", ["latent_action_perplexity", "perplexity"], "Latent Code Perplexity"),
    ]:
        xs, ys, key = _extract_series(records, keys)
        if ys:
            path = figures_dir / output_name
            _save_line(path, xs, ys, title, key or title)
            generated.append(str(path))
        else:
            skipped.append(f"{title}: missing {keys}")

    code_usage = None
    for record in records:
        for key in ("latent_code_usage", "latent_action_code_usage", "code_usage"):
            if key in record:
                try:
                    code_usage = np.asarray(record[key], dtype=np.float32).reshape(-1)
                    break
                except Exception:
                    pass
        if code_usage is not None:
            break
    if code_usage is not None and code_usage.size:
        path = figures_dir / "latent_code_usage_step_latest.png"
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(np.arange(code_usage.size), code_usage)
        ax.set_title("Latent Code Usage")
        ax.set_xlabel("code")
        ax.set_ylabel("usage")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        generated.append(str(path))
    else:
        skipped.append("latent code usage histogram: no code usage vector")

    manifest = {"records": len(records), "generated": generated, "skipped": skipped}
    (output_dir / "visualization_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="JSON, JSONL, or CSV route/training records.")
    parser.add_argument("--output_dir", "--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = _load_records(args.input)
    if not records:
        raise ValueError(f"no records loaded from {args.input}")
    manifest = visualize(records, args.output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
