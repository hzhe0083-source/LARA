#!/usr/bin/env python3
"""Preflight storage and local assets for LARA LIBERO100 experiments.

The server target for LIBERO100 is storage-constrained, so this script is
intentionally conservative: it only inspects paths, estimates new experiment
footprint, and writes reports. It never downloads or deletes data.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/home/ryan/Documents/robot/benchmark_data/raw/libero100")
DEFAULT_OUTPUT_DIR = Path("runs/libero100_preflight")
EXPECTED_PARQUET_FILES = 279
EXPECTED_TASKS = 100
EXPECTED_FPS = 30


def _bytes_from_gb(value: float) -> int:
    return int(value * 1024**3)


def _format_bytes(num_bytes: int | float) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def _path_size(path: Path) -> int:
    path = path.expanduser()
    if not path.exists():
        return 0
    if path.is_file() or path.is_symlink():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for root, dirs, files in os.walk(path):
        # Do not charge symlink targets to this experiment directory.
        dirs[:] = [d for d in dirs if not (Path(root) / d).is_symlink()]
        for name in files:
            item = Path(root) / name
            try:
                if item.is_symlink():
                    continue
                total += item.stat().st_size
            except OSError:
                continue
    return total


def _safe_relative(path: Path) -> str:
    try:
        return str(path.expanduser().resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.expanduser())


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _find_libero_dataset_dir(data_root: Path) -> Path:
    """Accept either the raw libero100 dir or the concrete HF repo dir."""
    candidates = [
        data_root,
        data_root / "kevin_libero100_lerobot",
        data_root / "raw/libero100/kevin_libero100_lerobot",
    ]
    for candidate in candidates:
        if (candidate / "meta/info.json").exists() or (candidate / "data").exists():
            return candidate
    return candidates[1] if data_root.name == "libero100" else data_root


def inspect_libero100(data_root: Path) -> dict[str, Any]:
    dataset_dir = _find_libero_dataset_dir(data_root.expanduser())
    meta_dir = dataset_dir / "meta"
    data_dir = dataset_dir / "data"
    info_path = meta_dir / "info.json"
    stats_path = meta_dir / "stats.json"
    tasks_parquet = meta_dir / "tasks.parquet"
    tasks_jsonl = meta_dir / "tasks.jsonl"
    chunk_parquets = sorted(data_dir.glob("chunk-*/*.parquet")) if data_dir.exists() else []
    all_parquets = sorted(data_dir.glob("**/*.parquet")) if data_dir.exists() else []
    videos = []
    videos_dir = dataset_dir / "videos"
    if videos_dir.exists():
        for suffix in ("*.mp4", "*.avi", "*.mov"):
            videos.extend(videos_dir.glob(f"**/{suffix}"))

    problems: list[str] = []
    info = _load_json(info_path) if info_path.exists() else None
    features = info.get("features", {}) if isinstance(info, dict) else {}
    image_keys = [
        key for key, value in features.items() if isinstance(value, dict) and value.get("dtype") == "image"
    ]
    action_shape = features.get("action", {}).get("shape") if isinstance(features.get("action"), dict) else None
    state_keys = [
        key
        for key, value in features.items()
        if key.startswith("observation") and isinstance(value, dict) and value.get("dtype") == "float32"
    ]

    if not dataset_dir.exists():
        problems.append("dataset directory does not exist")
    if not info_path.exists():
        problems.append("missing meta/info.json")
    if not stats_path.exists():
        problems.append("missing meta/stats.json")
    if not (tasks_parquet.exists() or tasks_jsonl.exists()):
        problems.append("missing meta/tasks.parquet or meta/tasks.jsonl")
    if not chunk_parquets:
        problems.append("missing data/chunk-*/*.parquet")
    elif len(chunk_parquets) < EXPECTED_PARQUET_FILES:
        problems.append(f"expected at least {EXPECTED_PARQUET_FILES} data chunk parquet files, found {len(chunk_parquets)}")
    if info:
        if info.get("fps") != EXPECTED_FPS:
            problems.append(f"expected fps {EXPECTED_FPS}, found {info.get('fps')}")
        if info.get("total_tasks") != EXPECTED_TASKS:
            problems.append(f"expected {EXPECTED_TASKS} tasks, found {info.get('total_tasks')}")
        if "action" not in features:
            problems.append("missing action feature")
        if not image_keys:
            problems.append("missing image features")

    return {
        "path": str(dataset_dir),
        "exists": dataset_dir.exists(),
        "size_bytes": _path_size(dataset_dir),
        "size_human": _format_bytes(_path_size(dataset_dir)),
        "meta_info_exists": info_path.exists(),
        "meta_stats_exists": stats_path.exists(),
        "meta_tasks_exists": tasks_parquet.exists() or tasks_jsonl.exists(),
        "chunk_parquet_files": len(chunk_parquets),
        "parquet_files": len(all_parquets),
        "video_files": len(videos),
        "fps": info.get("fps") if info else None,
        "total_tasks": info.get("total_tasks") if info else None,
        "total_episodes": info.get("total_episodes") if info else None,
        "total_frames": info.get("total_frames") if info else None,
        "image_keys": image_keys,
        "state_keys": state_keys,
        "action_shape": action_shape,
        "ready": not problems,
        "problems": problems,
    }


def inspect_paths(paths: list[Path], label: str) -> list[dict[str, Any]]:
    results = []
    for path in paths:
        resolved = path.expanduser()
        size = _path_size(resolved)
        results.append(
            {
                "label": label,
                "path": str(resolved),
                "exists": resolved.exists(),
                "is_symlink": resolved.is_symlink(),
                "size_bytes": size,
                "size_human": _format_bytes(size),
            }
        )
    return results


def disk_report(output_dir: Path, max_new_disk_gb: float, min_free_disk_gb: float) -> dict[str, Any]:
    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output_dir)
    max_new_bytes = _bytes_from_gb(max_new_disk_gb)
    min_free_bytes = _bytes_from_gb(min_free_disk_gb)
    projected_free = usage.free - max_new_bytes
    ok = projected_free >= min_free_bytes
    return {
        "path": str(output_dir),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "total_human": _format_bytes(usage.total),
        "used_human": _format_bytes(usage.used),
        "free_human": _format_bytes(usage.free),
        "max_new_disk_gb": max_new_disk_gb,
        "min_free_disk_gb": min_free_disk_gb,
        "projected_free_bytes": projected_free,
        "projected_free_human": _format_bytes(projected_free),
        "ok": ok,
        "problem": None if ok else "projected free space is below the configured minimum",
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    data = inspect_libero100(args.data_root)
    model_paths = []
    for value in args.model_path:
        model_paths.append(Path(value))
    if args.pretrained_root is not None:
        model_paths.extend(
            [
                args.pretrained_root / "VLA-JEPA/Pretrain/checkpoints/VLA-JEPA-pretrain.pt",
                args.pretrained_root / "Qwen3-VL-2B-Instruct",
                args.pretrained_root / "vjepa2-vitl-fpc64-256",
            ]
        )
    model_reports = inspect_paths(model_paths, "model_or_pretrained")
    cache_reports = inspect_paths([args.model_cache, args.hf_cache, args.checkpoint_root], "cache_or_checkpoint_root")
    output_dir = args.output_dir.expanduser()
    output_report = disk_report(output_dir, args.max_new_disk_gb, args.min_free_disk_gb)
    output_size = _path_size(output_dir)

    problems: list[str] = []
    if not data["ready"]:
        problems.extend([f"LIBERO100: {problem}" for problem in data["problems"]])
    for item in model_reports:
        if args.require_model_paths and not item["exists"]:
            problems.append(f"missing model path: {item['path']}")
    if not output_report["ok"]:
        problems.append(str(output_report["problem"]))

    return {
        "ok": not problems,
        "repo_root": str(REPO_ROOT),
        "libero100": data,
        "models": model_reports,
        "storage_roots": cache_reports,
        "output": {
            "path": str(output_dir),
            "current_size_bytes": output_size,
            "current_size_human": _format_bytes(output_size),
        },
        "disk": output_report,
        "constraints": {
            "max_new_disk_gb": args.max_new_disk_gb,
            "min_free_disk_gb": args.min_free_disk_gb,
            "max_checkpoints_to_keep": args.max_checkpoints_to_keep,
            "local_files_only": args.local_files_only,
            "no_delete_without_explicit_allow": True,
        },
        "problems": problems,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# LIBERO100 Storage Preflight",
        "",
        f"Status: {'PASS' if report['ok'] else 'FAIL'}",
        f"Repo: `{report['repo_root']}`",
        "",
        "## Dataset",
        f"- Path: `{report['libero100']['path']}`",
        f"- Ready: `{report['libero100']['ready']}`",
        f"- Size: `{report['libero100']['size_human']}`",
        f"- FPS/tasks: `{report['libero100'].get('fps')}` / `{report['libero100'].get('total_tasks')}`",
        f"- Chunk parquet files: `{report['libero100']['chunk_parquet_files']}`",
        f"- Image keys: `{report['libero100']['image_keys']}`",
        "",
        "## Disk",
        f"- Output: `{report['disk']['path']}`",
        f"- Free now: `{report['disk']['free_human']}`",
        f"- Max new disk: `{report['disk']['max_new_disk_gb']} GiB`",
        f"- Projected free: `{report['disk']['projected_free_human']}`",
        f"- Minimum free required: `{report['disk']['min_free_disk_gb']} GiB`",
        "",
        "## Models And Caches",
    ]
    for item in [*report["models"], *report["storage_roots"]]:
        lines.append(f"- `{item['path']}` exists=`{item['exists']}` size=`{item['size_human']}`")
    lines.extend(["", "## Problems"])
    if report["problems"]:
        lines.extend(f"- {problem}" for problem in report["problems"])
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", "--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model_cache", "--model-cache", type=Path, default=Path("models"))
    parser.add_argument("--hf_cache", "--hf-cache", type=Path, default=Path(os.environ.get("HF_HOME", "~/.cache/huggingface")))
    parser.add_argument("--checkpoint_root", "--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--pretrained_root", "--pretrained-root", type=Path, default=None)
    parser.add_argument("--model_path", "--model-path", action="append", default=[], help="Required local model/checkpoint path. Repeatable.")
    parser.add_argument("--require_model_paths", "--require-model-paths", action="store_true")
    parser.add_argument("--output_dir", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_new_disk_gb", "--max-new-disk-gb", type=float, default=25.0)
    parser.add_argument("--min_free_disk_gb", "--min-free-disk-gb", type=float, default=10.0)
    parser.add_argument("--max_checkpoints_to_keep", "--max-checkpoints-to-keep", type=int, default=2)
    parser.add_argument("--local_files_only", "--local-files-only", action="store_true")
    parser.add_argument("--json_output", "--json-output", type=Path, default=None)
    parser.add_argument("--md_output", "--md-output", type=Path, default=None)
    parser.add_argument("--allow_incomplete", "--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    report = build_report(args)
    json_path = args.json_output or args.output_dir / "preflight_report.json"
    md_path = args.md_output or args.output_dir / "preflight_report.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(md_path, report)
    print(json.dumps({"ok": report["ok"], "json": str(json_path), "markdown": str(md_path), "problems": report["problems"]}, indent=2))
    return 0 if report["ok"] or args.allow_incomplete else 2


if __name__ == "__main__":
    raise SystemExit(main())
