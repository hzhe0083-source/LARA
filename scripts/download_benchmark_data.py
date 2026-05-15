#!/usr/bin/env python3
"""Download and preflight benchmark LeRobot datasets for LARA experiments."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATA_ROOT = Path("/home/ryan/Documents/robot/benchmark_data")


@dataclass(frozen=True)
class BenchmarkDataset:
    name: str
    repo_id: str
    local_dir: Path
    expected_fps: int
    expected_tasks: int
    expected_data_parquet_files: int


DATASETS = {
    "libero100": BenchmarkDataset(
        name="libero100",
        repo_id="kevin-ys-zhang/libero100_lerobot",
        local_dir=DEFAULT_DATA_ROOT / "raw/libero100/kevin_libero100_lerobot",
        expected_fps=30,
        expected_tasks=100,
        expected_data_parquet_files=279,
    ),
    "metaworld": BenchmarkDataset(
        name="metaworld",
        repo_id="lerobot/metaworld_mt50",
        local_dir=DEFAULT_DATA_ROOT / "raw/metaworld/lerobot_metaworld_mt50",
        expected_fps=80,
        expected_tasks=49,
        expected_data_parquet_files=492,
    ),
}

DEFAULT_INCLUDE_PATTERNS = ["meta/**", "data/chunk-*/*.parquet"]


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _format_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def _dataset_names(selection: str) -> list[str]:
    if selection == "all":
        return list(DATASETS)
    return [selection]


def _resolve_dataset(name: str, data_root: Path) -> BenchmarkDataset:
    dataset = DATASETS[name]
    if data_root == DEFAULT_DATA_ROOT:
        return dataset
    return BenchmarkDataset(
        name=dataset.name,
        repo_id=dataset.repo_id,
        local_dir=data_root / dataset.local_dir.relative_to(DEFAULT_DATA_ROOT),
        expected_fps=dataset.expected_fps,
        expected_tasks=dataset.expected_tasks,
        expected_data_parquet_files=dataset.expected_data_parquet_files,
    )


def _ensure_dirs(data_root: Path) -> dict[str, Path]:
    paths = {
        "hf_home": data_root / "hf_home",
        "hf_cache": data_root / "hf_cache",
        "xet_cache": data_root / "xet_cache",
        "logs": data_root / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _download_env(data_root: Path, disable_xet: bool) -> dict[str, str]:
    paths = _ensure_dirs(data_root)
    env = os.environ.copy()
    env.update(
        {
            "HF_HOME": str(paths["hf_home"]),
            "HF_HUB_CACHE": str(paths["hf_cache"]),
            "HF_XET_CACHE": str(paths["xet_cache"]),
        }
    )
    if disable_xet:
        env["HF_HUB_DISABLE_XET"] = "1"
    return env


def build_hf_command(
    dataset: BenchmarkDataset,
    *,
    dry_run: bool,
    max_workers: int,
    force_download: bool,
    include: list[str],
    exclude: list[str],
) -> list[str]:
    cmd = [
        "hf",
        "download",
        dataset.repo_id,
        "--repo-type",
        "dataset",
        "--local-dir",
        str(dataset.local_dir),
        "--max-workers",
        str(max_workers),
    ]
    if dry_run:
        cmd.append("--dry-run")
    if force_download:
        cmd.append("--force-download")
    include_patterns = include if include else DEFAULT_INCLUDE_PATTERNS
    for pattern in include_patterns:
        cmd.extend(["--include", pattern])
    for pattern in exclude:
        cmd.extend(["--exclude", pattern])
    return cmd


def run_logged(cmd: list[str], env: dict[str, str], log_path: Path) -> int:
    print(f"command: {' '.join(cmd)}")
    print(f"log: {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(cmd)}\n")
        log.write(f"HF_HOME={env.get('HF_HOME', '')}\n")
        log.write(f"HF_HUB_CACHE={env.get('HF_HUB_CACHE', '')}\n")
        log.write(f"HF_XET_CACHE={env.get('HF_XET_CACHE', '')}\n")
        log.write(f"HF_HUB_DISABLE_XET={env.get('HF_HUB_DISABLE_XET', '')}\n\n")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        return proc.wait()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def preflight_dataset(dataset: BenchmarkDataset) -> dict[str, object]:
    info_path = dataset.local_dir / "meta/info.json"
    stats_path = dataset.local_dir / "meta/stats.json"
    data_dir = dataset.local_dir / "data"
    videos_dir = dataset.local_dir / "videos"
    parquet_files = sorted(data_dir.glob("**/*.parquet")) if data_dir.exists() else []
    video_files = (
        sorted(videos_dir.glob("**/*.mp4"))
        + sorted(videos_dir.glob("**/*.avi"))
        + sorted(videos_dir.glob("**/*.mov"))
    ) if videos_dir.exists() else []
    total_local_bytes = sum(p.stat().st_size for p in dataset.local_dir.glob("**/*") if p.is_file())

    report: dict[str, object] = {
        "dataset": dataset.name,
        "repo_id": dataset.repo_id,
        "local_dir": str(dataset.local_dir),
        "meta_info_exists": info_path.exists(),
        "meta_stats_exists": stats_path.exists(),
        "parquet_files": len(parquet_files),
        "video_files": len(video_files),
        "local_size": _format_size(total_local_bytes),
        "ready": False,
        "problems": [],
    }
    problems: list[str] = []
    if not info_path.exists():
        problems.append("missing meta/info.json")
    if not stats_path.exists():
        problems.append("missing meta/stats.json")
    if not parquet_files:
        problems.append("missing data/**/*.parquet")
    elif len(parquet_files) < dataset.expected_data_parquet_files:
        problems.append(
            f"expected {dataset.expected_data_parquet_files} data parquet files, found {len(parquet_files)}"
        )

    if info_path.exists():
        info = _load_json(info_path)
        features = info.get("features", {})
        image_keys = [key for key, value in features.items() if isinstance(value, dict) and value.get("dtype") == "image"]
        report.update(
            {
                "fps": info.get("fps"),
                "total_episodes": info.get("total_episodes"),
                "total_frames": info.get("total_frames"),
                "total_tasks": info.get("total_tasks"),
                "image_keys": image_keys,
                "state_keys": [
                    key
                    for key, value in features.items()
                    if key.startswith("observation") and isinstance(value, dict) and value.get("dtype") == "float32"
                ],
                "action_shape": features.get("action", {}).get("shape") if isinstance(features.get("action"), dict) else None,
            }
        )
        if info.get("fps") != dataset.expected_fps:
            problems.append(f"expected fps {dataset.expected_fps}, found {info.get('fps')}")
        if info.get("total_tasks") != dataset.expected_tasks:
            problems.append(f"expected {dataset.expected_tasks} tasks, found {info.get('total_tasks')}")
        if "action" not in features:
            problems.append("missing action feature")
        if not image_keys:
            problems.append("missing image feature")

    report["ready"] = not problems
    report["problems"] = problems
    return report


def print_preflight(report: dict[str, object]) -> None:
    status = "READY" if report["ready"] else "NOT READY"
    print(f"[{status}] {report['dataset']} at {report['local_dir']}")
    for key in (
        "repo_id",
        "fps",
        "total_episodes",
        "total_frames",
        "total_tasks",
        "image_keys",
        "state_keys",
        "action_shape",
        "parquet_files",
        "video_files",
        "local_size",
    ):
        if key in report:
            print(f"  {key}: {report[key]}")
    for problem in report.get("problems", []):
        print(f"  problem: {problem}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*DATASETS.keys(), "all"], default="all")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--download", action="store_true", help="Perform the download. Defaults to dry-run only.")
    parser.add_argument("--preflight-only", action="store_true", help="Skip hf download and only inspect local files.")
    parser.add_argument("--max-workers", type=int, default=1, help="Use low concurrency by default to avoid HF/Xet cache races.")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--include", action="append", default=[], help="Additional hf include glob. Can be repeated.")
    parser.add_argument("--exclude", action="append", default=[], help="Additional hf exclude glob. Can be repeated.")
    parser.add_argument("--enable-xet", action="store_true", help="Do not set HF_HUB_DISABLE_XET=1.")
    parser.add_argument("--json", action="store_true", help="Print preflight reports as JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    env = _download_env(data_root, disable_xet=not args.enable_xet)
    dry_run = not args.download
    failures = 0
    reports = []

    for dataset_name in _dataset_names(args.dataset):
        dataset = _resolve_dataset(dataset_name, data_root)
        dataset.local_dir.mkdir(parents=True, exist_ok=True)
        if not args.preflight_only:
            cmd = build_hf_command(
                dataset,
                dry_run=dry_run,
                max_workers=args.max_workers,
                force_download=args.force_download,
                include=args.include,
                exclude=args.exclude,
            )
            mode = "dryrun" if dry_run else "download"
            log_path = data_root / "logs" / f"{_timestamp()}-{dataset.name}-{mode}.log"
            rc = run_logged(cmd, env=env, log_path=log_path)
            if rc != 0:
                failures += 1
                print(f"{dataset.name}: hf download exited with {rc}", file=sys.stderr)

        report = preflight_dataset(dataset)
        reports.append(report)
        if not args.json:
            print_preflight(report)

    if args.json:
        print(json.dumps(reports, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
