#!/usr/bin/env python3
"""Run storage-aware staged LARA LIBERO100 experiments.

This is a thin orchestration wrapper around the existing trainer. It writes a
derived config and run manifest, performs storage preflight, launches training
or evaluation stages, and prunes only checkpoints generated inside the selected
output directory.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "scripts/config/lara_libero100_baseline.yaml"
DEFAULT_DEEPSPEED_CONFIG = REPO_ROOT / "Lara/config/deepseeds/deepspeed_zero2.yaml"
STAGES = ("dense", "latent", "experts", "router", "joint", "utility_proxy", "eval")
DEFAULT_SAVE_INTERVAL = 10000
PROVENANCE_HASH_LIMIT_BYTES = 256 * 1024 * 1024
EXPERTS_WARMSTART_MODULES = (
    "qwen,vj2,"
    "action_head.action_model,"
    "action_head.latent_action_head,"
    "action_head.transition_head,"
    "action_head.condition_norm,"
    "action_head.latent_norm"
)
LARA_FLAG_KEYS = (
    "use_latent_action_head",
    "lara_use_transition_head",
    "use_lara_moe",
    "lara_use_direct_action_experts",
    "lara_use_expert_loss_posterior",
    "lara_use_direct_action_output",
    "lara_direct_expert_action_mode",
    "lara_direct_expert_improvement_posterior",
    "lara_direct_expert_hard_assignment",
    "lara_direct_expert_posterior_top_r",
    "lara_direct_expert_improvement_margin",
    "lara_direct_expert_shared_only_gate",
    "lara_direct_expert_residual_scale",
    "lara_direct_expert_residual_max_norm",
    "lara_direct_expert_residual_warmup_steps",
    "lara_direct_expert_residual_cost_weight",
    "lara_direct_residual_norm_loss_weight",
    "lara_direct_residual_diversity_loss_weight",
    "lara_num_experts",
    "lara_episode_pool_size",
    "lara_top_k",
    "lara_router_loss_weight",
    "lara_pool_loss_weight",
    "lara_pool_coverage_loss_weight",
    "lara_utility_loss_weight",
    "lara_utility_rank_loss_weight",
    "lara_utility_head_loss_weight",
)


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None, log_path: Path | None = None) -> int:
    print("$ " + " ".join(cmd), flush=True)
    if log_path is None:
        return subprocess.call(cmd, cwd=cwd, env=env)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as handle:
        handle.write("$ " + " ".join(cmd) + "\n\n")
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            handle.write(line)
        return proc.wait()


def _maybe_git(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _set_nested(mapping: dict[str, Any], keys: tuple[str, ...], value: Any) -> None:
    current = mapping
    for key in keys[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            child = {}
            current[key] = child
        current = child
    current[keys[-1]] = value


def _get_nested(mapping: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _utility_off() -> dict[tuple[str, ...], float]:
    return {
        ("framework", "action_model", "lara_utility_loss_weight"): 0.0,
        ("framework", "action_model", "lara_utility_rank_loss_weight"): 0.0,
        ("framework", "action_model", "lara_utility_head_loss_weight"): 0.0,
    }


def _stage_overrides(stage: str) -> dict[tuple[str, ...], Any]:
    """Keep the staged plan explicit while preserving the current code paths."""
    if stage == "dense":
        return {
            ("framework", "action_model", "use_latent_action_head"): False,
            ("framework", "action_model", "lara_use_transition_head"): False,
            ("framework", "action_model", "use_lara_moe"): False,
            ("framework", "action_model", "lara_use_direct_action_experts"): False,
            ("framework", "action_model", "lara_use_direct_action_output"): False,
            **_utility_off(),
        }
    if stage == "latent":
        return {
            ("framework", "action_model", "use_latent_action_head"): True,
            ("framework", "action_model", "lara_use_transition_head"): True,
            ("framework", "action_model", "use_lara_moe"): False,
            ("framework", "action_model", "lara_use_direct_action_experts"): False,
            ("framework", "action_model", "lara_use_direct_action_output"): False,
            **_utility_off(),
        }
    if stage == "experts":
        return {
            ("framework", "action_model", "use_latent_action_head"): True,
            ("framework", "action_model", "lara_use_transition_head"): True,
            ("framework", "action_model", "use_lara_moe"): True,
            ("framework", "action_model", "lara_use_direct_action_experts"): True,
            ("framework", "action_model", "lara_use_expert_loss_posterior"): True,
            ("framework", "action_model", "lara_use_direct_action_output"): False,
            ("framework", "action_model", "lara_direct_expert_action_mode"): "residual",
            ("framework", "action_model", "lara_direct_expert_improvement_posterior"): True,
            ("framework", "action_model", "lara_direct_expert_hard_assignment"): True,
            ("framework", "action_model", "lara_direct_expert_posterior_top_r"): 2,
            ("framework", "action_model", "lara_direct_expert_improvement_margin"): 0.0,
            ("framework", "action_model", "lara_direct_expert_shared_only_gate"): True,
            ("framework", "action_model", "lara_direct_expert_residual_scale"): 0.08,
            ("framework", "action_model", "lara_direct_expert_residual_max_norm"): 0.06,
            ("framework", "action_model", "lara_direct_expert_residual_warmup_steps"): 800,
            ("framework", "action_model", "lara_direct_expert_residual_cost_weight"): 0.04,
            ("framework", "action_model", "lara_direct_residual_norm_loss_weight"): 0.01,
            ("framework", "action_model", "lara_router_loss_weight"): 0.0,
            ("framework", "action_model", "lara_pool_loss_weight"): 0.0,
            ("framework", "action_model", "lara_pool_coverage_loss_weight"): 0.0,
            (
                "trainer",
                "freeze_modules",
            ): "qwen_vl_interface,action_head.action_model,action_head.latent_action_head,action_head.transition_head,action_head.condition_norm,action_head.latent_norm,vj_predictor",
            **_utility_off(),
        }
    if stage == "router":
        return {
            ("framework", "action_model", "use_latent_action_head"): True,
            ("framework", "action_model", "lara_use_transition_head"): True,
            ("framework", "action_model", "use_lara_moe"): True,
            ("framework", "action_model", "lara_use_direct_action_experts"): True,
            ("framework", "action_model", "lara_use_direct_action_output"): True,
            ("framework", "action_model", "lara_router_loss_weight"): 1.0,
            ("framework", "action_model", "lara_pool_loss_weight"): 1.0,
            ("framework", "action_model", "lara_pool_coverage_loss_weight"): 0.25,
            **_utility_off(),
        }
    if stage == "joint":
        return {
            ("framework", "action_model", "use_latent_action_head"): True,
            ("framework", "action_model", "lara_use_transition_head"): True,
            ("framework", "action_model", "use_lara_moe"): True,
            ("framework", "action_model", "lara_use_direct_action_experts"): True,
            ("framework", "action_model", "lara_use_direct_action_output"): True,
            ("framework", "action_model", "lara_router_loss_weight"): 0.5,
            ("framework", "action_model", "lara_pool_loss_weight"): 0.5,
            ("framework", "action_model", "lara_pool_coverage_loss_weight"): 0.1,
            **_utility_off(),
        }
    if stage == "utility_proxy":
        return {
            ("framework", "action_model", "use_latent_action_head"): True,
            ("framework", "action_model", "lara_use_transition_head"): True,
            ("framework", "action_model", "use_lara_moe"): True,
            ("framework", "action_model", "lara_use_direct_action_experts"): True,
            ("framework", "action_model", "lara_use_direct_action_output"): True,
            ("framework", "action_model", "lara_router_loss_weight"): 0.5,
            ("framework", "action_model", "lara_pool_loss_weight"): 0.5,
            ("framework", "action_model", "lara_pool_coverage_loss_weight"): 0.1,
            ("framework", "action_model", "lara_utility_loss_weight"): 0.05,
            ("framework", "action_model", "lara_utility_rank_loss_weight"): 0.02,
            ("framework", "action_model", "lara_utility_head_loss_weight"): 0.05,
        }
    if stage == "eval":
        return {}
    raise ValueError(f"unknown stage: {stage}")


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file() or path.is_symlink():
        return path.stat().st_size
    total = 0
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if not (Path(root) / d).is_symlink()]
        for name in files:
            item = Path(root) / name
            try:
                if not item.is_symlink():
                    total += item.stat().st_size
            except OSError:
                pass
    return total


def _format_bytes(num_bytes: int | float) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_provenance(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    expanded = path.expanduser()
    record: dict[str, Any] = {
        "path": str(expanded),
        "exists": expanded.exists(),
        "is_file": expanded.is_file(),
        "is_dir": expanded.is_dir(),
        "is_symlink": expanded.is_symlink(),
        "size_bytes": None,
        "sha256": None,
        "sha256_skipped": None,
    }
    if not expanded.exists():
        return record
    try:
        record["resolved_path"] = str(expanded.resolve())
    except OSError:
        record["resolved_path"] = str(expanded)
    if expanded.is_file():
        size = expanded.stat().st_size
        record["size_bytes"] = size
        if size <= PROVENANCE_HASH_LIMIT_BYTES:
            record["sha256"] = _sha256_file(expanded)
        else:
            record["sha256_skipped"] = f"file larger than {_format_bytes(PROVENANCE_HASH_LIMIT_BYTES)}"
    elif expanded.is_dir():
        marker_names = ("config.json", "model.safetensors", "pytorch_model.bin", "preprocessor_config.json")
        record["marker_files"] = [
            {"path": str(expanded / name), "exists": (expanded / name).exists()}
            for name in marker_names
        ]
    return record


def _required_paths(args: argparse.Namespace, cfg: dict[str, Any] | None) -> dict[str, Path | None]:
    if cfg is None:
        return {
            "eval_checkpoint": args.eval_checkpoint,
        }
    qwen_path = _get_nested(cfg, ("framework", "qwenvl", "base_vlm"))
    vjepa_path = _get_nested(cfg, ("framework", "vj2_model", "base_encoder"))
    pretrained_checkpoint = _get_nested(cfg, ("trainer", "pretrained_checkpoint"))
    return {
        "data_root": args.data_root,
        "qwen_path": Path(qwen_path) if qwen_path else None,
        "vjepa_path": Path(vjepa_path) if vjepa_path else None,
        "pretrained_checkpoint": Path(pretrained_checkpoint) if pretrained_checkpoint else None,
        "resume_from": args.resume_from,
        "counterfactual_utility_labels_path": args.counterfactual_utility_labels_path,
    }


def validate_required_paths(args: argparse.Namespace, cfg: dict[str, Any] | None) -> list[str]:
    problems = []
    for label, path in _required_paths(args, cfg).items():
        if path is None:
            continue
        expanded = path.expanduser()
        if label == "counterfactual_utility_labels_path" and not args.counterfactual_utility_labels_path:
            continue
        if label == "resume_from" and not args.resume_from:
            continue
        if not expanded.exists():
            problems.append(f"{label} does not exist: {expanded}")
    return problems


def stage_flag_summary(cfg: dict[str, Any] | None) -> dict[str, Any]:
    action_cfg = _get_nested(cfg or {}, ("framework", "action_model"), {})
    if not isinstance(action_cfg, dict):
        return {}
    return {key: action_cfg.get(key) for key in LARA_FLAG_KEYS if key in action_cfg}


def print_stage_banner(stage: str, cfg: dict[str, Any] | None, run_dir: Path) -> None:
    payload = {
        "stage": stage,
        "run_dir": str(run_dir),
        "lara_flags": stage_flag_summary(cfg),
    }
    print("LARA stage configuration:")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _collect_checkpoint_files(checkpoint_dir: Path) -> list[Path]:
    if not checkpoint_dir.exists():
        return []
    files = [path for path in checkpoint_dir.glob("steps_*_pytorch_model.pt") if path.is_file()]
    return sorted(files, key=lambda path: path.stat().st_mtime)


def prune_checkpoints(output_dir: Path, keep: int, *, allow_delete: bool) -> dict[str, Any]:
    checkpoint_dir = output_dir / "checkpoints"
    files = _collect_checkpoint_files(checkpoint_dir)
    if keep < 0:
        raise ValueError("--max_checkpoints_to_keep must be non-negative")
    doomed = files[:-keep] if keep else files
    deleted = []
    for path in doomed:
        if not allow_delete:
            continue
        path.unlink()
        deleted.append(str(path))
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "found": len(files),
        "keep": keep,
        "delete_enabled": allow_delete,
        "deleted": deleted,
        "would_delete": [str(path) for path in doomed] if not allow_delete else [],
    }


def build_training_config(
    args: argparse.Namespace,
    *,
    base_output_dir: Path,
    run_dir: Path,
    run_id: str,
) -> tuple[dict[str, Any], Path, Path]:
    cfg = _load_yaml(args.config)
    _set_nested(cfg, ("run_id",), run_id)
    _set_nested(cfg, ("run_root_dir",), str(base_output_dir))
    _set_nested(cfg, ("datasets", "vla_data", "data_root_dir"), str(args.data_root))
    if args.pretrained_root is not None:
        _set_nested(cfg, ("framework", "qwenvl", "base_vlm"), str(args.pretrained_root / "Qwen3-VL-2B-Instruct"))
        _set_nested(cfg, ("framework", "vj2_model", "base_encoder"), str(args.pretrained_root / "vjepa2-vitl-fpc64-256"))
        _set_nested(
            cfg,
            ("trainer", "pretrained_checkpoint"),
            str(args.pretrained_root / "VLA-JEPA/Pretrain/checkpoints/VLA-JEPA-pretrain.pt"),
        )
    if args.qwen_path:
        _set_nested(cfg, ("framework", "qwenvl", "base_vlm"), str(args.qwen_path))
    if args.vjepa_path:
        _set_nested(cfg, ("framework", "vj2_model", "base_encoder"), str(args.vjepa_path))
    if args.pretrained_checkpoint:
        _set_nested(cfg, ("trainer", "pretrained_checkpoint"), str(args.pretrained_checkpoint))
    if args.resume_from:
        _set_nested(cfg, ("trainer", "pretrained_checkpoint"), str(args.resume_from))
        _set_nested(cfg, ("trainer", "is_resume"), bool(args.resume_training_state))
        if args.resume_training_state:
            _set_nested(cfg, ("resume_from_checkpoint",), str(args.resume_from))
    if args.attn_implementation:
        _set_nested(cfg, ("framework", "qwenvl", "attn_implementation"), args.attn_implementation)
    if args.per_device_batch_size is not None:
        _set_nested(cfg, ("datasets", "vla_data", "per_device_batch_size"), args.per_device_batch_size)
    if args.gradient_accumulation_steps is not None:
        _set_nested(cfg, ("trainer", "gradient_accumulation_steps"), args.gradient_accumulation_steps)
    if args.max_train_steps is not None:
        _set_nested(cfg, ("trainer", "max_train_steps"), args.max_train_steps)
    _set_nested(
        cfg,
        ("trainer", "save_interval"),
        args.save_interval if args.save_interval is not None else DEFAULT_SAVE_INTERVAL,
    )
    if args.eval_interval is not None:
        _set_nested(cfg, ("trainer", "eval_interval"), args.eval_interval)
    if args.counterfactual_utility_labels_path:
        _set_nested(
            cfg,
            ("datasets", "vla_data", "counterfactual_utility_labels_path"),
            str(args.counterfactual_utility_labels_path),
        )
    for keys, value in _stage_overrides(args.stage).items():
        _set_nested(cfg, keys, value)
    if args.resume_from and not args.resume_training_state:
        if args.stage == "experts":
            _set_nested(cfg, ("trainer", "reload_modules"), EXPERTS_WARMSTART_MODULES)
        elif args.stage in {"router", "joint", "utility_proxy"}:
            _set_nested(cfg, ("trainer", "reload_modules"), "")
    if args.stage == "utility_proxy" and not args.counterfactual_utility_labels_path:
        _set_nested(cfg, ("datasets", "vla_data", "counterfactual_utility_sample_labeled_only"), False)

    config_path = run_dir / "config" / f"{run_id}.yaml"
    _write_yaml(config_path, cfg)
    return cfg, config_path, base_output_dir / run_id


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_preflight_cmd(args: argparse.Namespace, run_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/preflight_libero100_storage.py"),
        "--data_root",
        str(args.data_root),
        "--model_cache",
        str(args.model_cache),
        "--hf_cache",
        str(args.cache_dir),
        "--checkpoint_root",
        str(args.checkpoint_root),
        "--output_dir",
        str(run_dir),
        "--max_new_disk_gb",
        str(args.max_new_disk_gb),
        "--min_free_disk_gb",
        str(args.min_free_disk_gb),
        "--max_checkpoints_to_keep",
        str(args.max_checkpoints_to_keep),
    ]
    if args.local_files_only:
        cmd.append("--local_files_only")
    if args.require_model_paths:
        cmd.append("--require_model_paths")
    if args.pretrained_root is not None:
        cmd.extend(["--pretrained_root", str(args.pretrained_root)])
    for path in args.model_path:
        cmd.extend(["--model_path", str(path)])
    for path in (args.pretrained_checkpoint, args.qwen_path, args.vjepa_path):
        if path:
            cmd.extend(["--model_path", str(path)])
    return cmd


def build_train_cmd(args: argparse.Namespace, config_path: Path) -> list[str]:
    if args.no_accelerate or args.num_gpus <= 1:
        return [
            sys.executable,
            str(REPO_ROOT / "Lara/training/train_lara.py"),
            "--config_yaml",
            str(config_path),
        ]
    cmd = ["accelerate", "launch"]
    if args.deepspeed_config:
        cmd.extend(["--config_file", str(args.deepspeed_config)])
    cmd.extend(
        [
            "--num_processes",
            str(args.num_gpus),
            str(REPO_ROOT / "Lara/training/train_lara.py"),
            "--config_yaml",
            str(config_path),
        ]
    )
    return cmd


def build_eval_cmd(args: argparse.Namespace, run_dir: Path) -> list[str]:
    if not args.eval_checkpoint:
        raise ValueError("--eval_checkpoint is required for --stage eval")
    return [
        sys.executable,
        str(REPO_ROOT / "scripts/eval_libero100_headless.py"),
        "--checkpoint",
        str(args.eval_checkpoint),
        "--output_dir",
        str(run_dir / "eval"),
        "--task_suite_name",
        args.task_suite_name,
        "--num_trials_per_task",
        str(args.num_trials_per_task),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]


def build_env(args: argparse.Namespace, run_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["HF_HOME"] = str(args.cache_dir)
    env["HF_HUB_CACHE"] = str(args.cache_dir / "hub")
    env["TRANSFORMERS_CACHE"] = str(args.cache_dir / "transformers")
    env["LARA_EXPERIMENT_OUTPUT_DIR"] = str(run_dir)
    if args.local_files_only:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    if args.stage == "eval" or args.mujoco_egl:
        env["MUJOCO_GL"] = "egl"
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data_root", "--data-root", type=Path, required=True)
    parser.add_argument("--pretrained_root", "--pretrained-root", type=Path, default=None)
    parser.add_argument("--pretrained_checkpoint", "--pretrained-checkpoint", type=Path)
    parser.add_argument("--qwen_path", "--qwen-path", type=Path)
    parser.add_argument("--vjepa_path", "--vjepa-path", type=Path)
    parser.add_argument("--model_cache", "--model-cache", type=Path, default=Path("models"))
    parser.add_argument("--cache_dir", "--cache-dir", type=Path, default=Path("runs/hf_cache"))
    parser.add_argument("--checkpoint_root", "--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output_dir", "--output-dir", type=Path, required=True)
    parser.add_argument("--run_id", "--run-id")
    parser.add_argument("--resume_from", "--resume-from", type=Path)
    parser.add_argument("--resume_training_state", "--resume-training-state", action="store_true")
    parser.add_argument("--counterfactual_utility_labels_path", "--counterfactual-utility-labels-path", type=Path)
    parser.add_argument("--num_gpus", "--num-gpus", type=int, default=2)
    parser.add_argument("--deepspeed_config", "--deepspeed-config", type=Path, default=DEFAULT_DEEPSPEED_CONFIG)
    parser.add_argument("--no_accelerate", "--no-accelerate", action="store_true")
    parser.add_argument("--attn_implementation", "--attn-implementation", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--per_device_batch_size", "--per-device-batch-size", type=int)
    parser.add_argument("--gradient_accumulation_steps", "--gradient-accumulation-steps", type=int)
    parser.add_argument("--max_train_steps", "--max-train-steps", type=int)
    parser.add_argument("--save_interval", "--save-interval", type=int)
    parser.add_argument("--eval_interval", "--eval-interval", type=int)
    parser.add_argument("--max_checkpoints_to_keep", "--max-checkpoints-to-keep", type=int, default=2)
    parser.add_argument("--allow_delete_generated_artifacts", "--allow-delete-generated-artifacts", action="store_true")
    parser.add_argument("--max_new_disk_gb", "--max-new-disk-gb", type=float, default=25.0)
    parser.add_argument("--min_free_disk_gb", "--min-free-disk-gb", type=float, default=10.0)
    parser.add_argument("--local_files_only", "--local-files-only", action="store_true")
    parser.add_argument("--require_model_paths", "--require-model-paths", action="store_true")
    parser.add_argument("--model_path", "--model-path", type=Path, action="append", default=[])
    parser.add_argument("--skip_preflight", "--skip-preflight", action="store_true")
    parser.add_argument("--dry_run", "--dry-run", action="store_true")
    parser.add_argument("--mujoco_egl", "--mujoco-egl", action="store_true")
    parser.add_argument("--eval_checkpoint", "--eval-checkpoint", type=Path)
    parser.add_argument("--task_suite_name", "--task-suite-name", default="libero_100")
    parser.add_argument("--num_trials_per_task", "--num-trials-per-task", type=int, default=10)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_output_dir = args.output_dir.expanduser().resolve()
    run_id = args.run_id or f"libero100_{args.stage}_{_timestamp()}"
    run_dir = base_output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    env = build_env(args, run_dir)
    cfg = None
    config_path = None
    train_output_dir = None
    if args.stage != "eval":
        cfg, config_path, train_output_dir = build_training_config(
            args,
            base_output_dir=base_output_dir,
            run_dir=run_dir,
            run_id=run_id,
        )
    path_problems = validate_required_paths(args, cfg)
    if path_problems and not args.dry_run:
        for problem in path_problems:
            print(f"path validation error: {problem}", file=sys.stderr)
        return 2
    if args.stage != "eval":
        print_stage_banner(args.stage, cfg, run_dir)

    manifest_path = run_dir / "manifest.json"
    provenance_paths = _required_paths(args, cfg)
    manifest = {
        "created_at": _timestamp(),
        "stage": args.stage,
        "repo_root": str(REPO_ROOT),
        "git_commit": _maybe_git(["rev-parse", "HEAD"]),
        "git_status_short": _maybe_git(["status", "--short"]),
        "input_config": str(args.config),
        "derived_config": str(config_path) if config_path else None,
        "run_id": run_id,
        "base_output_dir": str(base_output_dir),
        "output_dir": str(run_dir),
        "trainer_output_dir": str(train_output_dir) if train_output_dir else None,
        "data_root": str(args.data_root),
        "pretrained_root": str(args.pretrained_root) if args.pretrained_root is not None else None,
        "pretrained_checkpoint": str(args.pretrained_checkpoint) if args.pretrained_checkpoint else None,
        "qwen_path": str(args.qwen_path) if args.qwen_path else None,
        "vjepa_path": str(args.vjepa_path) if args.vjepa_path else None,
        "local_files_only": args.local_files_only,
        "path_validation": {"ok": not path_problems, "problems": path_problems},
        "provenance": {
            label: _path_provenance(path)
            for label, path in provenance_paths.items()
            if path is not None
        },
        "max_checkpoints_to_keep": args.max_checkpoints_to_keep,
        "allow_delete_generated_artifacts": args.allow_delete_generated_artifacts,
        "stage_settings": {
            "action_horizon": _get_nested(cfg or {}, ("framework", "action_model", "action_horizon")),
            "execution_horizon": _get_nested(cfg or {}, ("framework", "action_model", "execution_horizon")),
            "num_experts": _get_nested(cfg or {}, ("framework", "action_model", "lara_num_experts")),
            "episode_pool_size": _get_nested(cfg or {}, ("framework", "action_model", "lara_episode_pool_size")),
            "top_k": _get_nested(cfg or {}, ("framework", "action_model", "lara_top_k")),
            "use_lara_moe": _get_nested(cfg or {}, ("framework", "action_model", "use_lara_moe")),
        },
        "lara_flags": stage_flag_summary(cfg),
    }
    write_manifest(manifest_path, manifest)

    if not args.skip_preflight:
        preflight_cmd = build_preflight_cmd(args, run_dir)
        if args.dry_run:
            print("preflight command:", " ".join(preflight_cmd))
        else:
            rc = _run(preflight_cmd, cwd=REPO_ROOT, env=env, log_path=run_dir / "logs/preflight.log")
            if rc != 0:
                return rc

    if args.dry_run:
        if args.stage == "eval":
            print("eval command:", " ".join(build_eval_cmd(args, run_dir)))
        else:
            print("train command:", " ".join(build_train_cmd(args, config_path)))
        return 0

    if args.stage == "eval":
        rc = _run(build_eval_cmd(args, run_dir), cwd=REPO_ROOT, env=env, log_path=run_dir / "logs/eval.log")
    else:
        rc = _run(build_train_cmd(args, config_path), cwd=REPO_ROOT, env=env, log_path=run_dir / "logs/train.log")
    prune_target_dir = train_output_dir if train_output_dir is not None else run_dir
    prune_report = prune_checkpoints(
        prune_target_dir,
        args.max_checkpoints_to_keep,
        allow_delete=args.allow_delete_generated_artifacts,
    )
    manifest["completed_at"] = _timestamp()
    manifest["return_code"] = rc
    manifest["output_size_bytes"] = _path_size(run_dir)
    manifest["output_size_human"] = _format_bytes(manifest["output_size_bytes"])
    manifest["checkpoint_pruning"] = prune_report
    write_manifest(manifest_path, manifest)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
