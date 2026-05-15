#!/usr/bin/env python3
"""Smoke-check real LARA component availability and optional one-step execution.

Default mode is intentionally lightweight: it validates the config and required
local paths without loading Qwen/V-JEPA. Use --instantiate or --run-step in the
real training environment to exercise the actual framework components.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _resolve_repo_path(path_value: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def load_config(config_path: str | Path) -> Any:
    return OmegaConf.load(_resolve_repo_path(config_path))


def apply_smoke_overrides(
    cfg: Any,
    *,
    attn_implementation: str | None = None,
) -> Any:
    if attn_implementation is not None:
        cfg.framework.qwenvl.attn_implementation = attn_implementation
    return cfg


def required_component_paths(cfg: Any, repo_root: Path = REPO_ROOT, require_data: bool = False) -> dict[str, Path]:
    paths = {
        "qwen_base_vlm": _resolve_repo_path(cfg.framework.qwenvl.base_vlm, repo_root),
        "vjepa_base_encoder": _resolve_repo_path(cfg.framework.vj2_model.base_encoder, repo_root),
        "pretrained_checkpoint": _resolve_repo_path(cfg.trainer.pretrained_checkpoint, repo_root),
    }
    if require_data:
        paths["so101_data_root"] = _resolve_repo_path(cfg.datasets.vla_data.data_root_dir, repo_root)
    return paths


def check_required_paths(paths: dict[str, Path]) -> dict[str, str]:
    missing = {name: str(path) for name, path in paths.items() if not path.exists()}
    if missing:
        return {"status": "missing_paths", "missing": missing}
    return {"status": "ok", "missing": {}}


def smoke_config_summary(cfg: Any) -> dict[str, Any]:
    action_cfg = cfg.framework.action_model
    return {
        "framework": cfg.framework.name,
        "action_horizon": int(action_cfg.action_horizon),
        "execution_horizon": int(action_cfg.execution_horizon),
        "num_world_model_views": int(cfg.framework.vj2_model.get("num_world_model_views", 2)),
        "use_latent_action_head": bool(action_cfg.use_latent_action_head),
        "use_lara_moe": bool(action_cfg.use_lara_moe),
        "use_lara_moe_default_safe": not bool(action_cfg.use_lara_moe),
        "attn_implementation": cfg.framework.qwenvl.get("attn_implementation", None),
        "reload_modules": cfg.trainer.get("reload_modules", None),
    }


def build_dummy_examples(cfg: Any, batch_size: int = 1) -> list[dict[str, Any]]:
    action_cfg = cfg.framework.action_model
    image_size = int(cfg.datasets.vla_data.get("resolution_size", 224))
    video_size = int(cfg.datasets.vla_data.get("video_resolution_size", 256))
    num_frames = int(cfg.framework.vj2_model.num_frames)
    num_world_model_views = int(cfg.framework.vj2_model.get("num_world_model_views", 2))
    action_horizon = int(action_cfg.action_horizon)
    action_dim = int(action_cfg.action_dim)
    state_dim = int(action_cfg.state_dim)
    image = Image.new("RGB", (image_size, image_size), color=0)
    video = np.zeros((num_world_model_views, num_frames, video_size, video_size, 3), dtype=np.uint8)
    examples = []
    for idx in range(batch_size):
        examples.append(
            {
                "image": [image],
                "video": video.copy(),
                "lang": "move the follower arm safely",
                "future_actions": np.zeros((action_horizon, action_dim), dtype=np.float32),
                "current_state": np.zeros((1, state_dim), dtype=np.float32),
                "trajectory_id": idx,
            }
        )
    return examples


def instantiate_lara(cfg: Any):
    from Lara.model.framework import build_framework

    model = build_framework(cfg)
    return model


def _module_device(module: Any) -> torch.device:
    model = getattr(module, "model", module)
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def place_smoke_trainable_components(model: Any) -> Any:
    """Mirror trainer/server device placement for standalone run-step smoke checks."""
    qwen_interface = getattr(model, "qwen_vl_interface", None)
    device = _module_device(qwen_interface) if qwen_interface is not None else _module_device(model)
    for attr in ("vj2", "action_head"):
        component = getattr(model, attr, None)
        if component is not None:
            component.to(device)
    return model


def _exception_status(exc: Exception) -> dict[str, str]:
    return {
        "status": "error",
        "error_type": exc.__class__.__name__,
        "message": str(exc),
    }


def run_one_step(model, cfg: Any) -> dict[str, float]:
    model = place_smoke_trainable_components(model)
    model.train()
    examples = build_dummy_examples(cfg, batch_size=1)
    output = model(examples)
    scalar_losses = {key: value for key, value in output.items() if torch.is_tensor(value) and value.numel() == 1}
    if not scalar_losses:
        raise RuntimeError("Model forward did not return scalar losses")
    total_loss = sum(scalar_losses.values())
    total_loss.backward()
    return {key: float(value.detach().cpu()) for key, value in scalar_losses.items()}


def smoke_lara_real_components(
    config_path: str | Path,
    *,
    require_data: bool = False,
    instantiate: bool = False,
    run_step: bool = False,
    attn_implementation: str | None = None,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    cfg = apply_smoke_overrides(cfg, attn_implementation=attn_implementation)
    summary = smoke_config_summary(cfg)
    path_status = check_required_paths(required_component_paths(cfg, require_data=require_data))
    result = {"summary": summary, "paths": path_status}
    if path_status["status"] != "ok":
        return result
    if instantiate or run_step:
        try:
            model = instantiate_lara(cfg)
        except Exception as exc:
            result["instantiate"] = _exception_status(exc)
            return result
        result["instantiate"] = {"status": "ok", "class": model.__class__.__name__}
        if run_step:
            try:
                losses = run_one_step(model, cfg)
            except Exception as exc:
                result["one_step"] = _exception_status(exc)
                return result
            result["one_step"] = {"status": "ok", "losses": losses}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="scripts/config/lara_so101_ft.yaml")
    parser.add_argument("--require-data", action="store_true")
    parser.add_argument("--instantiate", action="store_true")
    parser.add_argument("--run-step", action="store_true")
    parser.add_argument(
        "--attn-implementation",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Temporarily override framework.qwenvl.attn_implementation for smoke checks.",
    )
    args = parser.parse_args()

    result = smoke_lara_real_components(
        args.config,
        require_data=args.require_data,
        instantiate=args.instantiate,
        run_step=args.run_step,
        attn_implementation=args.attn_implementation,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["paths"]["status"] != "ok":
        return 2
    for key in ("instantiate", "one_step"):
        if result.get(key, {}).get("status") == "error":
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
