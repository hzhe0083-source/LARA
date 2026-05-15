#!/usr/bin/env python3
"""Smoke-check real LARA component availability and optional one-step execution.

Default mode is intentionally lightweight: it validates the config and required
local paths without loading Qwen/V-JEPA. Use --instantiate or --run-step in the
real training environment to exercise the actual framework components.
"""

from __future__ import annotations

import argparse
import json
import os
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
    use_latent_action_head: bool | None = None,
    use_transition_head: bool | None = None,
    transition_loss_weight: float | None = None,
    use_lara_moe: bool | None = None,
    use_direct_action_experts: bool | None = None,
    use_direct_action_output: bool | None = None,
) -> Any:
    if attn_implementation is not None:
        cfg.framework.qwenvl.attn_implementation = attn_implementation
    action_cfg = cfg.framework.action_model
    if use_latent_action_head is not None:
        action_cfg.use_latent_action_head = bool(use_latent_action_head)
    if use_transition_head is not None:
        action_cfg.lara_use_transition_head = bool(use_transition_head)
        configured_weight = float(action_cfg.get("lara_transition_loss_weight", 0.0))
        if use_transition_head and transition_loss_weight is None and configured_weight == 0.0:
            action_cfg.lara_transition_loss_weight = 1.0
    if transition_loss_weight is not None:
        action_cfg.lara_transition_loss_weight = float(transition_loss_weight)
    if use_lara_moe is not None:
        action_cfg.use_lara_moe = bool(use_lara_moe)
    if use_direct_action_experts is not None:
        action_cfg.lara_use_direct_action_experts = bool(use_direct_action_experts)
        if use_direct_action_experts:
            action_cfg.use_lara_moe = True
    if use_direct_action_output is not None:
        action_cfg.lara_use_direct_action_output = bool(use_direct_action_output)
        if use_direct_action_output:
            action_cfg.use_lara_moe = True
            action_cfg.lara_use_direct_action_experts = True
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
        "lara_use_direct_action_experts": bool(action_cfg.get("lara_use_direct_action_experts", False)),
        "lara_use_direct_action_output": bool(action_cfg.get("lara_use_direct_action_output", False)),
        "lara_use_transition_head": bool(action_cfg.get("lara_use_transition_head", False)),
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
                "execution_state_target": np.zeros((1, state_dim), dtype=np.float32),
                "execution_state_target_mask": True,
                "prediction_state_target": np.zeros((1, state_dim), dtype=np.float32),
                "prediction_state_target_mask": True,
                "trajectory_id": idx,
            }
        )
    return examples


def build_real_examples(cfg: Any, batch_size: int = 1, start_index: int = 0) -> list[dict[str, Any]]:
    if batch_size < 1:
        raise ValueError(f"real batch size must be >= 1, got {batch_size}")
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    from Lara.dataloader.lerobot_datasets import get_vla_dataset

    dataset = get_vla_dataset(
        data_cfg=cfg.datasets.vla_data,
        mode="val",
        action_horizon=cfg.framework.action_model.action_horizon,
        video_horizon=cfg.framework.vj2_model.num_frames,
        execution_horizon=cfg.framework.action_model.get("execution_horizon", None),
    )
    if len(dataset) == 0:
        raise RuntimeError("Configured SO101 dataset produced zero examples")
    return [dataset[(start_index + idx) % len(dataset)] for idx in range(batch_size)]


def _json_safe_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def summarize_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    if not examples:
        return {"batch_size": 0, "keys": [], "fields": {}}
    keys = sorted(examples[0].keys())
    fields = {}
    for key in [
        "future_actions",
        "current_state",
        "video",
        "execution_state_target",
        "prediction_state_target",
    ]:
        value = examples[0].get(key)
        if hasattr(value, "shape"):
            fields[key] = {"shape": list(value.shape), "dtype": str(getattr(value, "dtype", None))}
    return {
        "batch_size": len(examples),
        "keys": keys,
        "fields": fields,
        "trajectory_ids": [_json_safe_scalar(example.get("trajectory_id")) for example in examples],
        "base_indices": [_json_safe_scalar(example.get("base_index")) for example in examples],
    }


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


def smoke_optimizer_parameters(model: Any) -> list[torch.nn.Parameter]:
    target_module = getattr(model, "action_head", model)
    params = [param for param in target_module.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("No trainable action-head parameters available for optimizer-step smoke")
    return params


def _parameter_update_probe(params: list[torch.nn.Parameter]) -> tuple[torch.Tensor, int, torch.Tensor] | None:
    for param in params:
        grad = param.grad
        if grad is None or grad.numel() == 0:
            continue
        flat_grad = grad.detach().flatten()
        grad_max = flat_grad.abs().max()
        if float(grad_max.detach().cpu()) == 0.0:
            continue
        index = int(flat_grad.abs().argmax().detach().cpu())
        return param, index, param.detach().flatten()[index].clone()
    return None


def _exception_status(exc: Exception) -> dict[str, str]:
    return {
        "status": "error",
        "error_type": exc.__class__.__name__,
        "message": str(exc),
    }


def run_one_step(
    model,
    cfg: Any,
    examples: list[dict[str, Any]] | None = None,
    *,
    optimizer_step: bool = False,
    optimizer_lr: float = 1e-4,
) -> dict[str, float]:
    model = place_smoke_trainable_components(model)
    model.train()
    if examples is None:
        examples = build_dummy_examples(cfg, batch_size=1)
    model.zero_grad(set_to_none=True)
    output = model(examples)
    scalar_losses = {key: value for key, value in output.items() if torch.is_tensor(value) and value.numel() == 1}
    if not scalar_losses:
        raise RuntimeError("Model forward did not return scalar losses")
    total_loss = sum(scalar_losses.values())
    total_loss.backward()
    result = {key: float(value.detach().cpu()) for key, value in scalar_losses.items()}
    if optimizer_step:
        params = smoke_optimizer_parameters(model)
        grad_params = [param for param in params if param.grad is not None]
        if not grad_params:
            raise RuntimeError("Optimizer-step smoke found no action-head gradients")
        grad_norm_sq = sum(float(param.grad.detach().float().norm().cpu()) ** 2 for param in grad_params)
        update_probe = _parameter_update_probe(grad_params)
        optimizer = torch.optim.SGD(params, lr=optimizer_lr)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        changed_samples = 0
        if update_probe is not None:
            param, index, before = update_probe
            after = param.detach().flatten()[index]
            changed_samples = int(not torch.equal(before.cpu(), after.detach().cpu()))
        result.update(
            {
                "optimizer/stepped": 1.0,
                "optimizer/param_count": float(sum(param.numel() for param in params)),
                "optimizer/grad_param_count": float(len(grad_params)),
                "optimizer/grad_l2_norm": grad_norm_sq**0.5,
                "optimizer/changed_param_samples": float(changed_samples),
            }
        )
    return result


def smoke_lara_real_components(
    config_path: str | Path,
    *,
    require_data: bool = False,
    instantiate: bool = False,
    run_step: bool = False,
    attn_implementation: str | None = None,
    use_latent_action_head: bool | None = None,
    use_transition_head: bool | None = None,
    transition_loss_weight: float | None = None,
    use_lara_moe: bool | None = None,
    use_direct_action_experts: bool | None = None,
    use_direct_action_output: bool | None = None,
    use_real_batch: bool = False,
    real_batch_size: int = 1,
    real_batch_start_index: int = 0,
    optimizer_step: bool = False,
    optimizer_lr: float = 1e-4,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    cfg = apply_smoke_overrides(
        cfg,
        attn_implementation=attn_implementation,
        use_latent_action_head=use_latent_action_head,
        use_transition_head=use_transition_head,
        transition_loss_weight=transition_loss_weight,
        use_lara_moe=use_lara_moe,
        use_direct_action_experts=use_direct_action_experts,
        use_direct_action_output=use_direct_action_output,
    )
    summary = smoke_config_summary(cfg)
    path_status = check_required_paths(required_component_paths(cfg, require_data=require_data or use_real_batch))
    result = {"summary": summary, "paths": path_status}
    if path_status["status"] != "ok":
        return result
    examples = None
    if use_real_batch:
        try:
            examples = build_real_examples(
                cfg,
                batch_size=real_batch_size,
                start_index=real_batch_start_index,
            )
        except Exception as exc:
            result["batch"] = _exception_status(exc)
            return result
        result["batch"] = {"status": "ok", **summarize_examples(examples)}
    if instantiate or run_step:
        try:
            model = instantiate_lara(cfg)
        except Exception as exc:
            result["instantiate"] = _exception_status(exc)
            return result
        result["instantiate"] = {"status": "ok", "class": model.__class__.__name__}
        if run_step:
            try:
                losses = run_one_step(
                    model,
                    cfg,
                    examples=examples,
                    optimizer_step=optimizer_step,
                    optimizer_lr=optimizer_lr,
                )
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
        "--optimizer-step",
        action="store_true",
        help="After the one-step backward smoke, run one lightweight SGD step over action_head parameters.",
    )
    parser.add_argument("--optimizer-lr", type=float, default=1e-4)
    parser.add_argument(
        "--use-real-batch",
        action="store_true",
        help="Load examples from the configured SO101 dataset instead of using synthetic dummy examples.",
    )
    parser.add_argument("--real-batch-size", type=int, default=1)
    parser.add_argument("--real-batch-start-index", type=int, default=0)
    parser.add_argument(
        "--attn-implementation",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Temporarily override framework.qwenvl.attn_implementation for smoke checks.",
    )
    parser.add_argument(
        "--use-latent-action-head",
        action="store_true",
        help="Temporarily enable the default-off Stage-1 latent action head scaffold.",
    )
    parser.add_argument(
        "--use-transition-head",
        action="store_true",
        help=(
            "Temporarily enable the default-off execution/prediction boundary-state "
            "transition head scaffold. If the configured loss weight is zero, smoke "
            "sets it to 1.0 unless --transition-loss-weight is provided."
        ),
    )
    parser.add_argument(
        "--transition-loss-weight",
        type=float,
        help="Temporarily override framework.action_model.lara_transition_loss_weight for smoke checks.",
    )
    parser.add_argument(
        "--use-lara-moe",
        action="store_true",
        help="Temporarily enable the default-off Stage-2 MoE/router scaffold.",
    )
    parser.add_argument(
        "--use-direct-action-experts",
        action="store_true",
        help="Temporarily enable direct action-chunk experts; this also enables --use-lara-moe.",
    )
    parser.add_argument(
        "--use-direct-action-output",
        action="store_true",
        help=(
            "Temporarily train/predict from routed direct action-chunk experts; "
            "this also enables --use-lara-moe and --use-direct-action-experts."
        ),
    )
    args = parser.parse_args()

    result = smoke_lara_real_components(
        args.config,
        require_data=args.require_data,
        instantiate=args.instantiate,
        run_step=args.run_step or args.optimizer_step,
        attn_implementation=args.attn_implementation,
        use_latent_action_head=args.use_latent_action_head or None,
        use_transition_head=args.use_transition_head or None,
        transition_loss_weight=args.transition_loss_weight,
        use_lara_moe=args.use_lara_moe or None,
        use_direct_action_experts=args.use_direct_action_experts or None,
        use_direct_action_output=args.use_direct_action_output or None,
        use_real_batch=args.use_real_batch,
        real_batch_size=args.real_batch_size,
        real_batch_start_index=args.real_batch_start_index,
        optimizer_step=args.optimizer_step,
        optimizer_lr=args.optimizer_lr,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["paths"]["status"] != "ok":
        return 2
    for key in ("batch", "instantiate", "one_step"):
        if result.get(key, {}).get("status") == "error":
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
