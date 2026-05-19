#!/usr/bin/env python3
"""Rewrite downloaded cloud run configs for this local checkout."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = REPO_ROOT / "runs" / "libero100_complete20g"

STAGES = {
    "dense_5000step_action30_bs4x2": {
        "pretrained_checkpoint": REPO_ROOT
        / "models"
        / "VLA-JEPA"
        / "Pretrain"
        / "checkpoints"
        / "VLA-JEPA-pretrain.pt",
    },
    "latent_5000step_action30_bs6x2_from_dense": {
        "pretrained_checkpoint": RUN_ROOT
        / "dense_5000step_action30_bs4x2"
        / "final_model"
        / "pytorch_model.pt",
    },
    "experts_residual_hardtop2_scale008_max006_warm800_cost004_norm001_7000step_action30_bs8x2_from_latent": {
        "pretrained_checkpoint": RUN_ROOT
        / "latent_5000step_action30_bs6x2_from_dense"
        / "final_model"
        / "pytorch_model.pt",
    },
    "router_from_experts_scale008_max006_warm800_3000step_action30_bs8x2": {
        "pretrained_checkpoint": RUN_ROOT
        / "experts_residual_hardtop2_scale008_max006_warm800_cost004_norm001_7000step_action30_bs8x2_from_latent"
        / "final_model"
        / "pytorch_model.pt",
    },
}


def _abs(path: Path) -> str:
    return str(path.absolute())


def _set_if_present(cfg, dotted_key: str, value: str) -> bool:
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        if not hasattr(node, part):
            return False
        node = getattr(node, part)
    if not hasattr(node, parts[-1]):
        return False
    setattr(node, parts[-1], value)
    return True


def localize_config(config_path: Path) -> bool:
    if not config_path.exists():
        return False
    cfg = OmegaConf.load(config_path)
    stage = config_path.parent.name

    changed = False
    replacements = {
        "run_root_dir": _abs(RUN_ROOT),
        "output_dir": _abs(config_path.parent),
        "framework.qwenvl.base_vlm": _abs(REPO_ROOT / "models" / "Qwen3-VL-2B-Instruct"),
        "framework.vj2_model.base_encoder": _abs(REPO_ROOT / "models" / "vjepa2-vitl-fpc64-256"),
        "datasets.vla_data.data_root_dir": _abs(REPO_ROOT / "data" / "libero100"),
    }
    pretrained = STAGES.get(stage, {}).get("pretrained_checkpoint")
    if pretrained is not None:
        replacements["trainer.pretrained_checkpoint"] = _abs(pretrained)

    for key, value in replacements.items():
        before = OmegaConf.select(cfg, key)
        if _set_if_present(cfg, key, value) and before != value:
            changed = True

    if changed:
        OmegaConf.save(config=cfg, f=config_path)
    return changed


def main() -> int:
    changed_paths: list[Path] = []
    for stage in STAGES:
        config_path = RUN_ROOT / stage / "config.yaml"
        if localize_config(config_path):
            changed_paths.append(config_path)

    if changed_paths:
        print("localized configs:")
        for path in changed_paths:
            print(path)
    else:
        print("configs already localized or not downloaded yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
