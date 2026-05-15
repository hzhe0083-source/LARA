"""Framework factory for the LARA model assembly."""

import importlib

from Lara.model.tools import FRAMEWORK_REGISTRY

LAZY_FRAMEWORKS = {
    "Lara": "Lara.model.framework.Lara_core:Lara",
}

LEGACY_FRAMEWORK_ALIASES = {
    "VLA_JEPA": "Lara",
    "QwenJEVLA": "Lara",
}


def _load_lazy_framework(framework_id: str):
    module_path, class_name = LAZY_FRAMEWORKS[framework_id].split(":")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def build_framework(cfg):
    """
    Build a framework model from config.
    Args:
        cfg: Config object (OmegaConf / namespace) containing:
             cfg.framework.name: Identifier string, currently "Lara".
    Returns:
        nn.Module: Instantiated framework model.
    """

    if not hasattr(cfg.framework, "name"): 
        cfg.framework.name = cfg.framework.framework_py  # Backward compatibility for legacy config yaml
        
    framework_id = LEGACY_FRAMEWORK_ALIASES.get(cfg.framework.name, cfg.framework.name)
    cfg.framework.name = framework_id
    if framework_id in FRAMEWORK_REGISTRY._registry:
        model_class = FRAMEWORK_REGISTRY[framework_id]
    elif framework_id in LAZY_FRAMEWORKS:
        model_class = _load_lazy_framework(framework_id)
    else:
        available = ", ".join(sorted(set(FRAMEWORK_REGISTRY._registry) | set(LAZY_FRAMEWORKS)))
        raise NotImplementedError(
            f"Framework {cfg.framework.name} is not implemented. Available frameworks: {available}"
        )
    return model_class(cfg)

__all__ = ["build_framework", "FRAMEWORK_REGISTRY", "LAZY_FRAMEWORKS"]
