import tempfile
import unittest
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from scripts.smoke_lara_real_components import (
    REPO_ROOT,
    apply_smoke_overrides,
    build_dummy_examples,
    check_required_paths,
    place_smoke_trainable_components,
    required_component_paths,
    smoke_lara_real_components,
    smoke_config_summary,
)


def tiny_smoke_config(tmpdir: Path):
    qwen_dir = tmpdir / "qwen"
    vjepa_dir = tmpdir / "vjepa"
    ckpt = tmpdir / "VLA-JEPA" / "Pretrain" / "checkpoints" / "VLA-JEPA-pretrain.pt"
    data_root = tmpdir / "so101"
    qwen_dir.mkdir(parents=True)
    vjepa_dir.mkdir(parents=True)
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"placeholder")
    data_root.mkdir()
    return OmegaConf.create(
        {
            "framework": {
                "name": "Lara",
                "qwenvl": {"base_vlm": str(qwen_dir)},
                "vj2_model": {"base_encoder": str(vjepa_dir), "num_frames": 8, "num_world_model_views": 2},
                "action_model": {
                    "action_horizon": 3,
                    "execution_horizon": 1,
                    "action_dim": 2,
                    "state_dim": 4,
                    "use_latent_action_head": False,
                    "use_lara_moe": False,
                    "lara_use_transition_head": False,
                    "lara_use_direct_action_experts": False,
                    "lara_use_direct_action_output": False,
                },
            },
            "datasets": {
                "vla_data": {
                    "data_root_dir": str(data_root),
                    "resolution_size": 16,
                    "video_resolution_size": 16,
                }
            },
            "trainer": {"pretrained_checkpoint": str(ckpt), "reload_modules": "qwen,vj2"},
        }
    )


class SmokeLaraRealComponentsTest(unittest.TestCase):
    def test_script_inserts_repo_root_for_direct_execution(self):
        self.assertIn(str(REPO_ROOT), sys.path)

    def test_required_component_paths_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = tiny_smoke_config(Path(tmp))

            paths = required_component_paths(cfg, require_data=True)
            status = check_required_paths(paths)
            summary = smoke_config_summary(cfg)

            self.assertEqual(status["status"], "ok")
            self.assertEqual(summary["framework"], "Lara")
            self.assertTrue(summary["use_lara_moe_default_safe"])
            self.assertEqual(summary["action_horizon"], 3)
            self.assertEqual(summary["num_world_model_views"], 2)
            self.assertFalse(summary["lara_use_direct_action_experts"])
            self.assertFalse(summary["lara_use_direct_action_output"])
            self.assertFalse(summary["lara_use_transition_head"])

    def test_attention_override_updates_smoke_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = tiny_smoke_config(Path(tmp))
            cfg.framework.qwenvl.attn_implementation = "flash_attention_2"

            apply_smoke_overrides(cfg, attn_implementation="sdpa")
            summary = smoke_config_summary(cfg)

            self.assertEqual(summary["attn_implementation"], "sdpa")

    def test_stage_scaffold_overrides_update_config_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = tiny_smoke_config(Path(tmp))

            apply_smoke_overrides(
                cfg,
                use_latent_action_head=True,
                use_lara_moe=True,
                use_direct_action_experts=True,
                use_direct_action_output=True,
            )
            summary = smoke_config_summary(cfg)

            self.assertTrue(cfg.framework.action_model.use_latent_action_head)
            self.assertTrue(cfg.framework.action_model.use_lara_moe)
            self.assertTrue(cfg.framework.action_model.lara_use_direct_action_experts)
            self.assertTrue(cfg.framework.action_model.lara_use_direct_action_output)
            self.assertTrue(summary["use_latent_action_head"])
            self.assertTrue(summary["use_lara_moe"])
            self.assertTrue(summary["lara_use_direct_action_experts"])
            self.assertTrue(summary["lara_use_direct_action_output"])
            self.assertFalse(summary["use_lara_moe_default_safe"])

    def test_direct_action_overrides_imply_moe_prerequisites(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = tiny_smoke_config(Path(tmp))

            apply_smoke_overrides(cfg, use_direct_action_output=True)

            self.assertTrue(cfg.framework.action_model.use_lara_moe)
            self.assertTrue(cfg.framework.action_model.lara_use_direct_action_experts)
            self.assertTrue(cfg.framework.action_model.lara_use_direct_action_output)

    def test_missing_paths_are_reported_without_loading_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = tiny_smoke_config(Path(tmp))
            cfg.framework.qwenvl.base_vlm = str(Path(tmp) / "missing-qwen")

            status = check_required_paths(required_component_paths(cfg))

            self.assertEqual(status["status"], "missing_paths")
            self.assertIn("qwen_base_vlm", status["missing"])

    def test_build_dummy_examples_match_action_and_state_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = tiny_smoke_config(Path(tmp))

            examples = build_dummy_examples(cfg, batch_size=2)

            self.assertEqual(len(examples), 2)
            self.assertEqual(examples[0]["future_actions"].shape, (3, 2))
            self.assertEqual(examples[0]["current_state"].shape, (1, 4))
            self.assertEqual(examples[0]["video"].shape, (2, 8, 16, 16, 3))

    def test_place_smoke_trainable_components_matches_qwen_device(self):
        class FakePolicy:
            def __init__(self):
                self.qwen_vl_interface = torch.nn.Linear(1, 1)
                self.vj2 = torch.nn.Linear(1, 1)
                self.action_head = torch.nn.Linear(1, 1)

        policy = FakePolicy()
        place_smoke_trainable_components(policy)
        expected_device = next(policy.qwen_vl_interface.parameters()).device

        self.assertEqual(next(policy.vj2.parameters()).device, expected_device)
        self.assertEqual(next(policy.action_head.parameters()).device, expected_device)

    def test_instantiate_failure_is_reported_as_structured_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.yaml"
            OmegaConf.save(tiny_smoke_config(Path(tmp)), cfg_path)

            with patch(
                "scripts.smoke_lara_real_components.instantiate_lara",
                side_effect=ModuleNotFoundError("missing_dependency"),
            ) as instantiate:
                result = smoke_lara_real_components(cfg_path, instantiate=True, attn_implementation="eager")

            self.assertEqual(result["paths"]["status"], "ok")
            self.assertEqual(result["instantiate"]["status"], "error")
            self.assertEqual(result["instantiate"]["error_type"], "ModuleNotFoundError")
            self.assertIn("missing_dependency", result["instantiate"]["message"])
            self.assertEqual(instantiate.call_args.args[0].framework.qwenvl.attn_implementation, "eager")

    def test_smoke_entrypoint_passes_stage_scaffold_overrides_to_instantiate(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.yaml"
            OmegaConf.save(tiny_smoke_config(Path(tmp)), cfg_path)

            with patch(
                "scripts.smoke_lara_real_components.instantiate_lara",
                side_effect=ModuleNotFoundError("missing_dependency"),
            ) as instantiate:
                result = smoke_lara_real_components(
                    cfg_path,
                    instantiate=True,
                    use_latent_action_head=True,
                    use_direct_action_output=True,
                )

            action_cfg = instantiate.call_args.args[0].framework.action_model
            self.assertEqual(result["instantiate"]["status"], "error")
            self.assertTrue(action_cfg.use_latent_action_head)
            self.assertTrue(action_cfg.use_lara_moe)
            self.assertTrue(action_cfg.lara_use_direct_action_experts)
            self.assertTrue(action_cfg.lara_use_direct_action_output)


if __name__ == "__main__":
    unittest.main()
