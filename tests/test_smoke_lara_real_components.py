import tempfile
import unittest
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from scripts.smoke_lara_real_components import (
    REPO_ROOT,
    apply_smoke_overrides,
    build_dummy_examples,
    build_real_examples,
    check_required_paths,
    place_smoke_trainable_components,
    required_component_paths,
    run_one_step,
    smoke_optimizer_parameters,
    smoke_lara_real_components,
    smoke_config_summary,
    summarize_examples,
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
                    "lara_transition_loss_weight": 0.0,
                    "lara_use_direct_action_experts": False,
                    "lara_use_direct_action_output": False,
                    "lara_use_action_loss_utility_components": False,
                    "lara_use_utility_head": False,
                    "lara_utility_head_loss_weight": 0.0,
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
            self.assertEqual(summary["latent_action_horizon"], 1)
            self.assertEqual(summary["router_horizon"], 1)
            self.assertEqual(summary["utility_horizon"], 1)
            self.assertEqual(summary["num_world_model_views"], 2)
            self.assertFalse(summary["lara_use_direct_action_experts"])
            self.assertFalse(summary["lara_use_direct_action_output"])
            self.assertFalse(summary["lara_use_action_loss_utility_components"])
            self.assertFalse(summary["lara_use_utility_head"])
            self.assertFalse(summary["lara_use_transition_head"])
            self.assertFalse(summary["include_episode_start"])

    def test_attention_override_updates_smoke_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = tiny_smoke_config(Path(tmp))
            cfg.framework.qwenvl.attn_implementation = "flash_attention_2"

            apply_smoke_overrides(cfg, attn_implementation="sdpa")
            summary = smoke_config_summary(cfg)

            self.assertEqual(summary["attn_implementation"], "sdpa")

    def test_include_episode_start_override_updates_data_config_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = tiny_smoke_config(Path(tmp))

            apply_smoke_overrides(cfg, include_episode_start=True)
            summary = smoke_config_summary(cfg)

            self.assertTrue(cfg.datasets.vla_data.include_episode_start)
            self.assertTrue(summary["include_episode_start"])

    def test_stage_scaffold_overrides_update_config_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = tiny_smoke_config(Path(tmp))

            apply_smoke_overrides(
                cfg,
                use_latent_action_head=True,
                use_transition_head=True,
                use_lara_moe=True,
                use_direct_action_experts=True,
                use_direct_action_output=True,
            )
            summary = smoke_config_summary(cfg)

            self.assertTrue(cfg.framework.action_model.use_latent_action_head)
            self.assertTrue(cfg.framework.action_model.lara_use_transition_head)
            self.assertEqual(cfg.framework.action_model.lara_transition_loss_weight, 1.0)
            self.assertTrue(cfg.framework.action_model.use_lara_moe)
            self.assertTrue(cfg.framework.action_model.lara_use_direct_action_experts)
            self.assertTrue(cfg.framework.action_model.lara_use_direct_action_output)
            self.assertTrue(summary["use_latent_action_head"])
            self.assertTrue(summary["lara_use_transition_head"])
            self.assertTrue(summary["use_lara_moe"])
            self.assertTrue(summary["lara_use_direct_action_experts"])
            self.assertTrue(summary["lara_use_direct_action_output"])
            self.assertFalse(summary["use_lara_moe_default_safe"])

    def test_transition_loss_weight_override_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = tiny_smoke_config(Path(tmp))

            apply_smoke_overrides(cfg, use_transition_head=True, transition_loss_weight=0.25)

            self.assertTrue(cfg.framework.action_model.lara_use_transition_head)
            self.assertEqual(cfg.framework.action_model.lara_transition_loss_weight, 0.25)

    def test_direct_action_overrides_imply_moe_prerequisites(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = tiny_smoke_config(Path(tmp))

            apply_smoke_overrides(cfg, use_direct_action_output=True)

            self.assertTrue(cfg.framework.action_model.use_lara_moe)
            self.assertTrue(cfg.framework.action_model.lara_use_direct_action_experts)
            self.assertTrue(cfg.framework.action_model.lara_use_direct_action_output)

    def test_action_loss_utility_component_override_implies_prerequisites(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = tiny_smoke_config(Path(tmp))

            apply_smoke_overrides(cfg, use_action_loss_utility_components=True)
            summary = smoke_config_summary(cfg)

            self.assertTrue(cfg.framework.action_model.use_lara_moe)
            self.assertTrue(cfg.framework.action_model.lara_use_direct_action_experts)
            self.assertTrue(cfg.framework.action_model.lara_use_action_loss_utility_components)
            self.assertTrue(cfg.framework.action_model.lara_use_utility_head)
            self.assertEqual(cfg.framework.action_model.lara_utility_head_loss_weight, 1.0)
            self.assertTrue(summary["lara_use_action_loss_utility_components"])
            self.assertTrue(summary["lara_use_utility_head"])

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
            self.assertEqual(examples[0]["future_action_mask"].shape, (3,))
            self.assertTrue(examples[0]["future_action_mask"].all())
            self.assertEqual(examples[0]["current_state"].shape, (1, 4))
            self.assertEqual(examples[0]["execution_state_target"].shape, (1, 4))
            self.assertTrue(examples[0]["execution_state_target_mask"])
            self.assertEqual(examples[0]["prediction_state_target"].shape, (1, 4))
            self.assertTrue(examples[0]["prediction_state_target_mask"])
            self.assertEqual(examples[0]["video"].shape, (2, 8, 16, 16, 3))
            self.assertNotIn("episode_start_image", examples[0])

    def test_build_dummy_examples_can_include_episode_start_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = tiny_smoke_config(Path(tmp))
            cfg.datasets.vla_data.include_episode_start = True

            examples = build_dummy_examples(cfg, batch_size=1)

            self.assertIn("episode_start_image", examples[0])
            self.assertEqual(len(examples[0]["episode_start_image"]), 1)

    def test_build_real_examples_uses_configured_so101_horizons(self):
        class FakeDataset:
            def __len__(self):
                return 4

            def __getitem__(self, index):
                return {
                    "future_actions": np.zeros((3, 2), dtype=np.float32),
                    "current_state": np.zeros((1, 4), dtype=np.float32),
                    "video": np.zeros((2, 8, 16, 16, 3), dtype=np.uint8),
                    "trajectory_id": index,
                    "base_index": 100 + index,
                }

        with tempfile.TemporaryDirectory() as tmp:
            cfg = tiny_smoke_config(Path(tmp))
            get_vla_dataset_calls = []

            def fake_get_vla_dataset(**kwargs):
                get_vla_dataset_calls.append(kwargs)
                return FakeDataset()

            fake_module = types.ModuleType("Lara.dataloader.lerobot_datasets")
            fake_module.get_vla_dataset = fake_get_vla_dataset
            with patch.dict(sys.modules, {"Lara.dataloader.lerobot_datasets": fake_module}):
                examples = build_real_examples(cfg, batch_size=2, start_index=1)

            self.assertEqual(len(examples), 2)
            self.assertEqual(examples[0]["trajectory_id"], 1)
            self.assertEqual(examples[1]["base_index"], 102)
            self.assertEqual(get_vla_dataset_calls[0]["action_horizon"], 3)
            self.assertEqual(get_vla_dataset_calls[0]["video_horizon"], 8)
            self.assertEqual(get_vla_dataset_calls[0]["execution_horizon"], 1)
            self.assertEqual(get_vla_dataset_calls[0]["mode"], "val")

    def test_summarize_examples_reports_real_batch_shapes(self):
        examples = [
            {
                "future_actions": np.zeros((3, 2), dtype=np.float32),
                "future_action_mask": np.array([True, True, False], dtype=bool),
                "current_state": np.zeros((1, 4), dtype=np.float16),
                "video": np.zeros((2, 8, 16, 16, 3), dtype=np.uint8),
                "episode_start_image": ["start"],
                "trajectory_id": np.int64(7),
                "base_index": np.int64(9),
            }
        ]

        summary = summarize_examples(examples)

        self.assertEqual(summary["batch_size"], 1)
        self.assertEqual(summary["fields"]["future_actions"]["shape"], [3, 2])
        self.assertEqual(summary["fields"]["future_action_mask"]["shape"], [3])
        self.assertEqual(summary["fields"]["video"]["dtype"], "uint8")
        self.assertTrue(summary["has_episode_start_image"])
        self.assertEqual(summary["trajectory_ids"], [7])
        self.assertEqual(summary["base_indices"], [9])

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

    def test_optimizer_step_smoke_updates_action_head_parameter_sample(self):
        class FakePolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.qwen_vl_interface = torch.nn.Linear(1, 1)
                self.vj2 = torch.nn.Linear(1, 1)
                self.action_head = torch.nn.Linear(2, 1)

            def forward(self, examples):
                return {"action_loss": self.action_head.weight.pow(2).sum()}

        model = FakePolicy()
        params = smoke_optimizer_parameters(model)

        self.assertEqual([id(param) for param in params], [id(param) for param in model.action_head.parameters()])

        losses = run_one_step(
            model,
            cfg=None,
            examples=[{}],
            optimizer_step=True,
            optimizer_lr=0.1,
        )

        self.assertEqual(losses["optimizer/stepped"], 1.0)
        self.assertGreater(losses["optimizer/grad_param_count"], 0.0)
        self.assertEqual(losses["optimizer/changed_param_samples"], 1.0)

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
                    use_transition_head=True,
                    transition_loss_weight=0.5,
                    use_direct_action_output=True,
                    include_episode_start=True,
                )

            action_cfg = instantiate.call_args.args[0].framework.action_model
            data_cfg = instantiate.call_args.args[0].datasets.vla_data
            self.assertEqual(result["instantiate"]["status"], "error")
            self.assertTrue(action_cfg.use_latent_action_head)
            self.assertTrue(action_cfg.lara_use_transition_head)
            self.assertEqual(action_cfg.lara_transition_loss_weight, 0.5)
            self.assertTrue(action_cfg.use_lara_moe)
            self.assertTrue(action_cfg.lara_use_direct_action_experts)
            self.assertTrue(action_cfg.lara_use_direct_action_output)
            self.assertTrue(data_cfg.include_episode_start)

    def test_smoke_entrypoint_can_report_real_batch_without_instantiating_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.yaml"
            OmegaConf.save(tiny_smoke_config(Path(tmp)), cfg_path)
            examples = [
                {
                    "future_actions": np.zeros((3, 2), dtype=np.float32),
                    "current_state": np.zeros((1, 4), dtype=np.float16),
                    "video": np.zeros((2, 8, 16, 16, 3), dtype=np.uint8),
                    "trajectory_id": 3,
                    "base_index": 11,
                }
            ]

            with patch("scripts.smoke_lara_real_components.build_real_examples", return_value=examples) as builder:
                result = smoke_lara_real_components(cfg_path, use_real_batch=True, real_batch_size=1)

            self.assertEqual(result["paths"]["status"], "ok")
            self.assertEqual(result["batch"]["status"], "ok")
            self.assertEqual(result["batch"]["trajectory_ids"], [3])
            builder.assert_called_once()


if __name__ == "__main__":
    unittest.main()
