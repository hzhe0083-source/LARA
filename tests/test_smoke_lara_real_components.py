import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from scripts.smoke_lara_real_components import (
    build_dummy_examples,
    check_required_paths,
    required_component_paths,
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
                "vj2_model": {"base_encoder": str(vjepa_dir), "num_frames": 8},
                "action_model": {
                    "action_horizon": 3,
                    "execution_horizon": 1,
                    "action_dim": 2,
                    "state_dim": 4,
                    "use_latent_action_head": False,
                    "use_lara_moe": False,
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
            self.assertEqual(examples[0]["video"].shape, (1, 8, 16, 16, 3))


if __name__ == "__main__":
    unittest.main()
