import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _make_minimal_libero(root: Path) -> Path:
    dataset = root / "kevin_libero100_lerobot"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "data/chunk-000").mkdir(parents=True)
    (dataset / "meta/info.json").write_text(
        json.dumps(
            {
                "fps": 30,
                "total_tasks": 100,
                "total_episodes": 1,
                "total_frames": 1,
                "features": {
                    "observation.images.image": {"dtype": "image"},
                    "observation.images.wrist_image": {"dtype": "image"},
                    "observation.state": {"dtype": "float32", "shape": [9]},
                    "action": {"dtype": "float32", "shape": [7]},
                },
            }
        ),
        encoding="utf-8",
    )
    (dataset / "meta/stats.json").write_text("{}", encoding="utf-8")
    (dataset / "meta/tasks.jsonl").write_text('{"task_index": 0, "task": "test"}\n', encoding="utf-8")
    for idx in range(279):
        (dataset / "data/chunk-000" / f"episode_{idx:06d}.parquet").write_bytes(b"")
    return root


class Libero100ExperimentScriptsTest(unittest.TestCase):
    def test_preflight_reports_ready_minimal_libero_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_root = _make_minimal_libero(tmp_path / "data")
            output_dir = tmp_path / "out"
            rc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/preflight_libero100_storage.py"),
                    "--data_root",
                    str(data_root),
                    "--output_dir",
                    str(output_dir),
                    "--max_new_disk_gb",
                    "0.01",
                    "--min_free_disk_gb",
                    "0.01",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(rc.returncode, 0, rc.stderr + rc.stdout)
            report = json.loads((output_dir / "preflight_report.json").read_text(encoding="utf-8"))
            self.assertTrue(report["ok"])
            self.assertEqual(report["libero100"]["chunk_parquet_files"], 279)
            self.assertTrue((output_dir / "preflight_report.md").exists())

    def test_runner_dry_run_writes_stage_config_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_root = _make_minimal_libero(tmp_path / "data")
            output_dir = tmp_path / "run"
            rc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_lara_libero100_experiment.py"),
                    "--stage",
                    "latent",
                    "--data_root",
                    str(data_root),
                    "--pretrained_root",
                    str(tmp_path / "models"),
                    "--output_dir",
                    str(output_dir),
                    "--run_id",
                    "unit_latent",
                    "--dry_run",
                    "--skip_preflight",
                    "--no_accelerate",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(rc.returncode, 0, rc.stderr + rc.stdout)
            run_dir = output_dir / "unit_latent"
            config_path = run_dir / "config/unit_latent.yaml"
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            self.assertEqual(cfg["run_root_dir"], str(output_dir))
            self.assertEqual(cfg["datasets"]["vla_data"]["data_root_dir"], str(data_root))
            self.assertEqual(cfg["trainer"]["save_interval"], 10000)
            self.assertTrue(cfg["framework"]["action_model"]["use_latent_action_head"])
            self.assertFalse(cfg["framework"]["action_model"]["use_lara_moe"])
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["stage"], "latent")
            self.assertEqual(manifest["base_output_dir"], str(output_dir))
            self.assertEqual(manifest["output_dir"], str(run_dir))
            self.assertEqual(manifest["trainer_output_dir"], str(run_dir))

    def test_runner_stage_boundaries_disable_router_and_utility_when_expected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_root = _make_minimal_libero(tmp_path / "data")
            output_dir = tmp_path / "runs"
            cases = {
                "experts": {
                    "lara_use_direct_action_output": False,
                    "lara_router_loss_weight": 0.0,
                    "lara_pool_loss_weight": 0.0,
                    "lara_pool_coverage_loss_weight": 0.0,
                    "lara_utility_loss_weight": 0.0,
                },
                "router": {
                    "lara_use_direct_action_output": True,
                    "lara_router_loss_weight": 1.0,
                    "lara_pool_loss_weight": 1.0,
                    "lara_utility_loss_weight": 0.0,
                },
                "joint": {
                    "lara_use_direct_action_output": True,
                    "lara_router_loss_weight": 0.5,
                    "lara_pool_loss_weight": 0.5,
                    "lara_utility_loss_weight": 0.0,
                },
                "utility_proxy": {
                    "lara_use_direct_action_output": True,
                    "lara_router_loss_weight": 0.5,
                    "lara_pool_loss_weight": 0.5,
                    "lara_utility_loss_weight": 0.05,
                    "lara_utility_rank_loss_weight": 0.02,
                    "lara_utility_head_loss_weight": 0.05,
                },
            }
            for stage, expected in cases.items():
                run_id = f"unit_{stage}"
                rc = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts/run_lara_libero100_experiment.py"),
                        "--stage",
                        stage,
                        "--data_root",
                        str(data_root),
                        "--output_dir",
                        str(output_dir),
                        "--run_id",
                        run_id,
                        "--dry_run",
                        "--skip_preflight",
                        "--no_accelerate",
                    ],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(rc.returncode, 0, rc.stderr + rc.stdout)
                cfg = yaml.safe_load((output_dir / run_id / "config" / f"{run_id}.yaml").read_text(encoding="utf-8"))
                action_model = cfg["framework"]["action_model"]
                for key, value in expected.items():
                    self.assertEqual(action_model[key], value, f"{stage} {key}")

    def test_visualize_lara_routes_generates_core_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            records_path = tmp_path / "records.jsonl"
            records_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "step": 1,
                                "router_probs_sequence": [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]],
                                "active_mask_sequence": [[1, 1, 0], [0, 1, 1]],
                                "pool_mask_sequence": [[1, 1, 0], [1, 1, 0]],
                                "moe_active_teacher_mass": 0.8,
                                "moe_dead_expert_ratio": 0.0,
                                "latent_action_perplexity": 12.0,
                                "latent_code_usage": [0.5, 0.25, 0.25],
                            }
                        ),
                        json.dumps(
                            {
                                "step": 2,
                                "router_probs_sequence": [[0.2, 0.7, 0.1]],
                                "active_mask_sequence": [[0, 1, 1]],
                                "pool_mask_sequence": [[0, 1, 1]],
                                "moe_active_teacher_mass": 0.85,
                                "moe_dead_expert_ratio": 0.0,
                                "latent_action_perplexity": 13.0,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            output_dir = tmp_path / "viz"
            rc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/visualize_lara_routes.py"),
                    "--input",
                    str(records_path),
                    "--output_dir",
                    str(output_dir),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(rc.returncode, 0, rc.stderr + rc.stdout)
            manifest = json.loads((output_dir / "visualization_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["generated"])
            self.assertTrue((output_dir / "figures/expert_usage_router_probs.png").exists())
            self.assertTrue((output_dir / "figures/route_heatmap_episode_000.png").exists())


if __name__ == "__main__":
    unittest.main()
