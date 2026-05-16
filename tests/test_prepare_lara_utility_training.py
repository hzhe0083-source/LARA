import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from scripts.prepare_lara_utility_training import main, prepare_utility_training_config


ROOT = Path(__file__).resolve().parents[1]


class PrepareLaraUtilityTrainingTest(unittest.TestCase):
    def test_prepare_writes_sidecar_training_config_and_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rollout_path = tmp_path / "forced_rollouts.jsonl"
            sidecar_path = tmp_path / "utility_sidecar.jsonl"
            config_path = tmp_path / "utility_train.yaml"
            summary_path = tmp_path / "summary.json"
            rollout_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "trajectory_id": 3,
                                "base_index": 5,
                                "forced_expert_id_sequence": [0, 0],
                                "success": 1,
                                "latency_ms": 10.0,
                            }
                        ),
                        json.dumps(
                            {
                                "trajectory_id": 3,
                                "base_index": 5,
                                "forced_expert_id_sequence": [1],
                                "success": 0,
                                "latency_ms": 5.0,
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            summary = prepare_utility_training_config(
                rollout_records_path=rollout_path,
                base_config_path=ROOT / "scripts/config/lara_so101_utility_pool.yaml",
                sidecar_output_path=sidecar_path,
                config_output_path=config_path,
                summary_output_path=summary_path,
                num_experts=8,
                cost_weight=0.01,
                utility_loss_weight=0.75,
                utility_rank_loss_weight=0.2,
                run_id="utility_from_forced_rollouts",
                num_processes=2,
            )

            sidecar = [json.loads(line) for line in sidecar_path.read_text(encoding="utf-8").splitlines()]
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            written_summary = json.loads(summary_path.read_text(encoding="utf-8"))

            self.assertEqual(summary["status"], "prepared")
            self.assertEqual(summary["num_contexts"], 1)
            self.assertEqual(summary["num_candidates"], 2)
            self.assertFalse(summary["paper_ready"])
            self.assertEqual(summary, written_summary)
            self.assertEqual(len(sidecar), 2)
            self.assertEqual(sidecar[0]["context_id"], "3:5")
            self.assertEqual(sidecar[0]["expert_id"], 0)
            self.assertEqual(sidecar[1]["expert_id"], 1)
            self.assertEqual(config["run_id"], "utility_from_forced_rollouts")
            self.assertTrue(config["framework"]["action_model"]["use_latent_action_head"])
            self.assertTrue(config["framework"]["action_model"]["use_lara_moe"])
            self.assertTrue(config["framework"]["action_model"]["lara_use_direct_action_experts"])
            self.assertEqual(config["framework"]["action_model"]["lara_utility_loss_weight"], 0.75)
            self.assertEqual(config["framework"]["action_model"]["lara_utility_rank_loss_weight"], 0.2)
            self.assertEqual(
                config["datasets"]["vla_data"]["counterfactual_utility_labels_path"],
                str(sidecar_path.resolve()),
            )
            self.assertTrue(config["datasets"]["vla_data"]["counterfactual_utility_sample_labeled_only"])
            self.assertIn(str(config_path.resolve()), summary["train_command"])
            self.assertIn("--num_processes", summary["train_command"])
            self.assertIn("2", summary["train_command"])

    def test_prepare_rejects_sparse_counterfactual_contexts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rollout_path = tmp_path / "forced_rollouts.jsonl"
            rollout_path.write_text(
                json.dumps(
                    {
                        "trajectory_id": 3,
                        "base_index": 5,
                        "forced_expert_id_sequence": [0],
                        "success": 1,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "at least 2 candidates per context"):
                prepare_utility_training_config(
                    rollout_records_path=rollout_path,
                    base_config_path=ROOT / "scripts/config/lara_so101_utility_pool.yaml",
                    sidecar_output_path=tmp_path / "utility_sidecar.jsonl",
                    config_output_path=tmp_path / "utility_train.yaml",
                    num_experts=8,
                )

    def test_main_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rollout_path = tmp_path / "forced_rollouts.jsonl"
            sidecar_path = tmp_path / "utility_sidecar.jsonl"
            config_path = tmp_path / "utility_train.yaml"
            summary_path = tmp_path / "summary.json"
            rollout_path.write_text(
                "\n".join(
                    [
                        json.dumps({"context_id": "c0", "forced_expert_id_sequence": [0], "success": 1}),
                        json.dumps({"context_id": "c0", "forced_expert_id_sequence": [1], "success": 0}),
                    ]
                ),
                encoding="utf-8",
            )

            with patch(
                "sys.argv",
                [
                    "prepare_lara_utility_training.py",
                    "--rollout-records",
                    str(rollout_path),
                    "--base-config",
                    str(ROOT / "scripts/config/lara_so101_utility_pool.yaml"),
                    "--sidecar-output",
                    str(sidecar_path),
                    "--config-output",
                    str(config_path),
                    "--summary-output",
                    str(summary_path),
                    "--num-experts",
                    "8",
                ],
            ):
                self.assertEqual(main(), 0)

            self.assertTrue(sidecar_path.exists())
            self.assertTrue(config_path.exists())
            self.assertEqual(json.loads(summary_path.read_text(encoding="utf-8"))["status"], "prepared")


if __name__ == "__main__":
    unittest.main()
