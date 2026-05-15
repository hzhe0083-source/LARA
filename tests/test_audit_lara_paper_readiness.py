import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from scripts.audit_lara_paper_readiness import audit_lara_paper_readiness, main


ROOT = Path(__file__).resolve().parents[1]


def base_args(**overrides):
    values = {
        "config": str(ROOT / "scripts/config/lara_so101_ft.yaml"),
        "counterfactual_utility_labels": None,
        "rollout_records": None,
        "full_so101_training_artifact": None,
        "closed_loop_robot_eval_artifact": None,
        "required_resident_fractions": None,
        "no_route_sequence_diagnostics": False,
        "output": None,
        "allow_incomplete": False,
    }
    values.update(overrides)
    return Namespace(**values)


class AuditLaraPaperReadinessTest(unittest.TestCase):
    def test_default_config_is_not_paper_ready_without_artifacts(self):
        report = audit_lara_paper_readiness(base_args())

        self.assertFalse(report["ok"])
        self.assertIn("counterfactual_utility_sidecar", report["missing"])
        self.assertIn("closed_loop_protocol_records", report["missing"])
        self.assertIn("full_so101_training_artifact", report["missing"])
        self.assertIn("closed_loop_robot_eval_artifact", report["missing"])
        self.assertTrue(
            next(check for check in report["checks"] if check["name"] == "baseline_defaults_safe")["ok"]
        )
        self.assertTrue(next(check for check in report["checks"] if check["name"] == "so101_horizon_contract")["ok"])

    def test_valid_sidecar_passes_but_does_not_make_paper_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            sidecar_path = Path(tmp) / "utility.jsonl"
            sidecar_path.write_text(
                "\n".join(
                    [
                        json.dumps({"context_id": "c0", "expert_id": 0, "utility_score": 1.0}),
                        json.dumps({"context_id": "c0", "expert_id": 1, "utility_score": 0.0}),
                    ]
                ),
                encoding="utf-8",
            )

            report = audit_lara_paper_readiness(
                base_args(counterfactual_utility_labels=str(sidecar_path))
            )

        sidecar_check = next(check for check in report["checks"] if check["name"] == "counterfactual_utility_sidecar")
        self.assertTrue(sidecar_check["ok"])
        self.assertFalse(report["ok"])
        self.assertIn("closed_loop_protocol_records", report["missing"])

    def test_rollout_protocol_check_uses_required_fractions(self):
        with tempfile.TemporaryDirectory() as tmp:
            records_path = Path(tmp) / "rollouts.jsonl"
            records_path.write_text(
                json.dumps(
                    {
                        "resident_fraction": 1.0,
                        "success": 1,
                        "flops": 8.0,
                        "latency_ms": 20.0,
                        "vram_mb": 1024.0,
                        "router_probs_sequence": [[0.9, 0.1], [0.8, 0.2]],
                        "pool_mask_sequence": [[True, True], [True, True]],
                    }
                ),
                encoding="utf-8",
            )

            report = audit_lara_paper_readiness(
                base_args(
                    rollout_records=str(records_path),
                    required_resident_fractions="0.5,1.0",
                )
            )

        protocol_check = next(check for check in report["checks"] if check["name"] == "closed_loop_protocol_records")
        self.assertFalse(protocol_check["ok"])
        self.assertEqual(protocol_check["evidence_audit"]["missing_required_fractions"], ["0.5"])

    def test_unsafe_config_default_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "unsafe.yaml"
            config_text = (ROOT / "scripts/config/lara_so101_ft.yaml").read_text(encoding="utf-8")
            config_path.write_text(config_text.replace("use_lara_moe: false", "use_lara_moe: true"), encoding="utf-8")

            report = audit_lara_paper_readiness(base_args(config=str(config_path)))

        default_check = next(check for check in report["checks"] if check["name"] == "baseline_defaults_safe")
        self.assertFalse(default_check["ok"])
        self.assertEqual(default_check["unsafe_defaults"], {"use_lara_moe": True})

    def test_main_writes_output_and_fails_when_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "audit.json"
            with patch(
                "sys.argv",
                [
                    "audit_lara_paper_readiness.py",
                    "--config",
                    str(ROOT / "scripts/config/lara_so101_ft.yaml"),
                    "--output",
                    str(output_path),
                ],
            ):
                self.assertEqual(main(), 2)

            report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertFalse(report["ok"])

    def test_main_can_allow_incomplete_for_preflight_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "audit.json"
            with patch(
                "sys.argv",
                [
                    "audit_lara_paper_readiness.py",
                    "--config",
                    str(ROOT / "scripts/config/lara_so101_ft.yaml"),
                    "--allow-incomplete",
                    "--output",
                    str(output_path),
                ],
            ):
                self.assertEqual(main(), 0)


if __name__ == "__main__":
    unittest.main()
