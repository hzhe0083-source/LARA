import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.summarize_lara_protocol import load_records, main, parse_resident_fractions


class SummarizeLARAProtocolTest(unittest.TestCase):
    def test_load_records_accepts_json_and_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "records.json"
            jsonl_path = Path(tmp) / "records.jsonl"
            json_path.write_text(json.dumps([{"resident_fraction": 1.0, "success": 1}]), encoding="utf-8")
            jsonl_path.write_text(
                "\n".join(
                    [
                        json.dumps({"resident_fraction": 0.5, "success": 1}),
                        json.dumps({"resident_fraction": 0.5, "success": 0}),
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(load_records(json_path)[0]["success"], 1)
            self.assertEqual(len(load_records(jsonl_path)), 2)

    def test_main_writes_protocol_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "records.jsonl"
            output_path = Path(tmp) / "summary.json"
            input_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "resident_fraction": 0.5,
                                "success": 1,
                                "flops": 4.0,
                                "router_probs_sequence": [[0.9, 0.1], [0.8, 0.2]],
                                "active_mask_sequence": [[True, False], [True, False]],
                                "pool_mask_sequence": [[True, True], [True, True]],
                            }
                        ),
                        json.dumps(
                            {
                                "resident_fraction": 1.0,
                                "success": 1,
                                "flops": 8.0,
                                "router_probs_sequence": [[0.9, 0.1], [0.2, 0.8]],
                                "active_mask_sequence": [[True, False], [False, True]],
                                "pool_mask_sequence": [[True, True], [True, True]],
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            with patch(
                "sys.argv",
                [
                    "summarize_lara_protocol.py",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--benchmark",
                    "SO101",
                    "--method",
                    "LARA",
                    "--total-experts",
                    "8",
                    "--active-experts",
                    "2",
                    "--shared-params",
                    "100",
                    "--params-per-expert",
                    "10",
                ],
            ):
                self.assertEqual(main(), 0)

            summary = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["benchmark"], "SO101")
            self.assertEqual(summary["num_records_by_fraction"], {"0.5": 1, "1": 1})
            self.assertEqual(summary["rows"][0]["active_experts"], 2)
            self.assertAlmostEqual(summary["route_diagnostics_by_fraction"]["route_switch_rate"]["1"], 1.0)

    def test_parse_resident_fractions(self):
        self.assertEqual(parse_resident_fractions("0.25,0.5,1.0"), [0.25, 0.5, 1.0])

    def test_main_can_fail_on_missing_required_paper_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "records.jsonl"
            output_path = Path(tmp) / "summary.json"
            input_path.write_text(
                json.dumps(
                    {
                        "resident_fraction": 1.0,
                        "success": 1,
                        "flops": 8.0,
                        "latency_ms": 30.0,
                        "posterior_router_kl": 0.04,
                        "pool_teacher_mass": 0.95,
                        "pool_critical_miss_rate": 0.0,
                        "router_probs_sequence": [[0.9, 0.1], [0.2, 0.8]],
                        "pool_mask_sequence": [[True, True], [True, True]],
                    }
                ),
                encoding="utf-8",
            )

            with patch(
                "sys.argv",
                [
                    "summarize_lara_protocol.py",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--benchmark",
                    "SO101",
                    "--method",
                    "LARA",
                    "--total-experts",
                    "8",
                    "--active-experts",
                    "2",
                    "--require-paper-metrics",
                    "--required-resident-fractions",
                    "0.5,1.0",
                ],
            ):
                self.assertEqual(main(), 2)

            summary = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertFalse(summary["evidence_audit"]["ok"])
            self.assertEqual(summary["evidence_audit"]["missing_required_fractions"], ["0.5"])
            self.assertIn("vram_mb", summary["evidence_audit"]["missing_metrics_by_fraction"]["1"])
            self.assertNotIn("posterior_router_kl", summary["evidence_audit"]["missing_metrics_by_fraction"]["1"])


if __name__ == "__main__":
    unittest.main()
