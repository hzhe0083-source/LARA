import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.build_counterfactual_utility_labels import load_records, main


class BuildCounterfactualUtilityLabelsTest(unittest.TestCase):
    def test_load_records_accepts_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            records_path = Path(tmp) / "forced.jsonl"
            records_path.write_text(
                "\n".join(
                    [
                        json.dumps({"context_id": "c0", "forced_expert_id_sequence": [0], "success": 1}),
                        json.dumps({"context_id": "c0", "forced_expert_id_sequence": [1], "success": 0}),
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(len(load_records(records_path)), 2)

    def test_main_writes_validated_sidecar_jsonl_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "forced.jsonl"
            output_path = Path(tmp) / "utility.jsonl"
            summary_path = Path(tmp) / "summary.json"
            input_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "trajectory_id": 3,
                                "base_index": 5,
                                "forced_expert_id_sequence": [0, 0],
                                "success": 1,
                            }
                        ),
                        json.dumps(
                            {
                                "trajectory_id": 3,
                                "base_index": 5,
                                "forced_expert_id_sequence": [1],
                                "success": 0,
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            with patch(
                "sys.argv",
                [
                    "build_counterfactual_utility_labels.py",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--summary-output",
                    str(summary_path),
                    "--num-experts",
                    "3",
                ],
            ):
                self.assertEqual(main(), 0)

            sidecar = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(len(sidecar), 2)
            self.assertEqual(sidecar[0]["context_id"], "3:5")
            self.assertEqual(sidecar[0]["expert_id"], 0)
            self.assertEqual(sidecar[1]["expert_id"], 1)
            self.assertEqual(summary["num_contexts"], 1)
            self.assertEqual(summary["num_candidates"], 2)


if __name__ == "__main__":
    unittest.main()
