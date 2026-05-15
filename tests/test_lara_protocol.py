import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from Lara.evaluation.lara_protocol import (
    counterfactual_utility_matrix_from_records,
    counterfactual_utility_records_from_rollouts,
    matched_budget_flags,
    matched_compute_row,
    matched_expert_budget_flags,
    normalize_protocol_records,
    pareto_frontier_flags,
    protocol_evidence_audit,
    protocol_summary_from_records,
    resident_experts_for_fraction,
    rollout_record_with_route_diagnostics,
    route_sequence_diagnostics,
    step_context_id,
    subset_retention_rows,
    subset_retention_success_curve,
)


class LARAProtocolTest(unittest.TestCase):
    def test_subset_retention_success_curve_reports_auc_and_drop(self):
        curve = subset_retention_success_curve(
            {
                0.25: [0.4, 0.6],
                0.5: [0.7, 0.7],
                1.0: [0.9, 1.0],
            }
        )

        self.assertAlmostEqual(curve["success_at_resident_0.25"], 0.5)
        self.assertAlmostEqual(curve["success_at_resident_0.5"], 0.7)
        self.assertAlmostEqual(curve["success_at_resident_1"], 0.95)
        self.assertGreater(curve["subset_retention_auc"], curve["success_at_resident_0.25"])
        self.assertAlmostEqual(curve["success_drop_1_to_0.25"], 0.45)

    def test_matched_compute_row_uses_sparse_route_budget(self):
        row = matched_compute_row(
            benchmark="SO101",
            method="LARA + route pool",
            total_experts=8,
            active_experts=2,
            resident_experts=4,
            shared_params=100,
            params_per_expert=10,
            success_rate=0.75,
            flops=2.0,
            latency_ms=12.5,
            vram_mb=1024,
        )

        self.assertEqual(row["total_params"], 180.0)
        self.assertEqual(row["active_params"], 120.0)
        self.assertEqual(row["resident_params"], 140.0)
        self.assertEqual(row["total_experts"], 8)
        self.assertEqual(row["active_experts"], 2)
        self.assertEqual(row["resident_experts"], 4)
        self.assertAlmostEqual(row["active_expert_fraction"], 0.25)
        self.assertAlmostEqual(row["resident_expert_fraction"], 0.5)
        self.assertAlmostEqual(row["active_param_fraction"], 120.0 / 180.0)
        self.assertEqual(row["success_rate"], 0.75)

    def test_matched_budget_flags_compare_active_and_resident_params(self):
        reference = matched_compute_row(
            benchmark="SO101",
            method="Dense VLA",
            total_params=1000,
            active_params=500,
            resident_params=1000,
        )
        candidate = matched_compute_row(
            benchmark="SO101",
            method="LARA",
            total_params=1500,
            active_params=520,
            resident_params=960,
        )

        flags = matched_budget_flags(reference, candidate, active_tolerance=0.05, resident_tolerance=0.05)

        self.assertTrue(flags["active_params_matched"])
        self.assertTrue(flags["resident_params_matched"])
        self.assertTrue(flags["matched_compute"])

    def test_matched_expert_budget_flags_compare_active_and_resident_counts(self):
        reference = matched_compute_row(
            benchmark="SO101",
            method="Dense expert budget",
            total_experts=8,
            active_experts=2,
            resident_experts=4,
        )
        candidate = matched_compute_row(
            benchmark="SO101",
            method="LARA",
            total_experts=8,
            active_experts=2,
            resident_experts=4,
        )

        flags = matched_expert_budget_flags(reference, candidate)

        self.assertTrue(flags["active_experts_matched"])
        self.assertTrue(flags["resident_experts_matched"])
        self.assertTrue(flags["matched_expert_budget"])

    def test_subset_retention_rows_build_matched_resident_table(self):
        rows = subset_retention_rows(
            benchmark="SO101",
            method="LARA",
            success_by_fraction={0.25: [1, 0], 0.5: [1, 1]},
            route_regret_by_fraction={0.25: [0.3, 0.1], 0.5: [0.0, 0.2]},
            total_experts=8,
            active_experts=2,
            shared_params=100,
            params_per_expert=10,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["method"], "LARA resident=0.25")
        self.assertEqual(rows[0]["resident_fraction_requested"], 0.25)
        self.assertEqual(rows[0]["resident_experts"], 2)
        self.assertAlmostEqual(rows[0]["success_rate"], 0.5)
        self.assertAlmostEqual(rows[0]["route_regret"], 0.2)
        self.assertEqual(rows[1]["resident_experts"], 4)
        self.assertAlmostEqual(rows[1]["resident_expert_fraction"], 0.5)

    def test_resident_experts_for_fraction_validates_bounds(self):
        self.assertEqual(resident_experts_for_fraction(8, 0.25), 2)
        self.assertEqual(resident_experts_for_fraction(8, 0.26), 3)
        self.assertEqual(resident_experts_for_fraction(8, 1.0), 8)
        with self.assertRaisesRegex(ValueError, "resident fraction"):
            resident_experts_for_fraction(8, 0.0)

    def test_pareto_frontier_flags_reject_dominated_rows(self):
        rows = [
            matched_compute_row("SO101", "A", success_rate=0.8, flops=10.0),
            matched_compute_row("SO101", "B", success_rate=0.7, flops=12.0),
            matched_compute_row("SO101", "C", success_rate=0.75, flops=8.0),
        ]

        flags = pareto_frontier_flags(rows)

        self.assertEqual(flags, [True, False, True])

    def test_route_sequence_diagnostics_reports_closed_loop_route_stability(self):
        router_probs = torch.tensor(
            [
                [
                    [0.9, 0.1, 0.0],
                    [0.2, 0.7, 0.1],
                    [0.1, 0.8, 0.1],
                ]
            ]
        )
        active_mask = torch.tensor(
            [
                [
                    [True, False, False],
                    [False, True, False],
                    [False, True, False],
                ]
            ]
        )
        pool_mask = torch.tensor(
            [
                [
                    [True, True, False],
                    [True, True, False],
                    [False, True, True],
                ]
            ]
        )

        diagnostics = route_sequence_diagnostics(
            router_probs,
            active_mask=active_mask,
            pool_mask=pool_mask,
        )

        self.assertEqual(diagnostics["valid_chunks"], 3.0)
        self.assertEqual(diagnostics["valid_transitions"], 2.0)
        self.assertAlmostEqual(diagnostics["route_switch_rate"], 0.5)
        self.assertAlmostEqual(diagnostics["active_set_switch_rate"], 0.5)
        self.assertAlmostEqual(diagnostics["active_set_jaccard"], 0.5)
        self.assertAlmostEqual(diagnostics["pool_switch_rate"], 0.5)
        self.assertAlmostEqual(diagnostics["pool_reuse_rate"], 0.5)
        self.assertAlmostEqual(diagnostics["pool_jaccard"], (1.0 + 1.0 / 3.0) / 2.0)
        self.assertAlmostEqual(diagnostics["resident_expert_fraction_mean"], 2.0 / 3.0)

    def test_rollout_record_with_route_diagnostics_accepts_single_episode_sequences(self):
        record = {
            "resident_fraction": 0.5,
            "success": 1,
            "router_probs_sequence": [
                [0.9, 0.1, 0.0],
                [0.2, 0.7, 0.1],
            ],
            "active_mask_sequence": [
                [True, False, False],
                [False, True, False],
            ],
            "pool_mask_sequence": [
                [True, True, False],
                [True, True, False],
            ],
        }

        enriched = rollout_record_with_route_diagnostics(record)

        self.assertAlmostEqual(enriched["route_switch_rate"], 1.0)
        self.assertAlmostEqual(enriched["active_set_switch_rate"], 1.0)
        self.assertAlmostEqual(enriched["pool_reuse_rate"], 1.0)
        self.assertAlmostEqual(enriched["pool_jaccard"], 1.0)
        self.assertIn("router_probs_sequence", enriched)

    def test_normalize_protocol_records_can_skip_route_diagnostics(self):
        records = [
            {
                "resident_fraction": 1.0,
                "success": 1,
                "router_probs_sequence": [[0.1, 0.9]],
            }
        ]

        self.assertIn("mean_router_entropy", normalize_protocol_records(records)[0])
        self.assertNotIn(
            "mean_router_entropy",
            normalize_protocol_records(records, add_route_diagnostics=False)[0],
        )

    def test_protocol_summary_from_records_builds_rows_curve_and_pareto_flags(self):
        records = [
            {
                "resident_fraction": 0.25,
                "success": 1,
                "return_score": 0.8,
                "route_regret": 0.2,
                "flops": 4.0,
                "latency_ms": 20.0,
                "vram_mb": 1000.0,
                "route_switch_rate": 0.0,
                "pool_reuse_rate": 1.0,
            },
            {
                "resident_fraction": 0.25,
                "success": 0,
                "return_score": 0.2,
                "route_regret": 0.4,
                "flops": 4.2,
                "latency_ms": 22.0,
                "vram_mb": 1020.0,
                "route_switch_rate": 0.5,
                "pool_reuse_rate": 0.5,
            },
            {
                "resident_fraction": 1.0,
                "success": 1,
                "return_score": 0.9,
                "route_regret": 0.0,
                "flops": 8.0,
                "latency_ms": 30.0,
                "vram_mb": 1400.0,
                "route_switch_rate": 0.25,
                "pool_reuse_rate": 0.75,
            },
        ]

        summary = protocol_summary_from_records(
            records,
            benchmark="SO101",
            method="LARA",
            total_experts=8,
            active_experts=2,
            shared_params=100,
            params_per_expert=10,
        )

        self.assertEqual(summary["num_records"], 3)
        self.assertEqual(summary["num_records_by_fraction"], {"0.25": 2, "1": 1})
        self.assertAlmostEqual(summary["curve"]["success_at_resident_0.25"], 0.5)
        self.assertAlmostEqual(summary["curve"]["success_at_resident_1"], 1.0)
        self.assertEqual(len(summary["rows"]), 2)
        self.assertEqual(summary["rows"][0]["resident_experts"], 2)
        self.assertAlmostEqual(summary["rows"][0]["flops"], 4.1, places=6)
        self.assertIn("compute_success_pareto", summary["rows"][0])
        self.assertAlmostEqual(summary["route_diagnostics_by_fraction"]["route_switch_rate"]["0.25"], 0.25)
        self.assertAlmostEqual(summary["route_diagnostics_by_fraction"]["pool_reuse_rate"]["1"], 0.75)

    def test_protocol_summary_from_records_requires_success_and_fraction(self):
        with self.assertRaisesRegex(ValueError, "resident_fraction"):
            protocol_summary_from_records(
                [{"success": 1}],
                benchmark="SO101",
                method="LARA",
                total_experts=8,
                active_experts=2,
            )

    def test_protocol_evidence_audit_reports_missing_paper_metrics(self):
        records = [
            {
                "resident_fraction": 0.5,
                "success": 1,
                "flops": 4.0,
                "latency_ms": 20.0,
                "router_probs_sequence": [[0.9, 0.1], [0.8, 0.2]],
                "pool_mask_sequence": [[True, True], [True, True]],
            },
            {
                "resident_fraction": 1.0,
                "success": 1,
                "flops": 8.0,
                "latency_ms": 30.0,
                "vram_mb": 1400.0,
                "router_probs_sequence": [[0.9, 0.1], [0.2, 0.8]],
                "pool_mask_sequence": [[True, True], [True, True]],
            },
        ]

        audit = protocol_evidence_audit(
            records,
            required_fractions=[0.25, 0.5, 1.0],
        )

        self.assertFalse(audit["ok"])
        self.assertEqual(audit["missing_required_fractions"], ["0.25"])
        self.assertIn("vram_mb", audit["missing_metrics_by_fraction"]["0.5"])
        self.assertNotIn("pool_reuse_rate", audit["missing_metrics_by_fraction"]["0.5"])

    def test_protocol_evidence_audit_accepts_complete_required_records(self):
        records = [
            {
                "resident_fraction": 0.5,
                "success": 1,
                "flops": 4.0,
                "latency_ms": 20.0,
                "vram_mb": 1000.0,
                "router_probs_sequence": [[0.9, 0.1], [0.8, 0.2]],
                "pool_mask_sequence": [[True, True], [True, True]],
            },
            {
                "resident_fraction": 1.0,
                "success_rate": 1.0,
                "flops": 8.0,
                "latency_ms": 30.0,
                "vram_mb": 1400.0,
                "router_probs_sequence": [[0.9, 0.1], [0.2, 0.8]],
                "pool_mask_sequence": [[True, True], [True, True]],
            },
        ]

        audit = protocol_evidence_audit(records, required_fractions=[0.5, 1.0])

        self.assertTrue(audit["ok"])
        self.assertEqual(audit["missing_metrics_by_fraction"], {})
        with self.assertRaisesRegex(ValueError, "success"):
            protocol_summary_from_records(
                [{"resident_fraction": 0.5}],
                benchmark="SO101",
                method="LARA",
                total_experts=8,
                active_experts=2,
            )

    def test_counterfactual_utility_matrix_groups_context_candidate_records(self):
        labels = counterfactual_utility_matrix_from_records(
            [
                {"context_id": "ep0:0", "expert_id": 0, "success": 1.0, "latency_ms": 10.0},
                {"context_id": "ep0:0", "candidate_expert_id": 1, "success": 0.5, "latency_ms": 20.0},
                {"context_id": "ep0:0", "expert_id": 1, "success": 1.0, "latency_ms": 10.0},
                {"context_id": "ep0:10", "expert_id": 1, "return_score": 0.6, "latency_ms": 5.0},
                {"context_id": "ep0:10", "expert_id": 2, "return_score": 0.4, "latency_ms": 5.0},
            ],
            num_experts=3,
            cost_weight=0.01,
        )

        self.assertEqual(labels["context_ids"], ["ep0:0", "ep0:10"])
        self.assertEqual(labels["utility_scores"].shape, (2, 3))
        self.assertTrue(torch.equal(labels["utility_candidate_mask"][0], torch.tensor([True, True, False])))
        self.assertTrue(torch.equal(labels["utility_candidate_mask"][1], torch.tensor([False, True, True])))
        self.assertTrue(torch.equal(labels["candidate_counts"][0], torch.tensor([1, 2, 0])))
        self.assertAlmostEqual(float(labels["utility_scores"][0, 0]), 0.9, places=6)
        self.assertAlmostEqual(float(labels["utility_scores"][0, 1]), 0.6, places=6)
        self.assertAlmostEqual(float(labels["utility_scores"][1, 1]), 0.55, places=6)
        self.assertEqual(labels["missing_candidates"], {"ep0:0": [2], "ep0:10": [0]})

    def test_counterfactual_utility_matrix_can_use_trajectory_step_keys(self):
        labels = counterfactual_utility_matrix_from_records(
            [
                {"trajectory_id": 7, "base_index": 20, "expert_id": 0, "success": 1.0},
                {"trajectory_id": 7, "base_index": 20, "expert_id": 1, "success": 0.0},
            ],
            num_experts=3,
        )

        self.assertEqual(step_context_id(7, 20), "7:20")
        self.assertEqual(labels["context_ids"], ["7:20"])
        self.assertTrue(torch.equal(labels["utility_candidate_mask"][0], torch.tensor([True, True, False])))

    def test_counterfactual_records_from_forced_rollouts_build_sidecar_records(self):
        records = counterfactual_utility_records_from_rollouts(
            [
                {
                    "trajectory_id": 7,
                    "base_index": 20,
                    "forced_expert_id_sequence": [1, 1],
                    "success": 1,
                    "latency_ms": 12.5,
                },
                {
                    "trajectory_id": 7,
                    "base_index": 20,
                    "candidate_expert_id": 2,
                    "return_score": 0.25,
                },
            ],
            num_experts=3,
        )

        self.assertEqual(records[0]["context_id"], "7:20")
        self.assertEqual(records[0]["expert_id"], 1)
        self.assertEqual(records[0]["utility_score"], 1.0)
        self.assertEqual(records[0]["utility_cost"], 12.5)
        self.assertEqual(records[1]["expert_id"], 2)
        self.assertEqual(records[1]["utility_score"], 0.25)

    def test_counterfactual_records_from_forced_rollouts_reject_ambiguous_routes(self):
        with self.assertRaisesRegex(ValueError, "exactly one forced expert"):
            counterfactual_utility_records_from_rollouts(
                [
                    {
                        "context_id": "ep0:0",
                        "forced_expert_id_sequence": [1, 2],
                        "success": 1,
                    }
                ],
                num_experts=3,
            )

    def test_lerobot_utility_label_loader_indexes_by_step_context(self):
        from Lara.dataloader.gr00t_lerobot.datasets import load_counterfactual_utility_label_index

        with TemporaryDirectory() as tmpdir:
            labels_path = Path(tmpdir) / "utility.jsonl"
            labels_path.write_text(
                "\n".join(
                    [
                        '{"trajectory_id": 3, "base_index": 5, "expert_id": 0, "success": 1.0}',
                        '{"trajectory_id": 3, "base_index": 5, "expert_id": 1, "success": 0.25}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            label_index = load_counterfactual_utility_label_index(labels_path, num_experts=3)

        self.assertEqual(sorted(label_index), ["3:5"])
        self.assertEqual(label_index["3:5"]["utility_scores"].dtype.name, "float32")
        self.assertEqual(label_index["3:5"]["utility_candidate_mask"].tolist(), [True, True, False])
        self.assertAlmostEqual(float(label_index["3:5"]["utility_scores"][1]), 0.25, places=6)

    def test_lerobot_mixture_can_sample_only_labeled_utility_steps(self):
        from Lara.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset

        class DummyDataset:
            all_steps = [(3, 5), (3, 6)]

        dataset = object.__new__(LeRobotMixtureDataset)
        dataset.datasets = [DummyDataset()]
        dataset.mode = "eval"
        dataset.epoch = 0
        dataset.seed = 42
        dataset._dataset_sampling_weights = np.array([1.0])
        dataset.counterfactual_utility_sample_labeled_only = True
        dataset._counterfactual_utility_labeled_step_indices = [np.array([1], dtype=np.int64)]

        _, trajectory_id, base_index = dataset.sample_step(0)

        self.assertEqual((trajectory_id, base_index), (3, 6))

    def test_counterfactual_utility_matrix_rejects_single_route_labels(self):
        with self.assertRaisesRegex(ValueError, "at least 2 candidates per context"):
            counterfactual_utility_matrix_from_records(
                [{"context_id": "ep0:0", "expert_id": 0, "success": 1.0}],
                num_experts=3,
            )

        with self.assertRaisesRegex(ValueError, "missing counterfactual candidates"):
            counterfactual_utility_matrix_from_records(
                [
                    {"context_id": "ep0:0", "expert_id": 0, "success": 1.0},
                    {"context_id": "ep0:0", "expert_id": 1, "success": 0.0},
                ],
                num_experts=3,
                require_all_experts=True,
            )

        with self.assertRaisesRegex(ValueError, "expert_id"):
            counterfactual_utility_matrix_from_records(
                [{"context_id": "ep0:0", "success": 1.0}],
                num_experts=3,
            )


if __name__ == "__main__":
    unittest.main()
