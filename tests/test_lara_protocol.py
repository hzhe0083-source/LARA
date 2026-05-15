import unittest

from Lara.evaluation.lara_protocol import (
    matched_budget_flags,
    matched_compute_row,
    matched_expert_budget_flags,
    pareto_frontier_flags,
    resident_experts_for_fraction,
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


if __name__ == "__main__":
    unittest.main()
