import unittest

from Lara.evaluation.lara_protocol import (
    matched_budget_flags,
    matched_compute_row,
    pareto_frontier_flags,
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
