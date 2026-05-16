import unittest

import torch

from Lara.training.trainer_utils.trainer_tools import (
    action_eval_metrics,
    scalarize_metrics,
    split_loss_and_metric_outputs,
)


class TrainerMetricsTest(unittest.TestCase):
    def test_split_loss_and_metric_outputs_keeps_metrics_out_of_loss_dict(self):
        output = {
            "action_loss": torch.tensor(2.0, requires_grad=True),
            "wm_loss": torch.tensor(1.0, requires_grad=True),
            "metric/moe_route_regret": torch.tensor(0.5),
        }

        losses, metrics = split_loss_and_metric_outputs(output)

        self.assertEqual(set(losses), {"action_loss", "wm_loss"})
        self.assertEqual(set(metrics), {"moe_route_regret"})
        total_loss = sum(losses.values())
        self.assertTrue(torch.allclose(total_loss, torch.tensor(3.0)))
        self.assertEqual(scalarize_metrics(metrics)["moe_route_regret"], 0.5)

    def test_action_eval_metrics_are_mask_aware_and_split_horizons(self):
        predicted = torch.tensor(
            [
                [[1.0, 1.0], [4.0, 4.0], [10.0, 10.0]],
            ]
        ).numpy()
        target = torch.tensor(
            [
                [[0.0, 0.0], [2.0, 2.0], [100.0, 100.0]],
            ]
        ).numpy()
        mask = torch.tensor([[True, True, False]]).numpy()

        metrics = action_eval_metrics(predicted, target, action_mask=mask, execution_horizon=1)

        self.assertEqual(metrics["eval/valid_action_steps"], 2.0)
        self.assertEqual(metrics["eval/prediction_horizon"], 3.0)
        self.assertEqual(metrics["eval/execution_horizon"], 1.0)
        self.assertAlmostEqual(metrics["eval/full_horizon_mae"], 1.5)
        self.assertAlmostEqual(metrics["eval/execution_horizon_mae"], 1.0)
        self.assertGreater(metrics["eval/full_horizon_rmse"], metrics["eval/execution_horizon_rmse"])


if __name__ == "__main__":
    unittest.main()
