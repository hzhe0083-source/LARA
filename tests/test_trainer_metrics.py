import unittest

import torch

from Lara.training.trainer_utils.trainer_tools import scalarize_metrics, split_loss_and_metric_outputs


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


if __name__ == "__main__":
    unittest.main()
