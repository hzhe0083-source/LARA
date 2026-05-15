import unittest

import numpy as np
import torch

from Lara.dataloader.lerobot_v3_datasets import collate_fn


class LeRobotV3CollateTest(unittest.TestCase):
    def test_collate_exposes_explicit_future_actions_and_current_state(self):
        batch = [
            {
                "observation.image": torch.zeros(2, 3, 4, 4),
                "observation.state": torch.arange(6, dtype=torch.float32).view(2, 3),
                "action": torch.ones(3, 2),
                "task": "pick",
            }
        ]

        examples = collate_fn(
            batch,
            img_keys=["observation.image"],
            state_key="observation.state",
            action_key="action",
            task_key="task",
            resize_size=8,
            execution_horizon=1,
            prediction_horizon=2,
        )

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["lang"], "pick")
        self.assertEqual(len(examples[0]["image"]), 1)
        self.assertTrue(np.array_equal(examples[0]["future_actions"], examples[0]["action"]))
        self.assertTrue(np.array_equal(examples[0]["current_state"], examples[0]["state"]))
        self.assertEqual(examples[0]["current_state"].shape, (1, 3))
        self.assertTrue(np.array_equal(examples[0]["execution_state_target"], np.array([[3, 4, 5]], dtype=np.float32)))
        self.assertTrue(np.array_equal(examples[0]["prediction_state_target"], np.array([[3, 4, 5]], dtype=np.float32)))
        self.assertTrue(examples[0]["execution_state_target_mask"])
        self.assertFalse(examples[0]["prediction_state_target_mask"])


if __name__ == "__main__":
    unittest.main()
