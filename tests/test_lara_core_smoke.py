import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from Lara.model.framework import Lara_core


def tiny_framework_config():
    return OmegaConf.create(
        {
            "framework": {
                "action_model": {
                    "future_action_window_size": 2,
                    "action_horizon": 3,
                    "execution_horizon": 1,
                    "past_action_window_size": 0,
                },
            },
            "datasets": {
                "vla_data": {"CoT_prompt": "act {actions} {e_actions}"},
                "video_data": {"CoT_prompt": "watch {actions}"},
            },
        }
    )


class FakeQwen:
    def __init__(self, config):
        self.config = config
        self.hidden_size = 8
        self.num_prediction_steps = None
        self.encode_calls = []

    def configure_world_tokens(self, num_prediction_steps):
        self.num_prediction_steps = num_prediction_steps

    def encode(self, images, instructions, prompt_template, include_embodied_tokens):
        self.encode_calls.append(
            {
                "images": images,
                "instructions": instructions,
                "prompt_template": prompt_template,
                "include_embodied_tokens": include_embodied_tokens,
            }
        )
        batch_size = len(instructions)
        return SimpleNamespace(
            action_tokens=torch.ones(batch_size, 2, self.hidden_size),
            embodied_action_tokens=torch.ones(batch_size, 3, self.hidden_size) * 2,
        )


class FakeVJ2:
    def __init__(self, config, action_embed_dim):
        self.config = config
        self.action_embed_dim = action_embed_dim
        self.num_prediction_steps = 2
        self.calls = []

    def __call__(self, batch_videos, action_tokens):
        self.calls.append({"batch_videos": batch_videos, "action_tokens": action_tokens})
        return torch.tensor(2.0)


class FakeActionHead:
    def __init__(self, config, context_hidden_size):
        self.config = config
        self.context_hidden_size = context_hidden_size
        self.calls = []

    def __call__(
        self,
        embodied_action_tokens,
        latent_action_tokens,
        actions,
        state,
        trajectory_ids,
        return_aux,
    ):
        self.calls.append(
            {
                "embodied_action_tokens": embodied_action_tokens,
                "latent_action_tokens": latent_action_tokens,
                "actions": actions,
                "state": state,
                "trajectory_ids": trajectory_ids,
                "return_aux": return_aux,
            }
        )
        return {"total_action_loss": torch.tensor(3.0)}


class LaraCoreSmokeTest(unittest.TestCase):
    def test_forward_uses_future_actions_and_passes_trajectory_ids(self):
        config = tiny_framework_config()
        with (
            patch.object(Lara_core, "QwenActionTokenizer", FakeQwen),
            patch.object(Lara_core, "VJ2WorldModel", FakeVJ2),
            patch.object(Lara_core, "ActionHeadAdapter", FakeActionHead),
        ):
            model = Lara_core.Lara(config=config)

        examples = [
            {
                "image": ["image-0"],
                "video": np.zeros((1, 2, 2, 2, 3), dtype=np.uint8),
                "lang": "pick",
                "action": np.ones((3, 2), dtype=np.float32),
                "future_actions": np.ones((3, 2), dtype=np.float32) * 5,
                "state": np.ones((1, 3), dtype=np.float32),
                "trajectory_id": 11,
            },
            {
                "image": ["image-1"],
                "video": np.zeros((1, 2, 2, 2, 3), dtype=np.uint8),
                "lang": "place",
                "action": np.ones((3, 2), dtype=np.float32) * 2,
                "future_actions": np.ones((3, 2), dtype=np.float32) * 7,
                "state": np.ones((1, 3), dtype=np.float32) * 2,
                "trajectory_id": 12,
            },
        ]

        output = model(examples)

        self.assertTrue(torch.allclose(output["action_loss"], torch.tensor(3.0)))
        self.assertTrue(torch.allclose(output["wm_loss"], torch.tensor(0.2)))
        self.assertEqual(model.qwen.num_prediction_steps, 2)
        self.assertTrue(model.qwen.encode_calls[0]["include_embodied_tokens"])
        self.assertEqual(model.qwen.encode_calls[0]["prompt_template"], "act {actions} {e_actions}")

        action_call = model.action_head.calls[0]
        self.assertEqual(action_call["trajectory_ids"], [11, 12])
        self.assertTrue(action_call["return_aux"])
        self.assertTrue(np.all(action_call["actions"][0] == examples[0]["future_actions"]))
        self.assertTrue(np.all(action_call["actions"][1] == examples[1]["future_actions"]))


if __name__ == "__main__":
    unittest.main()
