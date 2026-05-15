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


class FakeActionHead(torch.nn.Module):
    def __init__(self, config, context_hidden_size):
        super().__init__()
        self.config = config
        self.context_hidden_size = context_hidden_size
        self.loss_scale = torch.nn.Parameter(torch.tensor(1.0))
        self.calls = []
        self.lara_moe = object() if config.framework.action_model.get("use_lara_moe", False) else None

    def conditioning_tokens(self, embodied_action_tokens, latent_action_tokens=None):
        if latent_action_tokens is None:
            return embodied_action_tokens
        return torch.cat([latent_action_tokens, embodied_action_tokens], dim=1)

    def forward(
        self,
        embodied_action_tokens,
        latent_action_tokens,
        actions,
        actions_are_future=False,
        action_mask=None,
        past_actions=None,
        state=None,
        trajectory_ids=None,
        pool_mask=None,
        utility_scores=None,
        utility_candidate_mask=None,
        utility_cost_scores=None,
        utility_value_targets=None,
        utility_progress_targets=None,
        utility_uncertainty_targets=None,
        utility_target_mask=None,
        execution_state_target=None,
        prediction_state_target=None,
        execution_state_target_mask=None,
        prediction_state_target_mask=None,
        previous_router_probs=None,
        initial_context_tokens=None,
        return_aux=False,
    ):
        self.calls.append(
            {
                "embodied_action_tokens": embodied_action_tokens,
                "latent_action_tokens": latent_action_tokens,
                "actions": actions,
                "actions_are_future": actions_are_future,
                "action_mask": action_mask,
                "past_actions": past_actions,
                "state": state,
                "trajectory_ids": trajectory_ids,
                "pool_mask": pool_mask,
                "utility_scores": utility_scores,
                "utility_candidate_mask": utility_candidate_mask,
                "utility_cost_scores": utility_cost_scores,
                "utility_value_targets": utility_value_targets,
                "utility_progress_targets": utility_progress_targets,
                "utility_uncertainty_targets": utility_uncertainty_targets,
                "utility_target_mask": utility_target_mask,
                "execution_state_target": execution_state_target,
                "prediction_state_target": prediction_state_target,
                "execution_state_target_mask": execution_state_target_mask,
                "prediction_state_target_mask": prediction_state_target_mask,
                "previous_router_probs": previous_router_probs,
                "initial_context_tokens": initial_context_tokens,
                "return_aux": return_aux,
            }
        )
        return {
            "total_action_loss": self.loss_scale * 3.0,
            "moe_route_regret": torch.tensor(0.5),
            "moe_utility_scores": torch.ones(2, 3),
        }

    def predict_action(
        self,
        embodied_action_tokens,
        latent_action_tokens,
        state=None,
        pool_mask=None,
        previous_router_probs=None,
        forced_router_probs=None,
        return_aux=False,
    ):
        self.calls.append(
            {
                "predict": True,
                "embodied_action_tokens": embodied_action_tokens,
                "latent_action_tokens": latent_action_tokens,
                "state": state,
                "pool_mask": pool_mask,
                "previous_router_probs": previous_router_probs,
                "forced_router_probs": forced_router_probs,
                "return_aux": return_aux,
            }
        )
        batch_size = embodied_action_tokens.shape[0]
        actions = torch.ones(batch_size, 3, 2)
        if return_aux:
            return {
                "actions": actions,
                "router_probs": torch.full((batch_size, 2), 0.5),
                "active_mask": torch.tensor([[True, False]]).expand(batch_size, -1),
            }
        return actions


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
                "future_action_mask": np.array([True, True, False], dtype=bool),
                "past_actions": np.ones((2, 2), dtype=np.float32) * -5,
                "state": np.ones((1, 3), dtype=np.float32),
                "current_state": np.ones((1, 3), dtype=np.float32) * 9,
                "execution_state_target": np.ones((1, 3), dtype=np.float32) * 10,
                "execution_state_target_mask": True,
                "prediction_state_target": np.ones((1, 3), dtype=np.float32) * 11,
                "prediction_state_target_mask": False,
                "trajectory_id": 11,
                "pool_mask": np.array([True, True, False], dtype=bool),
                "utility_scores": np.array([1.0, 0.0, -1.0], dtype=np.float32),
                "utility_candidate_mask": np.array([True, True, False], dtype=bool),
                "utility_cost_scores": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                "utility_value_targets": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "utility_progress_targets": np.array([0.5, 0.6, 0.7], dtype=np.float32),
                "utility_uncertainty_targets": np.array([0.2, 0.1, 0.3], dtype=np.float32),
                "utility_target_mask": np.array([True, False, True], dtype=bool),
                "previous_router_probs": np.array([0.5, 0.3, 0.2], dtype=np.float32),
            },
            {
                "image": ["image-1"],
                "video": np.zeros((1, 2, 2, 2, 3), dtype=np.uint8),
                "lang": "place",
                "action": np.ones((3, 2), dtype=np.float32) * 2,
                "future_actions": np.ones((3, 2), dtype=np.float32) * 7,
                "future_action_mask": np.array([True, False, False], dtype=bool),
                "past_actions": np.ones((2, 2), dtype=np.float32) * -7,
                "state": np.ones((1, 3), dtype=np.float32) * 2,
                "current_state": np.ones((1, 3), dtype=np.float32) * 8,
                "execution_state_target": np.ones((1, 3), dtype=np.float32) * 20,
                "execution_state_target_mask": True,
                "prediction_state_target": np.ones((1, 3), dtype=np.float32) * 21,
                "prediction_state_target_mask": True,
                "trajectory_id": 12,
                "pool_mask": np.array([False, True, True], dtype=bool),
                "utility_scores": np.array([0.0, 1.0, -1.0], dtype=np.float32),
                "utility_candidate_mask": np.array([False, True, True], dtype=bool),
                "utility_cost_scores": np.array([0.4, 0.5, 0.6], dtype=np.float32),
                "utility_value_targets": np.array([2.0, 3.0, 4.0], dtype=np.float32),
                "utility_progress_targets": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                "utility_uncertainty_targets": np.array([0.7, 0.8, 0.9], dtype=np.float32),
                "utility_target_mask": np.array([False, True, True], dtype=bool),
                "previous_router_probs": np.array([0.2, 0.3, 0.5], dtype=np.float32),
            },
        ]

        output = model(examples)

        self.assertTrue(torch.allclose(output["action_loss"], torch.tensor(3.0)))
        self.assertTrue(torch.allclose(output["wm_loss"], torch.tensor(0.2)))
        self.assertTrue(torch.allclose(output["metric/moe_route_regret"], torch.tensor(0.5)))
        self.assertNotIn("metric/moe_utility_scores", output)
        self.assertEqual(model.qwen.num_prediction_steps, 2)
        self.assertEqual(len(model.qwen.encode_calls), 1)
        self.assertTrue(model.qwen.encode_calls[0]["include_embodied_tokens"])
        self.assertEqual(model.qwen.encode_calls[0]["prompt_template"], "act {actions} {e_actions}")

        action_call = model.action_head.calls[0]
        self.assertEqual(action_call["trajectory_ids"], [11, 12])
        self.assertTrue(action_call["return_aux"])
        self.assertTrue(action_call["actions_are_future"])
        self.assertTrue(np.all(action_call["actions"][0] == examples[0]["future_actions"]))
        self.assertTrue(np.all(action_call["actions"][1] == examples[1]["future_actions"]))
        self.assertTrue(np.all(action_call["action_mask"][0] == examples[0]["future_action_mask"]))
        self.assertTrue(np.all(action_call["action_mask"][1] == examples[1]["future_action_mask"]))
        self.assertTrue(np.all(action_call["past_actions"][0] == examples[0]["past_actions"]))
        self.assertTrue(np.all(action_call["past_actions"][1] == examples[1]["past_actions"]))
        self.assertTrue(np.all(action_call["state"][0] == examples[0]["current_state"]))
        self.assertTrue(np.all(action_call["state"][1] == examples[1]["current_state"]))
        self.assertTrue(np.all(action_call["pool_mask"][0] == examples[0]["pool_mask"]))
        self.assertTrue(np.all(action_call["utility_scores"][1] == examples[1]["utility_scores"]))
        self.assertTrue(np.all(action_call["utility_candidate_mask"][0] == examples[0]["utility_candidate_mask"]))
        self.assertTrue(np.all(action_call["utility_cost_scores"][1] == examples[1]["utility_cost_scores"]))
        self.assertTrue(np.all(action_call["utility_value_targets"][0] == examples[0]["utility_value_targets"]))
        self.assertTrue(np.all(action_call["utility_progress_targets"][1] == examples[1]["utility_progress_targets"]))
        self.assertTrue(np.all(action_call["utility_uncertainty_targets"][0] == examples[0]["utility_uncertainty_targets"]))
        self.assertTrue(np.all(action_call["utility_target_mask"][1] == examples[1]["utility_target_mask"]))
        self.assertTrue(np.all(action_call["previous_router_probs"][0] == examples[0]["previous_router_probs"]))
        self.assertTrue(np.all(action_call["execution_state_target"][0] == examples[0]["execution_state_target"]))
        self.assertTrue(np.all(action_call["prediction_state_target"][1] == examples[1]["prediction_state_target"]))
        self.assertEqual(action_call["execution_state_target_mask"], [True, True])
        self.assertEqual(action_call["prediction_state_target_mask"], [False, True])
        self.assertIsNone(action_call["initial_context_tokens"])

    def test_forward_uses_episode_start_image_for_moe_pool_context(self):
        config = tiny_framework_config()
        config.framework.action_model.use_lara_moe = True
        with (
            patch.object(Lara_core, "QwenActionTokenizer", FakeQwen),
            patch.object(Lara_core, "VJ2WorldModel", FakeVJ2),
            patch.object(Lara_core, "ActionHeadAdapter", FakeActionHead),
        ):
            model = Lara_core.Lara(config=config)

        examples = [
            {
                "image": ["current-0"],
                "episode_start_image": ["start-0"],
                "video": np.zeros((1, 2, 2, 2, 3), dtype=np.uint8),
                "lang": "pick",
                "future_actions": np.ones((3, 2), dtype=np.float32),
                "current_state": np.ones((1, 3), dtype=np.float32),
                "trajectory_id": 11,
            },
            {
                "image": ["current-1"],
                "episode_start_image": ["start-1"],
                "video": np.zeros((1, 2, 2, 2, 3), dtype=np.uint8),
                "lang": "place",
                "future_actions": np.ones((3, 2), dtype=np.float32),
                "current_state": np.ones((1, 3), dtype=np.float32),
                "trajectory_id": 12,
            },
        ]

        model(examples)

        self.assertEqual(len(model.qwen.encode_calls), 2)
        self.assertEqual(model.qwen.encode_calls[0]["images"], [["current-0"], ["current-1"]])
        self.assertEqual(model.qwen.encode_calls[1]["images"], [["start-0"], ["start-1"]])
        self.assertTrue(model.qwen.encode_calls[1]["include_embodied_tokens"])
        action_call = model.action_head.calls[0]
        self.assertIsNotNone(action_call["initial_context_tokens"])
        self.assertEqual(tuple(action_call["initial_context_tokens"].shape), (2, 5, 8))

    def test_fake_batch_loss_can_update_action_head_parameter(self):
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
                "future_actions": np.ones((3, 2), dtype=np.float32),
                "current_state": np.ones((1, 3), dtype=np.float32),
                "trajectory_id": 11,
            }
        ]
        optimizer = torch.optim.SGD(model.action_head.parameters(), lr=0.1)
        before = model.action_head.loss_scale.detach().clone()

        output = model(examples)
        total_loss = output["action_loss"] + output["wm_loss"]
        total_loss.backward()
        optimizer.step()

        self.assertIsNotNone(model.action_head.loss_scale.grad)
        self.assertFalse(torch.allclose(model.action_head.loss_scale.detach(), before))

    def test_predict_action_passes_previous_router_probs_and_returns_route_aux(self):
        config = tiny_framework_config()
        with (
            patch.object(Lara_core, "QwenActionTokenizer", FakeQwen),
            patch.object(Lara_core, "VJ2WorldModel", FakeVJ2),
            patch.object(Lara_core, "ActionHeadAdapter", FakeActionHead),
        ):
            model = Lara_core.Lara(config=config)

        previous_router_probs = np.array([[0.8, 0.2]], dtype=np.float32)

        output = model.predict_action(
            batch_images=[["image-0"]],
            instructions=["pick"],
            state=[np.ones((1, 3), dtype=np.float32)],
            previous_router_probs=previous_router_probs,
        )

        action_call = model.action_head.calls[0]
        self.assertTrue(action_call["predict"])
        self.assertTrue(action_call["return_aux"])
        self.assertTrue(np.all(action_call["previous_router_probs"] == previous_router_probs))
        self.assertEqual(output["normalized_actions"].shape, (1, 3, 2))
        self.assertEqual(output["execution_normalized_actions"].shape, (1, 1, 2))
        self.assertTrue(np.all(output["router_probs"] == 0.5))
        self.assertTrue(np.all(output["active_expert_mask"] == np.array([[True, False]])))

    def test_predict_action_can_force_single_expert_for_counterfactual_eval(self):
        config = tiny_framework_config()
        config.framework.action_model.use_lara_moe = True
        with (
            patch.object(Lara_core, "QwenActionTokenizer", FakeQwen),
            patch.object(Lara_core, "VJ2WorldModel", FakeVJ2),
            patch.object(Lara_core, "ActionHeadAdapter", FakeActionHead),
        ):
            model = Lara_core.Lara(config=config)
        model.action_head.lara_moe = SimpleNamespace(num_experts=3)

        output = model.predict_action(
            batch_images=[["image-0"], ["image-1"]],
            instructions=["pick", "place"],
            state=np.zeros((2, 1, 3), dtype=np.float32),
            forced_expert_id=[1, 2],
        )

        expected_forced = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        action_call = model.action_head.calls[0]
        self.assertTrue(np.array_equal(action_call["forced_router_probs"], expected_forced))
        self.assertTrue(np.array_equal(action_call["pool_mask"], expected_forced.astype(bool)))
        self.assertTrue(np.array_equal(output["forced_router_probs"], expected_forced))
        self.assertTrue(np.array_equal(output["forced_expert_id"], np.array([1, 2])))


if __name__ == "__main__":
    unittest.main()
