import unittest

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer


class _PoolEchoPolicy:
    def __init__(self):
        self.calls = []

    def predict_action(self, **payload):
        self.calls.append(payload)
        resident_pool_mask = payload.get("resident_pool_mask", [[True, False, True]])
        router_probs = payload.get("previous_router_probs", [[0.7, 0.2, 0.1]])
        return {
            "normalized_actions": [],
            "execution_normalized_actions": [],
            "resident_pool_mask": resident_pool_mask,
            "router_probs": router_probs,
        }


class WebsocketPolicyServerTest(unittest.TestCase):
    def test_infer_reuses_resident_pool_mask_per_session(self):
        policy = _PoolEchoPolicy()
        server = WebsocketPolicyServer(policy=policy)

        first = server._route_message(
            {
                "type": "infer",
                "request_id": "r1",
                "session_id": "episode-a",
                "payload": {"batch_images": [], "instructions": []},
            }
        )
        second = server._route_message(
            {
                "type": "infer",
                "request_id": "r2",
                "session_id": "episode-a",
                "payload": {"batch_images": [], "instructions": []},
            }
        )

        self.assertTrue(first["ok"])
        self.assertTrue(second["ok"])
        self.assertNotIn("resident_pool_mask", policy.calls[0])
        self.assertEqual(policy.calls[1]["resident_pool_mask"], [[True, False, True]])
        self.assertEqual(policy.calls[1]["previous_router_probs"], [[0.7, 0.2, 0.1]])
        self.assertEqual(second["session_id"], "episode-a")

    def test_reset_clears_resident_pool_session_state(self):
        policy = _PoolEchoPolicy()
        server = WebsocketPolicyServer(policy=policy)

        server._route_message(
            {
                "type": "infer",
                "session_id": "episode-a",
                "payload": {"batch_images": [], "instructions": []},
            }
        )
        reset = server._route_message({"type": "reset", "session_id": "episode-a", "payload": {}})
        after_reset = server._route_message(
            {
                "type": "infer",
                "session_id": "episode-a",
                "payload": {"batch_images": [], "instructions": []},
            }
        )

        self.assertTrue(reset["ok"])
        self.assertTrue(after_reset["ok"])
        self.assertNotIn("resident_pool_mask", policy.calls[-1])
        self.assertNotIn("previous_router_probs", policy.calls[-1])


if __name__ == "__main__":
    unittest.main()
