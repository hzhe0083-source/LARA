import json
import tempfile
import unittest
from pathlib import Path

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer


class _PoolEchoPolicy:
    def __init__(self):
        self.calls = []

    def predict_action(self, **payload):
        self.calls.append(payload)
        resident_pool_mask = payload.get("resident_pool_mask", [[True, False, True]])
        router_probs = payload.get("previous_router_probs", [[0.7, 0.2, 0.1]])
        response = {
            "normalized_actions": [],
            "execution_normalized_actions": [],
            "resident_pool_mask": resident_pool_mask,
            "router_probs": router_probs,
            "active_expert_mask": [[True, False, True]],
        }
        if "forced_expert_id" in payload:
            response["forced_expert_id"] = payload["forced_expert_id"]
        return response


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

    def test_reset_writes_lara_rollout_trace_jsonl_when_enabled(self):
        policy = _PoolEchoPolicy()
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "rollouts.jsonl"
            server = WebsocketPolicyServer(policy=policy, rollout_trace_path=str(trace_path))

            server._route_message(
                {
                    "type": "infer",
                    "session_id": "episode-a",
                    "payload": {"batch_images": [], "instructions": []},
                }
            )
            server._route_message(
                {
                    "type": "infer",
                    "session_id": "episode-a",
                    "payload": {"batch_images": [], "instructions": []},
                }
            )
            reset = server._route_message(
                {
                    "type": "reset",
                    "session_id": "episode-a",
                    "payload": {"success": 1, "return_score": 0.75},
                }
            )

            self.assertTrue(reset["rollout_trace_written"])
            records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["session_id"], "episode-a")
            self.assertEqual(records[0]["success"], 1)
            self.assertEqual(records[0]["router_probs_sequence"][0], [0.7, 0.2, 0.1])
            self.assertEqual(len(records[0]["router_probs_sequence"]), 2)
            self.assertEqual(records[0]["active_mask_sequence"][0], [True, False, True])
            self.assertEqual(len(records[0]["latency_ms_sequence"]), 2)
            self.assertGreaterEqual(records[0]["latency_ms"], 0.0)
            self.assertAlmostEqual(records[0]["resident_fraction"], 2.0 / 3.0)
            self.assertNotIn("episode-a", server._session_state)

    def test_record_outcome_writes_and_clears_trace_without_reset(self):
        policy = _PoolEchoPolicy()
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "rollouts.jsonl"
            server = WebsocketPolicyServer(policy=policy, rollout_trace_path=str(trace_path))

            server._route_message(
                {
                    "type": "infer",
                    "session_id": "episode-a",
                    "payload": {"batch_images": [], "instructions": []},
                }
            )
            outcome = server._route_message(
                {
                    "type": "record_outcome",
                    "session_id": "episode-a",
                    "payload": {"success": 0, "resident_fraction": 0.5},
                }
            )

            self.assertTrue(outcome["ok"])
            self.assertTrue(outcome["rollout_trace_written"])
            record = json.loads(trace_path.read_text(encoding="utf-8").strip())
            self.assertEqual(record["success"], 0)
            self.assertEqual(record["resident_fraction"], 0.5)
            self.assertEqual(len(record["latency_ms_sequence"]), 1)
            self.assertGreaterEqual(record["latency_ms"], 0.0)
            self.assertNotIn("episode-a", server._session_state)

    def test_rollout_trace_records_forced_expert_sequence(self):
        policy = _PoolEchoPolicy()
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "rollouts.jsonl"
            server = WebsocketPolicyServer(policy=policy, rollout_trace_path=str(trace_path))

            server._route_message(
                {
                    "type": "infer",
                    "session_id": "candidate-a",
                    "payload": {"batch_images": [], "instructions": [], "forced_expert_id": 2},
                }
            )
            server._route_message(
                {
                    "type": "record_outcome",
                    "session_id": "candidate-a",
                    "payload": {"success": 1, "candidate_expert_id": 2},
                }
            )

            record = json.loads(trace_path.read_text(encoding="utf-8").strip())
            self.assertEqual(policy.calls[0]["forced_expert_id"], 2)
            self.assertEqual(record["candidate_expert_id"], 2)
            self.assertEqual(record["forced_expert_id_sequence"], [2])

    def test_reset_payload_resource_metrics_override_measured_trace_metrics(self):
        policy = _PoolEchoPolicy()
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "rollouts.jsonl"
            server = WebsocketPolicyServer(policy=policy, rollout_trace_path=str(trace_path))

            server._route_message(
                {
                    "type": "infer",
                    "session_id": "episode-a",
                    "payload": {"batch_images": [], "instructions": []},
                }
            )
            server._route_message(
                {
                    "type": "reset",
                    "session_id": "episode-a",
                    "payload": {"success": 1, "latency_ms": 123.0, "vram_mb": 456.0},
                }
            )

            record = json.loads(trace_path.read_text(encoding="utf-8").strip())
            self.assertEqual(record["latency_ms"], 123.0)
            self.assertEqual(record["vram_mb"], 456.0)
            self.assertEqual(len(record["latency_ms_sequence"]), 1)

    def test_infer_response_includes_resource_metrics_when_trace_enabled(self):
        policy = _PoolEchoPolicy()
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "rollouts.jsonl"
            server = WebsocketPolicyServer(policy=policy, rollout_trace_path=str(trace_path))

            response = server._route_message(
                {
                    "type": "infer",
                    "session_id": "episode-a",
                    "payload": {"batch_images": [], "instructions": []},
                }
            )

            self.assertTrue(response["ok"])
            self.assertIn("latency_ms", response["data"])
            self.assertGreaterEqual(response["data"]["latency_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
