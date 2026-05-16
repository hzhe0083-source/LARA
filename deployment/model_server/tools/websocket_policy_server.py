# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].

import asyncio
import json
import logging
from pathlib import Path
import time
import traceback

import websockets.asyncio.server
import websockets.frames

# from openpi_client import base_policy as _base_policy
from . import image_tools

try:
    from . import msgpack_numpy
except ModuleNotFoundError as exc:
    msgpack_numpy = None
    _msgpack_numpy_import_error = exc
else:
    _msgpack_numpy_import_error = None

class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
        rollout_trace_path: str | None = None,
    ) -> None:
        self._policy = policy  #
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._rollout_trace_path = Path(rollout_trace_path) if rollout_trace_path else None
        self._session_state: dict[str, dict] = {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        if msgpack_numpy is None:
            raise ImportError("msgpack is required to run the websocket policy server") from _msgpack_numpy_import_error
        asyncio.run(self.run())

    async def run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                msg = msgpack_numpy.unpackb(await websocket.recv())
                ret = self._route_message(msg)  # route message
                await websocket.send(packer.pack(ret))
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    # route logic: recognize request from client
    def _route_message(self, msg: dict) -> dict:
        """
        Route rules (fault-tolerant):
        - Supports messages of form:
            {"type": "ping|init|infer|reset|record_outcome", "request_id": "...", "payload": {...}}
          or a flat dict (will be treated as payload).
        - Always returns a dict containing:
            {
              "status": "ok" | "error",
              "ok": bool,
              "type": <str>,
              "request_id": <str>,
              ... (data | error)
            }
        - Does NOT raise inside this function: all exceptions are caught and encoded in response.
        """
        req_id = msg.get("request_id", "default")
        mtype = msg.get("type", "infer")          # default = infer
        payload = msg.get("payload", msg)         # when no explicit payload, treat top-level as payload

        session_id = str(
            msg.get("session_id", payload.get("session_id", "default"))
            if isinstance(payload, dict)
            else "default"
        )

        # ping
        if mtype == "ping":
            return {"status": "ok", "ok": True, "type": "ping", "request_id": req_id, "session_id": session_id}

        if mtype == "reset":
            trace_written = self._write_rollout_trace(
                session_id,
                self._session_state.get(session_id, {}),
                payload if isinstance(payload, dict) else {},
            )
            self._session_state.pop(session_id, None)
            return {
                "status": "ok",
                "ok": True,
                "type": "reset",
                "request_id": req_id,
                "session_id": session_id,
                "rollout_trace_written": trace_written,
            }

        if mtype == "record_outcome":
            if not isinstance(payload, dict):
                return {
                    "status": "error",
                    "ok": False,
                    "type": "record_outcome",
                    "request_id": req_id,
                    "session_id": session_id,
                    "error": {"message": "Payload must be a dict"},
                }
            trace_written = self._write_rollout_trace(
                session_id,
                self._session_state.get(session_id, {}),
                payload,
            )
            self._session_state.pop(session_id, None)
            return {
                "status": "ok",
                "ok": True,
                "type": "record_outcome",
                "request_id": req_id,
                "session_id": session_id,
                "rollout_trace_written": trace_written,
            }

        # infer
        elif mtype == "infer":
            # Basic payload sanity
            if not isinstance(payload, dict):
                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {"message": "Payload must be a dict", "payload_type": str(type(payload))}
                }
            try:
                policy_payload = dict(payload)
                policy_payload.pop("session_id", None)
                session = self._session_state.setdefault(session_id, {})
                if "resident_pool_mask" not in policy_payload and "resident_pool_mask" in session:
                    policy_payload["resident_pool_mask"] = session["resident_pool_mask"]
                if "previous_router_probs" not in policy_payload and "router_probs" in session:
                    policy_payload["previous_router_probs"] = session["router_probs"]
                measurement = self._begin_resource_measurement() if self._rollout_trace_path is not None else None
                policy_payload["batch_images"] = image_tools.to_pil_preserve(policy_payload["batch_images"])
                ouput_dict = self._policy.predict_action(**policy_payload)
                resource_metrics = self._end_resource_measurement(measurement) if measurement is not None else None
                if isinstance(ouput_dict, dict):
                    if "resident_pool_mask" in ouput_dict:
                        session["resident_pool_mask"] = ouput_dict["resident_pool_mask"]
                    if "resident_pool_probs" in ouput_dict:
                        session["resident_pool_probs"] = ouput_dict["resident_pool_probs"]
                    if "router_probs" in ouput_dict:
                        session["router_probs"] = ouput_dict["router_probs"]
                    self._append_rollout_trace(session, ouput_dict, resource_metrics=resource_metrics)
            except Exception as e:
                logging.exception("Policy inference error (request_id=%s)", req_id)
                logging.exception(e)
                
                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {
                        "message": str(e),
                        # "traceback": traceback.format_exc(),
                    },
                }
            data = ouput_dict
            if isinstance(data, dict) and resource_metrics is not None:
                data = dict(data)
                for metric_key, metric_value in resource_metrics.items():
                    data.setdefault(metric_key, metric_value)
            return {
                "status": "ok",
                "ok": True,
                "type": "inference_result",
                "request_id": req_id,
                "session_id": session_id,
                "data": data,
            }

        # unknow request type
        else:
            return {
                "status": "error",
                "ok": False,
                "type": "unknown",
                "request_id": req_id,
                "error": {"message": f"Unsupported message type '{mtype}'"},
            }

    @staticmethod
    def _cuda_module():
        try:
            import torch
        except Exception:
            return None
        try:
            if not torch.cuda.is_available():
                return None
        except Exception:
            return None
        return torch.cuda

    @classmethod
    def _begin_resource_measurement(cls) -> dict:
        cuda = cls._cuda_module()
        if cuda is not None:
            try:
                cuda.synchronize()
                cuda.reset_peak_memory_stats()
            except Exception:
                cuda = None
        return {"start": time.perf_counter(), "cuda": cuda}

    @staticmethod
    def _end_resource_measurement(measurement: dict) -> dict:
        cuda = measurement.get("cuda")
        vram_mb = None
        if cuda is not None:
            try:
                cuda.synchronize()
                vram_mb = cuda.max_memory_allocated() / (1024 * 1024)
            except Exception:
                vram_mb = None
        metrics = {"latency_ms": (time.perf_counter() - measurement["start"]) * 1000.0}
        if vram_mb is not None:
            metrics["vram_mb"] = vram_mb
        return metrics

    @staticmethod
    def _jsonable(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "tolist"):
            return value.tolist()
        if hasattr(value, "item"):
            return value.item()
        if isinstance(value, dict):
            return {str(k): WebsocketPolicyServer._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [WebsocketPolicyServer._jsonable(v) for v in value]
        return value

    @staticmethod
    def _is_matrix(value) -> bool:
        return isinstance(value, list) and bool(value) and isinstance(value[0], list)

    @classmethod
    def _sequence_from_steps(cls, steps: list):
        if not steps:
            return None
        steps = [cls._jsonable(step) for step in steps]
        if all(cls._is_matrix(step) and len(step) == 1 for step in steps):
            return [step[0] for step in steps]
        if all(cls._is_matrix(step) for step in steps):
            batch_size = len(steps[0])
            if all(len(step) == batch_size for step in steps):
                return [[step[batch_idx] for step in steps] for batch_idx in range(batch_size)]
        return steps

    @staticmethod
    def _resident_fraction_from_pool_sequence(pool_sequence) -> float | None:
        if not isinstance(pool_sequence, list) or not pool_sequence:
            return None
        first = pool_sequence[0]
        if isinstance(first, list) and first and isinstance(first[0], list):
            first = first[0]
        if not isinstance(first, list) or not first:
            return None
        return float(sum(1 for value in first if bool(value)) / len(first))

    @staticmethod
    def _route_sequence_length(sequence) -> int | None:
        if not isinstance(sequence, list):
            return None
        if sequence and isinstance(sequence[0], list) and sequence[0] and isinstance(sequence[0][0], list):
            return len(sequence[0])
        return len(sequence)

    def _append_rollout_trace(self, session: dict, output: dict, *, resource_metrics: dict | None = None) -> None:
        if self._rollout_trace_path is None:
            return
        trace = session.setdefault("rollout_trace", {})
        for output_key, trace_key in [
            ("router_probs", "router_probs_sequence"),
            ("active_expert_mask", "active_mask_sequence"),
            ("resident_pool_mask", "pool_mask_sequence"),
            ("forced_expert_id", "forced_expert_id_sequence"),
        ]:
            if output_key in output:
                trace.setdefault(trace_key, []).append(self._jsonable(output[output_key]))
        for metric_key in ["latency_ms", "vram_mb"]:
            if resource_metrics is not None and metric_key in resource_metrics:
                trace.setdefault(f"{metric_key}_sequence", []).append(float(resource_metrics[metric_key]))

    def _rollout_trace_record(self, session_id: str, session: dict, outcome: dict) -> dict:
        record = {"session_id": session_id}
        for key, value in outcome.items():
            if key != "session_id":
                record[key] = self._jsonable(value)

        trace = session.get("rollout_trace", {})
        for key in [
            "router_probs_sequence",
            "active_mask_sequence",
            "pool_mask_sequence",
            "forced_expert_id_sequence",
            "latency_ms_sequence",
            "vram_mb_sequence",
        ]:
            sequence = self._sequence_from_steps(trace.get(key, []))
            if sequence is not None:
                record[key] = sequence
        router_sequence = record.get("router_probs_sequence")
        route_sequence_length = self._route_sequence_length(router_sequence)
        if route_sequence_length is not None:
            record["num_route_chunks"] = route_sequence_length
        if "resident_fraction" not in record and "resident_fraction_requested" not in record:
            resident_fraction = self._resident_fraction_from_pool_sequence(record.get("pool_mask_sequence"))
            if resident_fraction is not None:
                record["resident_fraction"] = resident_fraction
        if "latency_ms" not in record and record.get("latency_ms_sequence"):
            latency_sequence = record["latency_ms_sequence"]
            record["latency_ms"] = float(sum(latency_sequence) / len(latency_sequence))
        if "vram_mb" not in record and record.get("vram_mb_sequence"):
            record["vram_mb"] = float(max(record["vram_mb_sequence"]))
        return record

    def _write_rollout_trace(self, session_id: str, session: dict, outcome: dict) -> bool:
        if self._rollout_trace_path is None:
            return False
        if outcome.get("discard_trace"):
            return False
        record = self._rollout_trace_record(session_id, session, outcome)
        has_trace = any(key.endswith("_sequence") for key in record)
        has_outcome = any(key in record for key in ["success", "success_rate", "return_score", "return"])
        if not has_trace and not has_outcome:
            return False
        self._rollout_trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self._rollout_trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    # Example usage:
    # policy = YourPolicyClass()  # Replace with your actual policy class
    # server = WebsocketPolicyServer(policy, host="localhost", port=10091)
    # server.serve_forever()
    raise NotImplementedError("This module is not intended to be run directly.")
#
#  Instead, it should be imported and used in a server context.
