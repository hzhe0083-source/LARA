#!/usr/bin/env python3
"""Run headless LIBERO evaluation for LARA checkpoints.

The evaluator talks to the existing websocket policy server. It can either use
an already-running server or launch `deployment/model_server/server_policy.py`
as a child process. Rollout records are compact by default: every episode gets
a summary row, while full route sequences are preserved only for sampled
episodes to control disk use.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
DEFAULT_LEROBOT_TASKS_PATH = (
    REPO_ROOT / "data/libero100/kevin_libero100_lerobot/meta/tasks.parquet"
)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _libero_gripper_command(gripper_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(gripper_val, dtype=np.float32).reshape(-1)
    value = float(arr[0])
    return np.asarray([-1.0 if value <= 0.0 else 1.0], dtype=np.float32)


def _action_replan_period(action_chunk_size: int, replan_every: int | None) -> int:
    if replan_every is None:
        return action_chunk_size
    if replan_every <= 0:
        raise ValueError(f"replan_every must be positive, got {replan_every}")
    return min(int(replan_every), int(action_chunk_size))


def _transform_libero_action(
    raw_action_row: np.ndarray,
    *,
    xyz_scale: float,
    rot_scale: float,
    invert_x: bool,
    invert_y: bool,
    invert_z: bool,
    invert_rx: bool,
    invert_ry: bool,
    invert_rz: bool,
) -> np.ndarray:
    action = np.asarray(raw_action_row, dtype=np.float32).copy()
    action[:3] *= float(xyz_scale)
    action[3:6] *= float(rot_scale)
    if invert_x:
        action[0] *= -1.0
    if invert_y:
        action[1] *= -1.0
    if invert_z:
        action[2] *= -1.0
    if invert_rx:
        action[3] *= -1.0
    if invert_ry:
        action[4] *= -1.0
    if invert_rz:
        action[5] *= -1.0
    return action


def _libero_observation_state(obs: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(obs["robot0_joint_pos"], dtype=np.float32),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        )
    )[None]


def _resize_image(image: np.ndarray, image_size: list[int]) -> np.ndarray:
    import cv2 as cv

    return cv.resize(image, tuple(image_size), interpolation=cv.INTER_AREA)


def _policy_image(image: np.ndarray) -> np.ndarray:
    # MuJoCo's offscreen image is vertically flipped relative to the LeRobot frames.
    return np.ascontiguousarray(np.asarray(image)[::-1, :])


def _short_name(text: str, max_len: int = 80) -> str:
    import hashlib

    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    clean = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)[:max_len]
    return f"{clean}_{digest}"


def _load_lerobot_task_texts(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"LeRobot task metadata not found: {path}")

    def split_scene(text: str) -> tuple[str | None, str]:
        if ":" not in text:
            return None, text.strip()
        scene, instruction = text.split(":", 1)
        return scene.strip(), instruction.strip()

    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        rows = table.to_pylist()
        columns = table.column_names
        text_column = next(
            (candidate for candidate in ("task", "instruction", "__index_level_0__") if candidate in columns),
            None,
        )
    else:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        text_column = None

    task_texts: dict[str, dict[str, str]] = {}
    for row in rows:
        task_index = row.get("task_index")
        if task_index is None:
            continue
        raw_text = row.get(text_column) if text_column is not None else row.get("task", row.get("instruction"))
        if raw_text is None:
            raw_text = row.get("__index_level_0__")
        if raw_text is None:
            continue
        full_text = str(raw_text).strip()
        scene, instruction = split_scene(full_text)
        task_texts[str(int(task_index))] = {
            "full": full_text,
            "instruction": instruction,
            "scene": scene or "",
        }
    return task_texts


def _task_description_for_policy(
    *,
    suite_name: str,
    task_id: int,
    libero_language: str,
    problem_folder: str | None = None,
    lerobot_task_texts: dict[str, dict[str, str]],
    use_lerobot_task_text: bool,
) -> str:
    if not use_lerobot_task_text:
        return libero_language
    scene = Path(str(problem_folder)).name if problem_folder else None
    matches = [row for row in lerobot_task_texts.values() if row["instruction"] == libero_language]
    if scene:
        scene_matches = [row for row in matches if row["scene"] == scene]
        if len(scene_matches) == 1:
            return scene_matches[0]["full"]
    if len(matches) == 1:
        return matches[0]["full"]
    if len(matches) > 1:
        raise KeyError(
            f"Ambiguous LeRobot task text for {suite_name}:{task_id} "
            f"scene={scene!r} instruction={libero_language!r}"
        )
    raise KeyError(f"Could not map LIBERO task {suite_name}:{task_id} to LeRobot task text: {libero_language!r}")


def _max_steps(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 250
    if task_suite_name == "libero_object":
        return 280
    if task_suite_name == "libero_goal":
        return 300
    if task_suite_name in {"libero_10", "libero_mix"}:
        return 520
    if task_suite_name in {"libero_90", "libero_100"}:
        return 400
    raise ValueError(f"Unknown task suite: {task_suite_name}")


def _get_libero_env(task, resolution: int, seed: int):
    import libero.libero as libero_module
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    assets_path = Path(get_libero_path("assets"))
    if assets_path.exists():
        # The pip package's get_assets_path() does not read LIBERO_CONFIG_PATH.
        # Point its cache at the configured assets so arena XML paths resolve.
        libero_module._assets_path_cache = str(assets_path)

    env_args = {
        "bddl_file_name": str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
        "control_freq": 30,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task.language


def _load_benchmark(task_suite_name: str, category_value: str):
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    if task_suite_name == "libero_mix":
        return benchmark_dict[task_suite_name](category_value=category_value)
    return benchmark_dict[task_suite_name]()


def _suite_plan(task_suite_name: str, category_value: str) -> list[tuple[str, Any]]:
    if task_suite_name == "libero_100":
        return [
            ("libero_90", _load_benchmark("libero_90", category_value)),
            ("libero_10", _load_benchmark("libero_10", category_value)),
        ]
    return [(task_suite_name, _load_benchmark(task_suite_name, category_value))]


def _task_ids(value: str | None, num_tasks: int) -> list[int]:
    if not value:
        return list(range(num_tasks))
    result = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(part))
    for task_id in result:
        if task_id < 0 or task_id >= num_tasks:
            raise ValueError(f"task id {task_id} out of range [0, {num_tasks})")
    return result


def _suite_task_ids(value: str | None, suites: list[tuple[str, Any]]) -> dict[str, list[int]]:
    if not value:
        return {suite_name: list(range(task_suite.n_tasks)) for suite_name, task_suite in suites}
    if len(suites) == 1:
        suite_name, task_suite = suites[0]
        return {suite_name: _task_ids(value, task_suite.n_tasks)}
    selected: dict[str, list[int]] = {}
    current_suite_name: str | None = None
    suite_lookup = dict(suites)
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            suite_name, suite_ids = part.split(":", 1)
            current_suite_name = suite_name
        elif current_suite_name is not None:
            suite_name, suite_ids = current_suite_name, part
        else:
            raise ValueError(
                "multi-suite task ids must be prefixed, e.g. libero_90:0,libero_10:0-2"
            )
        if suite_name not in suite_lookup:
            raise ValueError(f"unknown suite in --task_ids: {suite_name}")
        selected.setdefault(suite_name, []).extend(_task_ids(suite_ids, suite_lookup[suite_name].n_tasks))
    return selected


def _write_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [float(record.get("success", 0.0)) for record in records]
    returns = [float(record.get("return_score", record.get("return", 0.0))) for record in records]
    lengths = [float(record.get("episode_length", 0.0)) for record in records]
    latencies = [
        float(record["latency_ms"])
        for record in records
        if record.get("latency_ms") is not None
    ]
    vram = [float(record["vram_mb"]) for record in records if record.get("vram_mb") is not None]
    by_task: dict[str, dict[str, Any]] = {}
    for record in records:
        task_id = str(record["task_id"])
        bucket = by_task.setdefault(task_id, {"episodes": 0, "successes": 0})
        bucket["episodes"] += 1
        bucket["successes"] += int(bool(record.get("success", False)))
        bucket["success_rate"] = bucket["successes"] / bucket["episodes"]
    return {
        "episodes": len(records),
        "successes": int(sum(successes)),
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "return_mean": float(np.mean(returns)) if returns else 0.0,
        "episode_length_mean": float(np.mean(lengths)) if lengths else 0.0,
        "latency_ms_mean": float(np.mean(latencies)) if latencies else None,
        "vram_mb_max": float(np.max(vram)) if vram else None,
        "by_task": by_task,
    }


class RouteClient:
    def __init__(
        self,
        host: str,
        port: int,
        checkpoint: Path,
        image_size: list[int],
        with_state: bool,
        *,
        replan_every: int | None,
        xyz_scale: float,
        rot_scale: float,
        invert_x: bool,
        invert_y: bool,
        invert_z: bool,
        invert_rx: bool,
        invert_ry: bool,
        invert_rz: bool,
        sticky_gripper_steps: int,
        gripper_lookahead_steps: int,
    ):
        from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
        from examples.LIBERO.model2libero_interface import M1Inference

        self.policy = WebsocketClientPolicy(host=host, port=port)
        # Reuse stats/chunking helpers from the existing interface without
        # constructing its websocket client; this evaluator sends typed
        # session-aware messages directly so route traces stay episode-scoped.
        self.unnorm_key = None
        self.use_ddim = True
        self.num_ddim_steps = 10
        self.image_size = image_size
        self.action_norm_stats = M1Inference.get_action_stats(self.unnorm_key, policy_ckpt_path=checkpoint)
        self.action_chunk_size = M1Inference.get_action_chunk_size(policy_ckpt_path=checkpoint)
        self.replan_period = _action_replan_period(self.action_chunk_size, replan_every)
        self.unnormalize_actions = M1Inference.unnormalize_actions
        self.with_state = with_state
        self.xyz_scale = xyz_scale
        self.rot_scale = rot_scale
        self.invert_x = invert_x
        self.invert_y = invert_y
        self.invert_z = invert_z
        self.invert_rx = invert_rx
        self.invert_ry = invert_ry
        self.invert_rz = invert_rz
        if sticky_gripper_steps < 0:
            raise ValueError(f"sticky_gripper_steps must be non-negative, got {sticky_gripper_steps}")
        if gripper_lookahead_steps < 0:
            raise ValueError(f"gripper_lookahead_steps must be non-negative, got {gripper_lookahead_steps}")
        self.sticky_gripper_steps = int(sticky_gripper_steps)
        self.gripper_lookahead_steps = int(gripper_lookahead_steps)
        self.sticky_gripper_remaining = 0
        self.task_description = None
        self.raw_actions = None
        self.last_latency_ms = None
        self.last_vram_mb = None

    def _send_control(self, message: dict[str, Any]) -> dict[str, Any]:
        from deployment.model_server.tools import msgpack_numpy

        self.policy._ws.send(self.policy._packer.pack(message))
        response = self.policy._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(response)
        payload = msgpack_numpy.unpackb(response)
        if isinstance(payload, dict) and not payload.get("ok", True):
            raise RuntimeError(payload)
        return payload

    def reset(self, task_description: str, session_id: str, *, write_previous_trace: bool = False) -> None:
        self.task_description = task_description
        self.raw_actions = None
        self.sticky_gripper_remaining = 0
        self.last_latency_ms = None
        self.last_vram_mb = None
        if write_previous_trace:
            self._send_control(
                {
                    "type": "reset",
                    "session_id": session_id,
                    "payload": {"task_description": task_description},
                }
            )

    def record_outcome(self, session_id: str, payload: dict[str, Any]) -> None:
        self._send_control({"type": "record_outcome", "session_id": session_id, "payload": payload})

    def clear_session(self, session_id: str) -> None:
        self._send_control({"type": "reset", "session_id": session_id, "payload": {"discard_trace": True}})

    def step(
        self,
        *,
        images: list[np.ndarray],
        state: np.ndarray | None,
        step: int,
        session_id: str,
        forced_expert_id: int | None = None,
    ) -> dict[str, Any]:
        images = [_resize_image(image, self.image_size) for image in images]
        if step % self.replan_period == 0 or self.raw_actions is None:
            payload: dict[str, Any] = {
                "batch_images": [images],
                "instructions": [self.task_description],
                "unnorm_key": self.unnorm_key,
                "do_sample": False,
                "use_ddim": self.use_ddim,
                "num_ddim_steps": self.num_ddim_steps,
                "session_id": session_id,
            }
            if self.with_state and state is not None:
                payload["state"] = [state]
            if forced_expert_id is not None:
                payload["forced_expert_id"] = forced_expert_id
            response = self.policy.infer(payload)
            self.last_latency_ms = response["data"].get("latency_ms")
            self.last_vram_mb = response["data"].get("vram_mb")
            normalized_actions = np.asarray(response["data"]["normalized_actions"], dtype=np.float32)[0]
            self.raw_actions = self.unnormalize_actions(
                normalized_actions=normalized_actions,
                action_norm_stats=self.action_norm_stats,
            )
        raw_action_row = self.raw_actions[step % self.replan_period]
        transformed_action_row = _transform_libero_action(
            raw_action_row,
            xyz_scale=self.xyz_scale,
            rot_scale=self.rot_scale,
            invert_x=self.invert_x,
            invert_y=self.invert_y,
            invert_z=self.invert_z,
            invert_rx=self.invert_rx,
            invert_ry=self.invert_ry,
            invert_rz=self.invert_rz,
        )
        raw_action = {
            "world_vector": np.array(transformed_action_row[:3]),
            "rotation_delta": np.array(transformed_action_row[3:6]),
            "open_gripper": np.array(transformed_action_row[6:7]),
        }
        gripper = _libero_gripper_command(raw_action["open_gripper"])
        if float(gripper[0]) <= 0.0 and self.gripper_lookahead_steps > 0:
            lookahead_end = min(
                len(self.raw_actions),
                step % self.replan_period + self.gripper_lookahead_steps + 1,
            )
            future_gripper = self.raw_actions[step % self.replan_period + 1 : lookahead_end, 6]
            if future_gripper.size and bool(np.any(future_gripper > 0.0)):
                raw_action["open_gripper"] = np.asarray([1.0], dtype=np.float32)
                gripper = np.asarray([1.0], dtype=np.float32)
        if float(gripper[0]) > 0.0 and self.sticky_gripper_steps > 0:
            self.sticky_gripper_remaining = self.sticky_gripper_steps
        elif self.sticky_gripper_remaining > 0:
            raw_action["open_gripper"] = np.asarray([1.0], dtype=np.float32)
            self.sticky_gripper_remaining -= 1
        return {
            "raw_action": raw_action,
            "raw_actions": transformed_action_row[None],
            "raw_model_action": np.asarray(raw_action_row, dtype=np.float32)[None],
        }


def maybe_start_server(args: argparse.Namespace, trace_path: Path) -> subprocess.Popen | None:
    if not args.start_server:
        return None
    cmd = [
        sys.executable,
        str(REPO_ROOT / "deployment/model_server/server_policy.py"),
        "--ckpt_path",
        str(args.checkpoint),
        "--port",
        str(args.port),
        "--cuda",
        str(args.cuda),
        "--rollout_trace_path",
        str(trace_path),
    ]
    if args.use_bf16:
        cmd.append("--use_bf16")
    log_path = args.output_dir / "policy_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8", buffering=1)
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    time.sleep(args.server_startup_delay)
    return proc


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.video_out_dir is not None:
        args.video_out_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "rollout_records.jsonl"
    summary_path = args.output_dir / "eval_summary.json"
    full_trace_path = args.output_dir / "sampled_route_traces.jsonl"
    server_trace_path = (
        full_trace_path
        if args.sample_full_route_every > 0
        else args.output_dir / "server_route_trace.tmp.jsonl"
    )

    server_proc = maybe_start_server(args, server_trace_path)
    try:
        suite_plan = _suite_plan(args.task_suite_name, args.category_value)
        selected_task_ids_by_suite = _suite_task_ids(args.task_ids, suite_plan)
        lerobot_task_texts = _load_lerobot_task_texts(
            args.lerobot_tasks_path if args.use_lerobot_task_text else None
        )
        client = RouteClient(
            host=args.host,
            port=args.port,
            checkpoint=args.checkpoint,
            image_size=args.resize_size,
            with_state=args.with_state,
            replan_every=args.replan_every,
            xyz_scale=args.xyz_scale,
            rot_scale=args.rot_scale,
            invert_x=args.invert_x,
            invert_y=args.invert_y,
            invert_z=args.invert_z,
            invert_rx=args.invert_rx,
            invert_ry=args.invert_ry,
            invert_rz=args.invert_rz,
            sticky_gripper_steps=args.sticky_gripper_steps,
            gripper_lookahead_steps=args.gripper_lookahead_steps,
        )
        rng = np.random.default_rng(args.seed)
        records: list[dict[str, Any]] = []

        for suite_name, task_suite in suite_plan:
            max_steps = _max_steps(suite_name)
            for task_id in selected_task_ids_by_suite.get(suite_name, []):
                task = task_suite.get_task(task_id)
                initial_states = task_suite.get_task_init_states(task_id)
                env, libero_task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
                task_description = _task_description_for_policy(
                    suite_name=suite_name,
                    task_id=task_id,
                    libero_language=libero_task_description,
                    problem_folder=getattr(task, "problem_folder", None),
                    lerobot_task_texts=lerobot_task_texts,
                    use_lerobot_task_text=args.use_lerobot_task_text,
                )
                for episode_idx in range(args.num_trials_per_task):
                    session_id = f"{suite_name}_task{task_id}_episode{episode_idx}_{int(time.time() * 1000)}"
                    client.reset(task_description=task_description, session_id=session_id)
                    env.reset()
                    obs = env.set_init_state(initial_states[episode_idx])
                    done = False
                    return_score = 0.0
                    step = 0
                    t = 0
                    forced_expert_id = args.forced_expert_id
                    if args.random_forced_expert and args.num_experts:
                        forced_expert_id = int(rng.integers(0, args.num_experts))
                    start_time = time.perf_counter()
                    latency_ms_values = []
                    vram_mb_values = []
                    replay_images = []
                    while t < max_steps + args.num_steps_wait:
                        if t < args.num_steps_wait:
                            obs, reward, done, _info = env.step(LIBERO_DUMMY_ACTION)
                            t += 1
                            continue
                        img = _policy_image(obs["agentview_image"])
                        wrist_img = _policy_image(obs["robot0_eye_in_hand_image"])
                        if args.video_out_dir is not None:
                            replay_images.append(img)
                        state = _libero_observation_state(obs)
                        response = client.step(
                            images=[img, wrist_img],
                            state=state,
                            step=step,
                            session_id=session_id,
                            forced_expert_id=forced_expert_id,
                        )
                        if client.last_latency_ms is not None:
                            latency_ms_values.append(float(client.last_latency_ms))
                        if client.last_vram_mb is not None:
                            vram_mb_values.append(float(client.last_vram_mb))
                        raw_action = response["raw_action"]
                        world_vector_delta = np.asarray(raw_action["world_vector"], dtype=np.float32).reshape(-1)
                        rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float32).reshape(-1)
                        gripper = _libero_gripper_command(raw_action["open_gripper"])
                        action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)
                        obs, reward, done, _info = env.step(action.tolist())
                        return_score += float(reward)
                        step += 1
                        t += 1
                        if done:
                            break
                    elapsed = time.perf_counter() - start_time
                    keep_full_trace = (
                        args.sample_full_route_every > 0
                        and len(records) % args.sample_full_route_every == 0
                    )
                    outcome = {
                        "task_suite": suite_name,
                        "requested_task_suite": args.task_suite_name,
                        "task_id": task_id,
                        "episode_idx": episode_idx,
                        "task_description": task_description,
                        "libero_task_description": libero_task_description,
                        "success": bool(done),
                        "return_score": return_score,
                        "episode_length": step,
                        "wall_time_s": elapsed,
                        "latency_ms": float(np.mean(latency_ms_values)) if latency_ms_values else None,
                        "vram_mb": float(np.max(vram_mb_values)) if vram_mb_values else None,
                        "forced_expert_id": forced_expert_id,
                        "sampled_full_trace": keep_full_trace,
                    }
                    if keep_full_trace:
                        client.record_outcome(session_id, outcome)
                    else:
                        # Reset without outcome clears server-side trace; compact
                        # outcome still lands in rollout_records.jsonl below.
                        client.clear_session(session_id)
                    _write_jsonl(records_path, outcome)
                    if args.video_out_dir is not None and replay_images:
                        import imageio

                        suffix = "success" if done else "failure"
                        video_name = (
                            f"{suite_name}_task{task_id}_episode{episode_idx}_{suffix}_"
                            f"{_short_name(task_description)}.mp4"
                        )
                        imageio.mimwrite(
                            args.video_out_dir / video_name,
                            [np.asarray(image) for image in replay_images],
                            fps=10,
                        )
                    records.append(outcome)
                    logging.info(
                        "suite=%s task=%s episode=%s success=%s return=%.3f length=%s",
                        suite_name,
                        task_id,
                        episode_idx,
                        done,
                        return_score,
                        step,
                    )
                env.close()

        summary = _summarize_records(records)
        summary.update(
            {
                "checkpoint": str(args.checkpoint),
                "task_suite_name": args.task_suite_name,
                "suite_plan": [suite_name for suite_name, _ in suite_plan],
                "task_ids": selected_task_ids_by_suite,
                "action_chunk_size": client.action_chunk_size,
                "replan_every": client.replan_period,
                "xyz_scale": args.xyz_scale,
                "rot_scale": args.rot_scale,
                "invert_x": args.invert_x,
                "invert_y": args.invert_y,
                "invert_z": args.invert_z,
                "invert_rx": args.invert_rx,
                "invert_ry": args.invert_ry,
                "invert_rz": args.invert_rz,
                "sticky_gripper_steps": args.sticky_gripper_steps,
                "gripper_lookahead_steps": args.gripper_lookahead_steps,
                "rollout_records": str(records_path),
                "sampled_route_traces": str(full_trace_path) if full_trace_path.exists() else None,
            }
        )
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary
    finally:
        if server_proc is not None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server_proc.kill()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output_dir", "--output-dir", required=True, type=Path)
    parser.add_argument("--video_out_dir", "--video-out-dir", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--start_server", "--start-server", action="store_true")
    parser.add_argument("--cuda", default=0)
    parser.add_argument("--use_bf16", "--use-bf16", action="store_true")
    parser.add_argument("--server_startup_delay", "--server-startup-delay", type=float, default=20.0)
    parser.add_argument(
        "--task_suite_name",
        "--task-suite-name",
        default="libero_100",
        help="Use libero_100 to evaluate libero_90 followed by libero_10.",
    )
    parser.add_argument("--category_value", "--category-value", default="Background Textures")
    parser.add_argument("--task_ids", "--task-ids", help="Comma/range task ids, e.g. 0,1,3-5. Defaults to all.")
    parser.add_argument("--num_trials_per_task", "--num-trials-per-task", type=int, default=10)
    parser.add_argument("--num_steps_wait", "--num-steps-wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resize_size", "--resize-size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--with_state", "--with-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--replan_every",
        "--replan-every",
        type=int,
        help="Request a fresh policy chunk every N executed steps. Defaults to the checkpoint chunk size.",
    )
    parser.add_argument("--xyz_scale", "--xyz-scale", type=float, default=1.5)
    parser.add_argument("--rot_scale", "--rot-scale", type=float, default=1.5)
    parser.add_argument("--invert_x", "--invert-x", action="store_true")
    parser.add_argument("--invert_y", "--invert-y", action="store_true")
    parser.add_argument("--invert_z", "--invert-z", action="store_true")
    parser.add_argument("--invert_rx", "--invert-rx", action="store_true")
    parser.add_argument("--invert_ry", "--invert-ry", action="store_true")
    parser.add_argument("--invert_rz", "--invert-rz", action="store_true")
    parser.add_argument(
        "--sticky_gripper_steps",
        "--sticky-gripper-steps",
        type=int,
        default=0,
        help="After a close command is predicted, keep sending close for this many additional env steps.",
    )
    parser.add_argument(
        "--gripper_lookahead_steps",
        "--gripper-lookahead-steps",
        type=int,
        default=0,
        help="If a future step in the current chunk predicts close, start closing this many steps early.",
    )
    parser.add_argument("--sample_full_route_every", "--sample-full-route-every", type=int, default=10)
    parser.add_argument("--forced_expert_id", "--forced-expert-id", type=int)
    parser.add_argument("--random_forced_expert", "--random-forced-expert", action="store_true")
    parser.add_argument("--num_experts", "--num-experts", type=int)
    parser.add_argument(
        "--use_lerobot_task_text",
        "--use-lerobot-task-text",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the full LeRobot metadata task text, including scene prefix when available, as the policy instruction.",
    )
    parser.add_argument(
        "--lerobot_tasks_path",
        "--lerobot-tasks-path",
        type=Path,
        default=DEFAULT_LEROBOT_TASKS_PATH,
        help="Path to LeRobot meta/tasks.parquet or tasks.jsonl used for task text mapping.",
    )
    return parser.parse_args(argv)


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    summary = run_eval(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
