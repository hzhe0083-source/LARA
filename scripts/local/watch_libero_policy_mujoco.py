#!/usr/bin/env python3
"""Watch a checkpoint act in LIBERO through the native MuJoCo viewer."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
DEFAULT_LEROBOT_TASKS_PATH = (
    REPO_ROOT / "data/libero100/kevin_libero100_lerobot/meta/tasks.parquet"
)


def _configure_assets() -> None:
    import libero.libero as libero_module
    from libero.libero import get_libero_path

    assets_path = Path(get_libero_path("assets"))
    if assets_path.exists():
        libero_module._assets_path_cache = str(assets_path)


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


def _json_array(value: Any) -> list[float]:
    return np.asarray(value, dtype=np.float32).reshape(-1).tolist()


def _obs_trace(obs: dict[str, Any]) -> dict[str, Any]:
    object_positions = {
        key: _json_array(value)
        for key, value in obs.items()
        if key.endswith("_pos") and not key.startswith("robot0_")
    }
    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)
    object_distances = {
        key.removesuffix("_pos"): float(np.linalg.norm(eef_pos - np.asarray(value, dtype=np.float32).reshape(-1)))
        for key, value in object_positions.items()
    }
    return {
        "eef_pos": eef_pos.tolist(),
        "eef_quat": _json_array(obs["robot0_eef_quat"]),
        "joint_pos": _json_array(obs["robot0_joint_pos"]),
        "gripper_qpos": _json_array(obs["robot0_gripper_qpos"]),
        "object_positions": object_positions,
        "eef_to_object_distances": object_distances,
    }


def _resize_image(image: np.ndarray, image_size: list[int]) -> np.ndarray:
    import cv2 as cv

    return cv.resize(image, tuple(image_size), interpolation=cv.INTER_AREA)


def _policy_image(image: np.ndarray) -> np.ndarray:
    # MuJoCo's offscreen image is vertically flipped relative to the LeRobot frames.
    return np.ascontiguousarray(np.asarray(image)[::-1, :])


def _make_env(task: Any, seed: int):
    from libero.libero import get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv

    _configure_assets()
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = ControlEnv(
        bddl_file_name=str(bddl_file),
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
        control_freq=30,
        horizon=1000,
    )
    env.seed(seed)
    return env


def _set_fixed_camera(viewer: Any, model: Any, camera_name: str) -> None:
    import mujoco

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id >= 0:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = camera_id


def _suite(task_suite_name: str, category_value: str):
    from scripts.eval_libero100_headless import _suite_plan

    suite_plan = _suite_plan(task_suite_name, category_value)
    if len(suite_plan) != 1:
        raise ValueError("watch viewer expects one suite; pass libero_10/libero_90/etc., not libero_100")
    return suite_plan[0]


def _max_steps(task_suite_name: str) -> int:
    from scripts.eval_libero100_headless import _max_steps as max_steps

    return max_steps(task_suite_name)


def _start_server(args: argparse.Namespace, trace_path: Path) -> subprocess.Popen:
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


class LiveRouteClient:
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
        self.last_raw_action_row = None
        self.last_transformed_action_row = None
        self.last_gripper_sticky_remaining = 0
        self.last_gripper_lookahead_triggered = False
        self.last_chunk_slot = None
        self.last_latency_ms = None
        self.last_vram_mb = None

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.raw_actions = None
        self.last_raw_action_row = None
        self.last_transformed_action_row = None
        self.sticky_gripper_remaining = 0
        self.last_gripper_sticky_remaining = 0
        self.last_gripper_lookahead_triggered = False
        self.last_chunk_slot = None
        self.last_latency_ms = None
        self.last_vram_mb = None

    def step(self, *, images: list[np.ndarray], state: np.ndarray | None, step: int, session_id: str) -> np.ndarray:
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
            response = self.policy.infer(payload)
            self.last_latency_ms = response["data"].get("latency_ms")
            self.last_vram_mb = response["data"].get("vram_mb")
            normalized_actions = np.asarray(response["data"]["normalized_actions"], dtype=np.float32)[0]
            self.raw_actions = self.unnormalize_actions(
                normalized_actions=normalized_actions,
                action_norm_stats=self.action_norm_stats,
            )

        chunk_slot = step % self.replan_period
        raw_action_row = self.raw_actions[chunk_slot]
        self.last_raw_action_row = np.asarray(raw_action_row, dtype=np.float32).copy()
        self.last_chunk_slot = int(chunk_slot)
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
        self.last_transformed_action_row = transformed_action_row.copy()
        gripper = _libero_gripper_command(transformed_action_row[6:7])
        lookahead_triggered = False
        if float(gripper[0]) <= 0.0 and self.gripper_lookahead_steps > 0:
            lookahead_end = min(
                len(self.raw_actions),
                chunk_slot + self.gripper_lookahead_steps + 1,
            )
            future_gripper = self.raw_actions[chunk_slot + 1 : lookahead_end, 6]
            if future_gripper.size and bool(np.any(future_gripper > 0.0)):
                gripper = np.asarray([1.0], dtype=np.float32)
                lookahead_triggered = True
        if float(gripper[0]) > 0.0 and self.sticky_gripper_steps > 0:
            self.sticky_gripper_remaining = self.sticky_gripper_steps
        elif self.sticky_gripper_remaining > 0:
            gripper = np.asarray([1.0], dtype=np.float32)
            self.sticky_gripper_remaining -= 1
        self.last_gripper_sticky_remaining = int(self.sticky_gripper_remaining)
        self.last_gripper_lookahead_triggered = lookahead_triggered
        return np.concatenate(
            [
                np.asarray(transformed_action_row[:3], dtype=np.float32),
                np.asarray(transformed_action_row[3:6], dtype=np.float32),
                gripper,
            ],
            axis=0,
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    import mujoco.viewer
    from scripts.eval_libero100_headless import (
        _load_lerobot_task_texts,
        _task_description_for_policy,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.output_dir / "sampled_route_traces.jsonl"
    step_trace_path = args.output_dir / "rollout_step_trace.jsonl"
    server_proc = _start_server(args, trace_path) if args.start_server else None
    env = None
    viewer = None
    step_trace = None
    try:
        if args.trace_steps:
            step_trace = step_trace_path.open("w", encoding="utf-8", buffering=1)
        suite_name, task_suite = _suite(args.task_suite_name, args.category_value)
        task = task_suite.get_task(args.task_id)
        initial_states = task_suite.get_task_init_states(args.task_id)
        libero_task_description = task.language
        lerobot_task_texts = _load_lerobot_task_texts(
            args.lerobot_tasks_path if args.use_lerobot_task_text else None
        )
        if args.task_description_override:
            task_description = args.task_description_override
        else:
            task_description = _task_description_for_policy(
                suite_name=suite_name,
                task_id=args.task_id,
                libero_language=libero_task_description,
                problem_folder=getattr(task, "problem_folder", None),
                lerobot_task_texts=lerobot_task_texts,
                use_lerobot_task_text=args.use_lerobot_task_text,
            )
        env = _make_env(task, args.seed)
        env.reset()
        obs = env.set_init_state(initial_states[args.episode_idx])
        sim = env.env.sim
        model = sim.model._model
        data = sim.data._data

        viewer = mujoco.viewer.launch_passive(model, data)
        _set_fixed_camera(viewer, model, args.camera)
        client = LiveRouteClient(
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
        session_id = f"{suite_name}_task{args.task_id}_live_{int(time.time() * 1000)}"
        client.reset(task_description)
        print(f"native_mujoco_viewer_started suite={suite_name} task={args.task_id}", flush=True)
        print(f"instruction={task_description}", flush=True)

        done = False
        return_score = 0.0
        step = 0
        t = 0
        max_steps = min(_max_steps(suite_name), args.max_steps)
        delay = 1.0 / max(args.fps, 1.0)
        start_time = time.perf_counter()
        latencies = []
        vram = []
        while viewer.is_running() and t < max_steps + args.num_steps_wait:
            loop_started = time.perf_counter()
            if t < args.num_steps_wait:
                obs, reward, done, _info = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                viewer.sync()
                time.sleep(max(delay - (time.perf_counter() - loop_started), 0.0))
                continue

            img = _policy_image(obs["agentview_image"])
            wrist_img = _policy_image(obs["robot0_eye_in_hand_image"])
            state = _libero_observation_state(obs)
            pre_obs_trace = _obs_trace(obs)
            action = client.step(images=[img, wrist_img], state=state, step=step, session_id=session_id)
            if client.last_latency_ms is not None:
                latencies.append(float(client.last_latency_ms))
            if client.last_vram_mb is not None:
                vram.append(float(client.last_vram_mb))
            obs, reward, done, _info = env.step(action.tolist())
            if step_trace is not None:
                step_trace.write(
                    json.dumps(
                        {
                            "step": step,
                            "t": t,
                            "chunk_slot": client.last_chunk_slot,
                            "pre": pre_obs_trace,
                            "post": _obs_trace(obs),
                            "state": _json_array(state),
                            "raw_model_action": _json_array(client.last_raw_action_row),
                            "transformed_model_action": _json_array(client.last_transformed_action_row),
                            "env_action": _json_array(action),
                            "gripper_sticky_remaining": client.last_gripper_sticky_remaining,
                            "gripper_lookahead_triggered": client.last_gripper_lookahead_triggered,
                            "reward": float(reward),
                            "done": bool(done),
                            "latency_ms": client.last_latency_ms,
                            "vram_mb": client.last_vram_mb,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            return_score += float(reward)
            step += 1
            t += 1
            viewer.sync()
            if step % args.print_every == 0 or done:
                print(
                    f"step={step} success={bool(done)} return={return_score:.3f} "
                    f"latency_ms={client.last_latency_ms} vram_mb={client.last_vram_mb}",
                    flush=True,
                )
            if done:
                break
            time.sleep(max(delay - (time.perf_counter() - loop_started), 0.0))

        outcome = {
            "checkpoint": str(args.checkpoint),
            "task_suite": suite_name,
            "task_id": args.task_id,
            "episode_idx": args.episode_idx,
            "task_description": task_description,
            "libero_task_description": libero_task_description,
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
            "success": bool(done),
            "return_score": return_score,
            "episode_length": step,
            "wall_time_s": time.perf_counter() - start_time,
            "latency_ms_mean": float(np.mean(latencies)) if latencies else None,
            "vram_mb_max": float(np.max(vram)) if vram else None,
        }
        (args.output_dir / "watch_summary.json").write_text(
            json.dumps(outcome, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(outcome, indent=2, sort_keys=True), flush=True)
        return outcome
    finally:
        if viewer is not None:
            viewer.close()
        if env is not None:
            env.close()
        if step_trace is not None:
            step_trace.close()
        if server_proc is not None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server_proc.kill()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output_dir", "--output-dir", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12000)
    parser.add_argument("--start_server", "--start-server", action="store_true")
    parser.add_argument("--cuda", default=0)
    parser.add_argument("--use_bf16", "--use-bf16", action="store_true")
    parser.add_argument("--server_startup_delay", "--server-startup-delay", type=float, default=60.0)
    parser.add_argument("--task_suite_name", "--task-suite-name", default="libero_10")
    parser.add_argument("--category_value", "--category-value", default="Background Textures")
    parser.add_argument("--task_id", "--task-id", type=int, default=0)
    parser.add_argument("--episode_idx", "--episode-idx", type=int, default=0)
    parser.add_argument("--num_steps_wait", "--num-steps-wait", type=int, default=10)
    parser.add_argument("--max_steps", "--max-steps", type=int, default=520)
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
    parser.add_argument("--camera", default="frontview")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--print_every", "--print-every", type=int, default=20)
    parser.add_argument("--trace_steps", "--trace-steps", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument(
        "--task_description_override",
        "--task-description-override",
        default=None,
        help="Explicit policy instruction text; bypasses automatic LIBERO/LeRobot task text mapping.",
    )
    return parser.parse_args()


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "glfw")
    os.environ.pop("WAYLAND_DISPLAY", None)
    os.environ["XDG_SESSION_TYPE"] = "x11"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    run(args)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
