#!/usr/bin/env python3
"""Open a native MuJoCo viewer for one LIBERO task."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def _configure_assets() -> None:
    from libero.libero import get_libero_path
    import libero.libero as libero_module

    assets_path = Path(get_libero_path("assets"))
    if assets_path.exists():
        libero_module._assets_path_cache = str(assets_path)


def _task(task_suite: str, task_id: int):
    from libero.libero import benchmark

    bench = benchmark.get_benchmark_dict()[task_suite]()
    return bench.get_task(task_id)


def _make_env(task, seed: int):
    from libero.libero import get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv

    _configure_assets()
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = ControlEnv(
        bddl_file_name=str(bddl_file),
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        control_freq=30,
        horizon=1000,
    )
    env.seed(seed)
    return env


def _set_fixed_camera(viewer, model, camera_name: str) -> None:
    import mujoco

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id >= 0:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = camera_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_suite", "--task-suite", default="libero_10")
    parser.add_argument("--task_id", "--task-id", type=int, default=0)
    parser.add_argument("--camera", default="frontview")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seconds", type=float, default=600.0)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument(
        "--no_force_x11",
        "--no-force-x11",
        action="store_true",
        help="Do not unset WAYLAND_DISPLAY before launching GLFW.",
    )
    parser.add_argument(
        "--no_force_exit",
        "--no-force-exit",
        action="store_true",
        help="Return through normal Python shutdown instead of os._exit(0).",
    )
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "glfw")
    if not args.no_force_x11:
        os.environ.pop("WAYLAND_DISPLAY", None)
        os.environ["XDG_SESSION_TYPE"] = "x11"

    import mujoco.viewer

    task = _task(args.task_suite, args.task_id)
    print(f"Opening {args.task_suite} task {args.task_id}: {task.language}", flush=True)
    env = _make_env(task, args.seed)
    env.reset()

    sim = env.env.sim
    model = sim.model._model
    data = sim.data._data
    action = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)
    delay = 1.0 / max(args.fps, 1.0)
    deadline = time.time() + args.seconds
    exit_code = 0
    viewer = None
    try:
        viewer = mujoco.viewer.launch_passive(model, data)
        _set_fixed_camera(viewer, model, args.camera)
        frame = 0
        print("native_mujoco_viewer_started", flush=True)
        while time.time() < deadline and viewer.is_running():
            env.step(action)
            viewer.sync()
            frame += 1
            if frame % int(max(args.fps, 1.0)) == 0:
                print(f"rendered_frames={frame}", flush=True)
            time.sleep(delay)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        exit_code = 1
        print(f"viewer_error={type(exc).__name__}: {exc}", flush=True)
        raise
    finally:
        print("closing_viewer", flush=True)
        if viewer is not None:
            viewer.close()
        env.close()
        sys.stdout.flush()
        sys.stderr.flush()
        if not args.no_force_exit:
            os._exit(exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
