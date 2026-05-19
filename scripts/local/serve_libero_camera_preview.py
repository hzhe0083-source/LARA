#!/usr/bin/env python3
"""Serve a live LIBERO camera preview over HTTP."""

from __future__ import annotations

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np


LATEST_JPEG: bytes | None = None
LATEST_LOCK = threading.Lock()
STOP = threading.Event()


def _configure_assets() -> None:
    from libero.libero import get_libero_path
    import libero.libero as libero_module

    assets_path = Path(get_libero_path("assets"))
    if assets_path.exists():
        libero_module._assets_path_cache = str(assets_path)


def _make_env(task_suite: str, task_id: int, resolution: int, seed: int):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    _configure_assets()
    bench = benchmark.get_benchmark_dict()[task_suite]()
    task = bench.get_task(task_id)
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=resolution,
        camera_widths=resolution,
        control_freq=30,
    )
    env.seed(seed)
    return env, task.language


def _render_loop(args: argparse.Namespace) -> None:
    global LATEST_JPEG
    env = None
    try:
        env, language = _make_env(args.task_suite, args.task_id, args.resolution, args.seed)
        print(f"Previewing {args.task_suite} task {args.task_id}: {language}", flush=True)
        env.reset()
        action = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)
        delay = 1.0 / max(args.fps, 1.0)
        frame = 0
        while not STOP.is_set():
            obs, _, _, _ = env.step(action)
            image = obs.get(args.camera)
            if image is None:
                image = obs.get("agentview_image")
            if image is None:
                image = next(v for k, v in obs.items() if k.endswith("_image"))
            image = np.asarray(image)
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if ok:
                with LATEST_LOCK:
                    LATEST_JPEG = encoded.tobytes()
            frame += 1
            if frame % int(max(args.fps, 1.0)) == 0:
                print(f"preview_frames={frame}", flush=True)
            time.sleep(delay)
    except Exception as exc:
        print(f"preview_error={type(exc).__name__}: {exc}", flush=True)
        import traceback

        traceback.print_exc()
    finally:
        if env is not None:
            env.close()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        return

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            body = b"""<!doctype html>
<html><head><title>LIBERO Preview</title>
<style>body{margin:0;background:#111;color:#eee;font-family:sans-serif;display:grid;place-items:center;height:100vh}img{max-width:96vw;max-height:96vh;image-rendering:auto}</style>
</head><body><img id="frame" src="/frame.jpg"><script>
setInterval(()=>{document.getElementById('frame').src='/frame.jpg?t='+Date.now()},100);
</script></body></html>"""
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/frame.jpg"):
            with LATEST_LOCK:
                frame = LATEST_JPEG
            if frame is None:
                self.send_response(503)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(frame)))
            self.end_headers()
            self.wfile.write(frame)
            return
        self.send_response(404)
        self.end_headers()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_suite", "--task-suite", default="libero_10")
    parser.add_argument("--task_id", "--task-id", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--camera", default="agentview_image")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    thread = threading.Thread(target=_render_loop, args=(args,), daemon=True)
    thread.start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Open http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        STOP.set()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
