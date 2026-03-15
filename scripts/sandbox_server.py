#!/usr/bin/env python3
"""Lightweight backend for sandbox_explorer.html.

Serves the static HTML and proxies sandbox operations to the ARL gateway.

API endpoints (all JSON):
  GET  /api/instances?dataset={r2egym|swebench}&q=<search>
  POST /api/session/start   {dataset, instance_id, gateway_url, experiment_id}
  POST /api/session/stop
  POST /api/session/exec    {cmd, timeout}
  GET  /api/session/status
  GET  /api/session/log?since=<ts>
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "swe"
HTML_FILE = ROOT / "sandbox_explorer.html"

# ── Instance loading ─────────────────────────────────────────────────────────


_instance_cache: dict[str, list[dict]] = {}
_instance_cache_lock = threading.Lock()


def _load_instances(dataset: str) -> list[dict]:
    """Load instance list from parquet. Cached after first call."""
    with _instance_cache_lock:
        if dataset in _instance_cache:
            return _instance_cache[dataset]

    if dataset == "r2egym":
        path = DATA_DIR / "R2E_Gym_Subset.parquet"
    elif dataset == "swebench":
        path = DATA_DIR / "SWE_Bench_Verified.parquet"
    else:
        return []

    if not path.exists():
        return []

    df = pd.read_parquet(path, columns=["extra_info"])
    extra = df["extra_info"]

    results = []
    if dataset == "r2egym":
        for ei_raw in extra:
            ei = json.loads(ei_raw) if isinstance(ei_raw, str) else ei_raw
            repo = ei.get("repo_name", "")
            commit = ei.get("commit_hash", "")[:8]
            results.append({
                "instance_id": f"{repo}__{commit}",
                "docker_image": ei.get("docker_image", ""),
            })
    else:
        for ei_raw in extra:
            ei = json.loads(ei_raw) if isinstance(ei_raw, str) else ei_raw
            results.append({
                "instance_id": ei.get("instance_id", ""),
                "docker_image": ei.get("docker_image", ei.get("image_name", "")),
            })

    # Deduplicate by instance_id
    seen = set()
    deduped = []
    for inst in results:
        if inst["instance_id"] not in seen:
            seen.add(inst["instance_id"])
            deduped.append(inst)

    with _instance_cache_lock:
        _instance_cache[dataset] = deduped
    return deduped


def search_instances(dataset: str, query: str) -> list[dict]:
    instances = _load_instances(dataset)
    if query:
        q = query.lower()
        instances = [i for i in instances if q in i["instance_id"].lower()]
    return instances[:200]


# ── Session state (single session) ───────────────────────────────────────────


class SessionState:
    def __init__(self):
        self.lock = threading.Lock()
        self.status = "idle"  # idle | starting | ready | error | closed
        self.status_msg = ""
        self.instance_id: str | None = None
        self.session = None  # ManagedSession
        self.cwd: str | None = None
        self.log: list[dict] = []  # [{type, text, ts}]

    def add_log(self, type_: str, text: str):
        self.log.append({"type": type_, "text": text, "ts": time.time()})

    def reset(self):
        self.status = "idle"
        self.status_msg = ""
        self.instance_id = None
        self.session = None
        self.cwd = None
        self.log.clear()


STATE = SessionState()


def _start_session(dataset: str, instance_id: str, gateway_url: str, experiment_id: str):
    """Background thread: create sandbox."""
    from arl import ManagedSession

    with STATE.lock:
        STATE.status = "starting"
        STATE.status_msg = f"Creating sandbox for {instance_id}..."
        STATE.instance_id = instance_id
    STATE.add_log("system", f"Starting sandbox for {instance_id}...")

    # Find the docker image for this instance
    instances = _load_instances(dataset)
    image = None
    for inst in instances:
        if inst["instance_id"] == instance_id:
            image = inst["docker_image"]
            break

    if not image:
        with STATE.lock:
            STATE.status = "error"
            STATE.status_msg = f"Instance {instance_id} not found"
        STATE.add_log("error", f"Instance {instance_id} not found in dataset")
        return

    try:
        session = ManagedSession(
            image=image,
            experiment_id=experiment_id,
            gateway_url=gateway_url,
            timeout=300,
        )
        session.create_sandbox()

        with STATE.lock:
            STATE.session = session
            STATE.status = "ready"
            STATE.status_msg = f"Connected to {instance_id}"
            STATE.cwd = None

        STATE.add_log("system", f"Sandbox ready (session_id={session.session_id})")

        # Get initial cwd
        try:
            resp = session.execute(steps=[{"name": "pwd", "command": ["pwd"]}])
            cwd = resp.results[0].output.stdout.strip()
            with STATE.lock:
                STATE.cwd = cwd
        except Exception:
            pass

    except Exception as e:
        with STATE.lock:
            STATE.status = "error"
            STATE.status_msg = f"Failed: {e}"
        STATE.add_log("error", f"Failed to create sandbox: {e}")


def _stop_session():
    """Stop and clean up the current session."""
    session = None
    with STATE.lock:
        session = STATE.session
        STATE.session = None

    if session:
        try:
            session.delete_sandbox()
            STATE.add_log("system", "Sandbox deleted.")
        except Exception as e:
            STATE.add_log("error", f"Delete failed: {e}")

    with STATE.lock:
        STATE.status = "idle"
        STATE.status_msg = ""
        STATE.instance_id = None
        STATE.cwd = None


def _exec_cmd(cmd: str, timeout: int) -> dict:
    """Execute a command in the sandbox. Returns {ok, cwd?, error?}."""
    with STATE.lock:
        session = STATE.session
        if not session or STATE.status != "ready":
            return {"ok": False, "error": "No active session"}

    STATE.add_log("system", f"$ {cmd}")

    try:
        # Wrap in bash -c for shell features; get cwd afterwards
        wrapped = f'{cmd}; echo "___CWD___"; pwd'
        resp = session.execute(
            steps=[{"name": "exec", "command": ["bash", "-c", wrapped], "timeout": timeout}]
        )
        result = resp.results[0]
        stdout = result.output.stdout or ""
        stderr = result.output.stderr or ""

        # Strip ANSI codes
        stdout = re.sub(r"\x1b\[[0-9;]*m|\r", "", stdout)
        stderr = re.sub(r"\x1b\[[0-9;]*m|\r", "", stderr)

        # Extract cwd from output
        cwd = None
        if "___CWD___" in stdout:
            parts = stdout.rsplit("___CWD___", 1)
            stdout = parts[0].rstrip("\n")
            cwd = parts[1].strip()
            with STATE.lock:
                STATE.cwd = cwd

        if stdout.strip():
            STATE.add_log("stdout", stdout)
        if stderr.strip():
            STATE.add_log("stderr", stderr)

        return {"ok": True, "cwd": cwd}

    except Exception as e:
        STATE.add_log("error", f"Exec failed: {e}")
        return {"ok": False, "error": str(e)}


# ── HTTP Handler ─────────────────────────────────────────────────────────────


class Handler(SimpleHTTPRequestHandler):
    default_gateway: str | None = None

    def log_message(self, format, *args):
        # Quieter logging
        pass

    def _json_response(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            content = HTML_FILE.read_bytes()
            self.send_header("Content-Length", len(content))
            self.end_headers()
            self.wfile.write(content)

        elif path == "/api/instances":
            dataset = qs.get("dataset", ["r2egym"])[0]
            q = qs.get("q", [""])[0]
            self._json_response(search_instances(dataset, q))

        elif path == "/api/session/status":
            with STATE.lock:
                self._json_response({
                    "status": STATE.status,
                    "status_msg": STATE.status_msg,
                    "instance_id": STATE.instance_id,
                    "cwd": STATE.cwd,
                })

        elif path == "/api/session/log":
            since = float(qs.get("since", [0])[0])
            entries = [e for e in STATE.log if e["ts"] > since]
            self._json_response(entries)

        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/session/start":
            body = self._read_body()
            dataset = body.get("dataset", "r2egym")
            instance_id = body.get("instance_id")
            gateway_url = self.default_gateway or body.get("gateway_url", "http://localhost:8080")
            experiment_id = body.get("experiment_id", "sandbox-explorer")

            if not instance_id:
                self._json_response({"ok": False, "error": "No instance_id"}, 400)
                return

            # Stop existing session first
            if STATE.session:
                _stop_session()

            threading.Thread(
                target=_start_session,
                args=(dataset, instance_id, gateway_url, experiment_id),
                daemon=True,
            ).start()
            self._json_response({"ok": True})

        elif path == "/api/session/stop":
            threading.Thread(target=_stop_session, daemon=True).start()
            self._json_response({"ok": True})

        elif path == "/api/session/exec":
            body = self._read_body()
            cmd = body.get("cmd", "")
            timeout = body.get("timeout", 60)
            result = _exec_cmd(cmd, timeout)
            self._json_response(result)

        else:
            self.send_error(404)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Sandbox Explorer server")
    parser.add_argument("--port", type=int, default=8501, help="Port (default: 8501)")
    parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    parser.add_argument("--gateway", default=None, help="Default ARL gateway URL (overrides frontend input)")
    args = parser.parse_args()

    if args.gateway:
        Handler.default_gateway = args.gateway

    # Pre-load instances in background so server starts immediately
    def _preload():
        print(f"Loading R2E-Gym instances from {DATA_DIR / 'R2E_Gym_Subset.parquet'}...")
        r2e = _load_instances("r2egym")
        print(f"  {len(r2e)} instances")
        print(f"Loading SWE-Bench instances from {DATA_DIR / 'SWE_Bench_Verified.parquet'}...")
        swe = _load_instances("swebench")
        print(f"  {len(swe)} instances")
        print("Instance loading complete.")

    threading.Thread(target=_preload, daemon=True).start()

    server = HTTPServer((args.host, args.port), Handler)
    print(f"\nSandbox Explorer running at http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        if STATE.session:
            _stop_session()
        server.server_close()


if __name__ == "__main__":
    main()
