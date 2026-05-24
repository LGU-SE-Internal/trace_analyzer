"""Upload client for Experiment Data Service.

Usage:
    # Login (one-time per machine, auto-registers if user doesn't exist)
    python -m utils.expdata.client login admin --password 42 --server http://expdata.default.svc.cluster.local:8502

    # Upload is typically called programmatically from pipeline scripts.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger("expdata_client")

DEFAULT_SERVER = "http://expdata.default.svc.cluster.local:8502"
TOKEN_PATH = Path.home() / ".config" / "expdata" / "token"


class ExperimentUploader:
    """Client for uploading experiment data to the expdata service."""

    def __init__(self, server_url: str = DEFAULT_SERVER, token_path: str = str(TOKEN_PATH)):
        self.server_url = server_url.rstrip("/")
        self.token_path = token_path
        self.token = self._load_token(token_path)
        self.session = requests.Session()
        if not self.token:
            self._auto_login()
        if self.token:
            self.session.headers["Authorization"] = f"Bearer {self.token}"

    def _auto_login(self):
        """Auto-login with default credentials if no token found."""
        import os

        username = os.environ.get("EXPDATA_USER", "admin")
        password = os.environ.get("EXPDATA_PASSWORD", "42")
        try:
            resp = requests.post(
                f"{self.server_url}/api/v1/auth/login",
                json={"username": username, "password": password},
                timeout=5,
            )
            if resp.status_code < 400:
                data = resp.json()
                self.token = data["token"]
                # Persist token for future use
                p = Path(self.token_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(self.token)
                p.chmod(0o600)
                logger.info(f"Auto-logged in as '{username}', token saved to {p}")
        except Exception as e:
            logger.warning(f"Auto-login failed: {e}")

    @staticmethod
    def _load_token(token_path: str) -> str | None:
        p = Path(token_path)
        if p.exists():
            return p.read_text().strip()
        # Also check env var
        import os

        return os.environ.get("EXPDATA_TOKEN")

    def _url(self, path: str) -> str:
        return f"{self.server_url}{path}"

    def _check(self, resp: requests.Response) -> dict:
        if resp.status_code >= 400:
            logger.error(f"API error {resp.status_code}: {resp.text}")
            resp.raise_for_status()
        return resp.json()

    def create_experiment(
        self,
        name: str,
        type: str,
        model: str | None = None,
        backend: str | None = None,
        scaffold: str | None = None,
        dataset: str | None = None,
        mode: str | None = None,
        n_samples: int | None = None,
        config: dict | None = None,
    ) -> int:
        """Create an experiment record. Returns experiment ID."""
        payload = {"name": name, "type": type}
        if model:
            payload["model"] = model
        if backend:
            payload["backend"] = backend
        if scaffold:
            payload["scaffold"] = scaffold
        if dataset:
            payload["dataset"] = dataset
        if mode:
            payload["mode"] = mode
        if n_samples is not None:
            payload["n_samples"] = n_samples
        if config:
            payload["config_json"] = json.dumps(config)
        resp = self.session.post(self._url("/api/v1/experiments"), json=payload)
        return self._check(resp)["id"]

    def upload_eval_results(self, exp_id: int, results: list[dict]) -> int:
        """Upload eval results as NDJSON. Returns count inserted."""
        ndjson = "\n".join(json.dumps(r) for r in results)
        resp = self.session.post(
            self._url(f"/api/v1/experiments/{exp_id}/upload/eval-results"),
            data=ndjson,
            headers={"Content-Type": "application/x-ndjson"},
        )
        return self._check(resp)["inserted"]

    def upload_trajectories(self, exp_id: int, trajectories: list[dict]) -> int:
        """Upload trajectories as NDJSON. Each dict should have instance_id, messages, etc."""
        ndjson = "\n".join(json.dumps(t) for t in trajectories)
        resp = self.session.post(
            self._url(f"/api/v1/experiments/{exp_id}/upload/trajectories"),
            data=ndjson,
            headers={"Content-Type": "application/x-ndjson"},
        )
        return self._check(resp)["inserted"]

    def upload_fault_traces(self, exp_id: int, traces_map: dict[str, list]) -> int:
        """Upload fault traces. traces_map: {instance_id: [trace_dicts]}."""
        resp = self.session.post(
            self._url(f"/api/v1/experiments/{exp_id}/upload/fault-traces"),
            json=traces_map,
        )
        return self._check(resp)["inserted"]

    def upload_test_outputs(self, exp_id: int, outputs_map: dict[str, str]) -> int:
        """Upload test outputs. outputs_map: {instance_id: output_text}."""
        resp = self.session.post(
            self._url(f"/api/v1/experiments/{exp_id}/upload/test-outputs"),
            json=outputs_map,
        )
        return self._check(resp)["inserted"]

    def upload_localization(self, exp_id: int, analysis: dict) -> None:
        """Upload localization analysis. analysis: {aggregate: {}, per_instance: []}."""
        resp = self.session.post(
            self._url(f"/api/v1/experiments/{exp_id}/upload/localization"),
            json=analysis,
        )
        self._check(resp)

    def upload_collection_parquet(self, exp_id: int, parquet_path: str) -> int:
        """Upload a collection parquet file."""
        with open(parquet_path, "rb") as f:
            resp = self.session.post(
                self._url(f"/api/v1/experiments/{exp_id}/upload/collection"),
                files={"file": (Path(parquet_path).name, f, "application/octet-stream")},
            )
        return self._check(resp)["inserted"]

    def mark_completed(self, exp_id: int, summary: dict | None = None) -> None:
        """Mark experiment as completed with optional summary."""
        payload: dict[str, Any] = {"status": "completed"}
        if summary:
            payload["summary_json"] = json.dumps(summary)
        resp = self.session.patch(self._url(f"/api/v1/experiments/{exp_id}"), json=payload)
        self._check(resp)


def login_cli(args):
    """Login to the expdata service and store token locally."""
    server = args.server.rstrip("/")
    resp = requests.post(f"{server}/api/v1/auth/login", json={"username": args.username, "password": args.password})
    if resp.status_code >= 400:
        print(f"Login failed: {resp.text}", file=sys.stderr)
        sys.exit(1)

    data = resp.json()
    token = data["token"]

    # Store token
    token_dir = TOKEN_PATH.parent
    token_dir.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(token)
    TOKEN_PATH.chmod(0o600)

    created = " (new account created)" if data.get("created") else ""
    print(f"Logged in as '{args.username}'{created}")
    print(f"Token saved to {TOKEN_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Experiment Data Service Client")
    sub = parser.add_subparsers(dest="command")

    login_cmd = sub.add_parser("login", help="Login and get API token")
    login_cmd.add_argument("username", help="Your username")
    login_cmd.add_argument("--password", required=True, help="Password")
    login_cmd.add_argument("--server", default=DEFAULT_SERVER, help="Server URL")

    args = parser.parse_args()
    if args.command == "login":
        login_cli(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
