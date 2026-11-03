"""Lightweight, token-authenticated HTTP server for remote subdomain access."""

from __future__ import annotations

import hmac
import http.server
import json
import secrets
import sys
import urllib.parse
from pathlib import Path
from typing import Any

from ..core.chat import ChatSession
from ..core.task_manager import TaskManager
from ..integrations.localai import DEFAULT_BASE_URL, LocalAIClient
from ..security.audit import audited
from ..security.permissions import Capability, PermissionPolicy, default_permission_policy

DEFAULT_REMOTE_HOST = "127.0.0.1"
DEFAULT_REMOTE_PORT = 8443


class RemoteServer(http.server.ThreadingHTTPServer):
    """Threading HTTP server carrying configuration for the remote request handler."""

    def __init__(
        self,
        server_address: tuple[str, int],
        token: str,
        repo_root: Path,
        *,
        client: LocalAIClient | None = None,
        manager: TaskManager | None = None,
        policy: PermissionPolicy | None = None,
    ) -> None:
        self.token = token
        self.repo_root = repo_root
        self.client = client or LocalAIClient()
        self.manager = manager
        self.policy = policy or default_permission_policy()
        self.policy.require(Capability.REMOTE_ACCESS)
        super().__init__(server_address, RemoteRequestHandler)


class RemoteRequestHandler(http.server.BaseHTTPRequestHandler):
    """Handles health, task list, and chat completions with Bearer token authentication."""

    server: RemoteServer

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if not self._authenticate(parsed):
            return

        if parsed.path == "/api/health":
            self._handle_health()
        elif parsed.path == "/api/tasks":
            self._handle_tasks()
        else:
            self._send_json(404, {"error": "endpoint not found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if not self._authenticate(parsed):
            return

        if parsed.path == "/api/chat":
            self._handle_chat()
        else:
            self._send_json(404, {"error": "endpoint not found"})

    def _authenticate(self, parsed: urllib.parse.ParseResult) -> bool:
        auth_header = self.headers.get("Authorization", "")
        token = ""
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
        else:
            query = urllib.parse.parse_qs(parsed.query)
            token = query.get("token", [""])[0].strip()

        if not token or not hmac.compare_digest(token, self.server.token):
            self._send_json(401, {"error": "unauthorized"})
            return False
        return True

    def _handle_health(self) -> None:
        model_id = self.server.client.resolve_model_id()
        payload = {
            "status": "ok",
            "repo_root": str(self.server.repo_root),
            "model": model_id,
        }
        self._send_json(200, payload)

    def _handle_tasks(self) -> None:
        tasks: list[dict[str, Any]] = []
        if self.server.manager is not None:
            tasks = [task.as_dict() for task in self.server.manager.list_tasks()]
        self._send_json(200, {"tasks": tasks})

    def _handle_chat(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 128 * 1024:
            self._send_json(413, {"error": "request body too large"})
            return

        raw_body = self.rfile.read(content_length)
        try:
            body = json.loads(raw_body.decode("utf-8"))
        except ValueError:
            self._send_json(400, {"error": "invalid JSON body"})
            return

        prompt = str(body.get("prompt", "")).strip()
        if not prompt:
            self._send_json(400, {"error": "prompt cannot be empty"})
            return

        stream = bool(body.get("stream", False)) or "text/event-stream" in self.headers.get(
            "Accept", ""
        )
        session = ChatSession(self.server.client, self.server.repo_root)

        if stream:
            self.close_connection = True
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            def on_delta(fragment: str) -> None:
                event_data = json.dumps({"delta": fragment})
                self.wfile.write(f"data: {event_data}\n\n".encode())
                self.wfile.flush()

            try:
                answer = session.run_turn(prompt, on_delta=on_delta)
                done_data = json.dumps({"done": True, "answer": answer})
                self.wfile.write(f"data: {done_data}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except Exception as exc:  # noqa: BLE001
                err_data = json.dumps({"error": str(exc)})
                self.wfile.write(f"data: {err_data}\n\n".encode())
                self.wfile.flush()
        else:
            try:
                answer = session.run_turn(prompt)
                self._send_json(200, {"answer": answer})
            except Exception as exc:  # noqa: BLE001
                self._send_json(500, {"error": str(exc)})

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        sys.stderr.write(f"remote: {format % args}\n")


@audited("remote.server_create", Capability.REMOTE_ACCESS)
def create_remote_server(
    host: str = DEFAULT_REMOTE_HOST,
    port: int = DEFAULT_REMOTE_PORT,
    *,
    token: str | None = None,
    repo_root: str | Path | None = None,
    endpoint: str = DEFAULT_BASE_URL,
    manager: TaskManager | None = None,
) -> RemoteServer:
    """Instantiate a configured remote server."""
    active_token = token or secrets.token_urlsafe(24)
    root = Path(repo_root or Path.cwd()).expanduser().resolve()
    client = LocalAIClient(endpoint)
    return RemoteServer(
        (host, port),
        active_token,
        root,
        client=client,
        manager=manager,
    )
