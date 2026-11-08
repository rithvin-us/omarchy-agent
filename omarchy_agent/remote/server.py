"""Lightweight, token-authenticated HTTP server for remote subdomain access."""

from __future__ import annotations

import hmac
import http.server
import io
import json
import secrets
import shutil
import subprocess
import sys
import threading
import urllib.parse
import wave
from pathlib import Path
from typing import Any

from ..core.chat import ChatSession
from ..core.task_manager import TaskManager
from ..integrations.localai import DEFAULT_BASE_URL, LocalAIClient
from ..security.audit import audited
from ..security.permissions import Capability, PermissionPolicy, default_permission_policy
from .approvals import ApprovalBroker
from .web import PAGE

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
        self.approvals = ApprovalBroker()
        self._voice_lock = threading.Lock()
        self._transcriber: Any = None
        self._speaker: Any = None
        self.policy = policy or default_permission_policy()
        self.policy.require(Capability.REMOTE_ACCESS)
        super().__init__(server_address, RemoteRequestHandler)


class RemoteRequestHandler(http.server.BaseHTTPRequestHandler):
    """Handles health, task list, and chat completions with Bearer token authentication."""

    server: RemoteServer

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":  # static page only; every data endpoint needs the token
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)
            return
        if not self._authenticate(parsed):
            return

        if parsed.path == "/api/approvals":
            self._send_json(200, {"pending": self.server.approvals.pending()})
        elif parsed.path == "/api/health":
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
        elif parsed.path.startswith("/api/approvals/"):
            self._handle_resolve(parsed.path.rsplit("/", 1)[1])
        elif parsed.path == "/api/transcribe":
            self._handle_transcribe()
        elif parsed.path == "/api/speak":
            self._handle_speak()
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
        pending_cb: Any = None

        def confirm(description: str, _cwd: Path) -> bool:
            return self.server.approvals.request(description, pending_cb)

        session = ChatSession(
            self.server.client,
            self.server.repo_root,
            on_confirm_command=confirm,
            memory=self.server.manager,
        )

        if stream:
            self.close_connection = True
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            def pending_cb(info: dict[str, Any]) -> None:  # noqa: F811
                self.wfile.write(f"data: {json.dumps({'approval': info})}\n\n".encode())
                self.wfile.flush()

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

    def _read_body(self, limit: int) -> bytes | None:
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0 or length > limit:
            self._send_json(413, {"error": "missing or oversized body"})
            return None
        return self.rfile.read(length)

    def _handle_resolve(self, raw_id: str) -> None:
        body = self._read_body(1024)
        if body is None:
            return
        try:
            approve = json.loads(body)["approve"] is True
            ok = self.server.approvals.resolve(int(raw_id), approve)
        except (ValueError, KeyError, TypeError):
            self._send_json(400, {"error": 'expected {"approve": true|false}'})
            return
        self._send_json(200 if ok else 404, {"resolved": ok})

    def _handle_transcribe(self) -> None:
        body = self._read_body(10 * 1024 * 1024)
        if body is None:
            return
        try:
            import numpy as np

            from ..voice.engine import Transcriber

            ff = subprocess.run(  # noqa: S603
                [
                    shutil.which("ffmpeg") or "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    "pipe:0",
                    "-f",
                    "s16le",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "pipe:1",
                ],
                input=body,
                capture_output=True,
                timeout=60,
                check=True,
            )
            audio = np.frombuffer(ff.stdout, dtype=np.int16).astype(np.float32) / 32768.0
            with self.server._voice_lock:
                if self.server._transcriber is None:
                    self.server._transcriber = Transcriber()
                text = self.server._transcriber.transcribe(audio)
        except ImportError:
            self._send_json(501, {"error": "voice extras missing: uv sync --extra voice"})
            return
        except (subprocess.SubprocessError, OSError, ValueError) as exc:
            self._send_json(400, {"error": f"transcription failed: {type(exc).__name__}"})
            return
        self._send_json(200, {"text": text})

    def _handle_speak(self) -> None:
        body = self._read_body(64 * 1024)
        if body is None:
            return
        try:
            text = str(json.loads(body)["text"])
            from ..voice.engine import Speaker, speakable

            spoken = speakable(text)
            with self.server._voice_lock:
                if self.server._speaker is None:
                    self.server._speaker = Speaker()
                voice = self.server._speaker._load()
                buf = io.BytesIO()
                with wave.open(buf, "wb") as wav:
                    voice.synthesize_wav(spoken, wav)
        except ImportError:
            self._send_json(501, {"error": "voice extras missing: uv sync --extra voice"})
            return
        except (ValueError, KeyError, OSError) as exc:
            self._send_json(400, {"error": f"speech failed: {type(exc).__name__}"})
            return
        data = buf.getvalue()
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

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
