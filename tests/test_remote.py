"""Unit tests for the remote access HTTP server."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from omarchy_agent.remote.server import create_remote_server


class RemoteServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "README.md").write_text("# Project\n", encoding="utf-8")
        self.token = "secret-test-token-12345"

        self.mock_client = mock.MagicMock()
        self.mock_client.resolve_model_id.return_value = "localai-test-model"

        self.mock_manager = mock.MagicMock()
        task1 = mock.MagicMock()
        task1.as_dict.return_value = {
            "id": 1,
            "title": "First Task",
            "status": "pending",
        }
        self.mock_manager.list_tasks.return_value = [task1]

        with mock.patch("omarchy_agent.remote.server.LocalAIClient", return_value=self.mock_client):
            self.server = create_remote_server(
                host="127.0.0.1",
                port=0,
                token=self.token,
                repo_root=self.root,
                manager=self.mock_manager,
            )

        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join(timeout=1.0)
        self.temp_dir.cleanup()

    def _request(
        self,
        path: str,
        *,
        method: str = "GET",
        token: str | None = None,
        data: dict | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict | str, dict[str, str]]:
        url = f"{self.base_url}{path}"
        req_headers = dict(headers or {})
        if token is not None:
            req_headers["Authorization"] = f"Bearer {token}"

        body_bytes = None
        if data is not None:
            body_bytes = json.dumps(data).encode("utf-8")
            req_headers["Content-Type"] = "application/json"

        req = urllib.request.Request(  # noqa: S310
            url, data=body_bytes, headers=req_headers, method=method
        )
        try:
            with urllib.request.urlopen(req, timeout=3.0) as resp:  # noqa: S310
                code = resp.getcode()
                content_type = resp.headers.get("Content-Type", "")
                raw = resp.read().decode("utf-8")
                resp_headers = dict(resp.headers)
                if "application/json" in content_type:
                    return code, json.loads(raw), resp_headers
                return code, raw, resp_headers
        except urllib.error.HTTPError as exc:
            with exc:
                raw = exc.read().decode("utf-8")
                resp_headers = dict(exc.headers)
                try:
                    return exc.code, json.loads(raw), resp_headers
                except ValueError:
                    return exc.code, raw, resp_headers

    def test_missing_token_returns_401(self) -> None:
        code, body, _ = self._request("/api/health")
        self.assertEqual(code, 401)
        self.assertEqual(body.get("error"), "unauthorized")

    def test_invalid_token_returns_401(self) -> None:
        code, body, _ = self._request("/api/health", token="wrong-token")
        self.assertEqual(code, 401)
        self.assertEqual(body.get("error"), "unauthorized")

    def test_health_with_bearer_token(self) -> None:
        code, body, _ = self._request("/api/health", token=self.token)
        self.assertEqual(code, 200)
        self.assertEqual(body.get("status"), "ok")
        self.assertEqual(body.get("model"), "localai-test-model")
        self.assertEqual(body.get("repo_root"), str(self.root.resolve()))

    def test_health_with_query_param_token(self) -> None:
        code, body, _ = self._request(f"/api/health?token={self.token}")
        self.assertEqual(code, 200)
        self.assertEqual(body.get("status"), "ok")

    def test_tasks_endpoint(self) -> None:
        code, body, _ = self._request("/api/tasks", token=self.token)
        self.assertEqual(code, 200)
        self.assertIn("tasks", body)
        self.assertEqual(len(body["tasks"]), 1)
        self.assertEqual(body["tasks"][0]["title"], "First Task")

    def test_chat_endpoint_json(self) -> None:
        with mock.patch("omarchy_agent.remote.server.ChatSession") as mock_session_cls:
            mock_session = mock.MagicMock()
            mock_session.run_turn.return_value = "Response from model"
            mock_session_cls.return_value = mock_session

            code, body, _ = self._request(
                "/api/chat",
                method="POST",
                token=self.token,
                data={"prompt": "Hello model"},
            )
            self.assertEqual(code, 200)
            self.assertEqual(body.get("answer"), "Response from model")
            mock_session.run_turn.assert_called_once_with("Hello model")

    def test_chat_endpoint_sse_streaming(self) -> None:
        with mock.patch("omarchy_agent.remote.server.ChatSession") as mock_session_cls:
            mock_session = mock.MagicMock()

            def fake_run_turn(prompt: str, on_delta=None):
                if on_delta:
                    on_delta("Hello ")
                    on_delta("World!")
                return "Hello World!"

            mock_session.run_turn.side_effect = fake_run_turn
            mock_session_cls.return_value = mock_session

            code, body, headers = self._request(
                "/api/chat",
                method="POST",
                token=self.token,
                data={"prompt": "Stream this", "stream": True},
            )
            self.assertEqual(code, 200)
            self.assertIn("text/event-stream", headers.get("Content-Type", ""))
            self.assertIn('data: {"delta": "Hello "}', body)
            self.assertIn('data: {"delta": "World!"}', body)
            self.assertIn("data: [DONE]", body)

    def test_chat_endpoint_empty_prompt_rejected(self) -> None:
        code, body, _ = self._request(
            "/api/chat",
            method="POST",
            token=self.token,
            data={"prompt": "   "},
        )
        self.assertEqual(code, 400)
        self.assertIn("cannot be empty", body.get("error", ""))

    def test_not_found_endpoint(self) -> None:
        code, body, _ = self._request("/api/nonexistent", token=self.token)
        self.assertEqual(code, 404)
        self.assertEqual(body.get("error"), "endpoint not found")


if __name__ == "__main__":
    unittest.main()
