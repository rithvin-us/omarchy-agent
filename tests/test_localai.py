"""Tests for the LocalAI client.

No test opens a socket: the URL opener is replaced with a fake that returns
canned bodies, so the suite needs neither LocalAI nor the network.
"""

from __future__ import annotations

import io
import json
import unittest
import urllib.error
from typing import Any

from omarchy_agent.integrations.localai import (
    DEFAULT_BASE_URL,
    LocalAIClient,
    LocalAIConfigurationError,
    LocalAIError,
    ModelInfo,
)
from omarchy_agent.security.audit import AuditLogger, get_audit_logger, set_audit_logger


class FakeResponse:
    """Minimal stand-in for an ``http.client.HTTPResponse``."""

    def __init__(self, body: bytes, status: int = 200) -> None:
        self._body = body
        self.status = status

    def read(self, amount: int | None = None) -> bytes:
        if amount is None:
            return self._body
        return self._body[:amount]

    def readline(self, amount: int | None = None) -> bytes:
        if not self._body:
            return b""
        index = self._body.find(b"\n")
        end = len(self._body) if index < 0 else index + 1
        if amount is not None:
            end = min(end, amount)
        line, self._body = self._body[:end], self._body[end:]
        return line

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


class FakeOpener:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.requests: list[Any] = []
        self.timeouts: list[float] = []

    def open(self, request: Any, timeout: float | None = None) -> Any:
        self.requests.append(request)
        if timeout is not None:
            self.timeouts.append(timeout)
        if isinstance(self.result, BaseException):
            raise self.result
        if callable(self.result):
            return self.result()
        return self.result


class SequenceOpener:
    def __init__(self, results: list[Any]) -> None:
        self.results = list(results)
        self.requests: list[Any] = []
        self.timeouts: list[float] = []

    def open(self, request: Any, timeout: float | None = None) -> Any:
        self.requests.append(request)
        self.timeouts.append(timeout or 0)
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def client_with(result: Any, **kwargs: Any) -> tuple[LocalAIClient, FakeOpener]:
    client = LocalAIClient(**kwargs)
    opener = FakeOpener(result)
    client._opener = opener  # noqa: SLF001 - deliberate injection point for tests
    return client, opener


def json_response(payload: Any, status: int = 200) -> FakeResponse:
    return FakeResponse(json.dumps(payload).encode("utf-8"), status=status)


MODELS_PAYLOAD = {
    "object": "list",
    "data": [
        {"id": "qwen3-4b-q8_0", "object": "model", "owned_by": "localai", "created": 1762000000},
        {"id": "whisper-base", "object": "model"},
    ],
}


class EndpointValidationTests(unittest.TestCase):
    def test_default_endpoint_is_loopback(self) -> None:
        self.assertEqual(DEFAULT_BASE_URL, "http://127.0.0.1:8080/v1")
        client = LocalAIClient()
        self.assertEqual(client.host, "127.0.0.1")
        self.assertEqual(client.port, 8080)

    def test_accepts_loopback_forms(self) -> None:
        for url in (
            "http://127.0.0.1:8080/v1",
            "http://localhost:8080/v1",
            "http://127.0.0.5:9000/v1",
            "http://[::1]:8080/v1",
        ):
            with self.subTest(url=url):
                self.assertTrue(LocalAIClient(url).base_url.startswith("http"))

    def test_rejects_non_loopback_hosts(self) -> None:
        for url in (
            "http://10.0.0.5:8080/v1",
            "http://192.168.1.10:8080/v1",
            "http://0.0.0.0:8080/v1",
            "http://api.openai.com/v1",
            "http://example.invalid/v1",
        ):
            with self.subTest(url=url), self.assertRaises(LocalAIConfigurationError):
                LocalAIClient(url)

    def test_rejects_unsupported_scheme(self) -> None:
        for url in ("ftp://127.0.0.1/v1", "file:///etc/passwd", "unix:///var/run/docker.sock"):
            with self.subTest(url=url), self.assertRaises(LocalAIConfigurationError):
                LocalAIClient(url)

    def test_trailing_slash_is_normalised(self) -> None:
        self.assertEqual(
            LocalAIClient("http://127.0.0.1:8080/v1/").base_url, "http://127.0.0.1:8080/v1"
        )

    def test_short_timeouts_by_default(self) -> None:
        client = LocalAIClient()
        self.assertLessEqual(client.connect_timeout, 5.0)
        self.assertLessEqual(client.read_timeout, 10.0)


class ConnectionSuccessTests(unittest.TestCase):
    def test_reports_reachable_with_models(self) -> None:
        client, opener = client_with(lambda: json_response(MODELS_PAYLOAD))
        status = client.check_connection()
        self.assertTrue(status.reachable)
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.model_count, 2)
        self.assertTrue(status.models_ready)
        self.assertIsNone(status.error)
        self.assertIsNone(status.hint)
        self.assertIsNotNone(status.latency_ms)
        self.assertTrue(opener.requests)

    def test_requests_the_models_endpoint(self) -> None:
        client, opener = client_with(lambda: json_response(MODELS_PAYLOAD))
        client.check_connection()
        self.assertEqual(opener.requests[0].full_url, "http://127.0.0.1:8080/v1/models")
        self.assertEqual(opener.requests[0].get_method(), "GET")

    def test_read_timeout_is_passed_to_the_opener(self) -> None:
        client, opener = client_with(lambda: json_response(MODELS_PAYLOAD))
        client.check_connection()
        self.assertEqual(opener.timeouts[0], client.read_timeout)

    def test_lists_models(self) -> None:
        client, _opener = client_with(lambda: json_response(MODELS_PAYLOAD))
        models = client.list_models()
        self.assertEqual([model.id for model in models], ["qwen3-4b-q8_0", "whisper-base"])
        self.assertEqual(models[0].owned_by, "localai")
        self.assertEqual(models[0].created, 1762000000)
        self.assertIsNone(models[1].owned_by)

    def test_has_models(self) -> None:
        client, _opener = client_with(lambda: json_response(MODELS_PAYLOAD))
        self.assertTrue(client.has_models())

    def test_resolve_model_id_without_preference(self) -> None:
        client, _opener = client_with(lambda: json_response(MODELS_PAYLOAD))
        self.assertEqual(client.resolve_model_id(), "qwen3-4b-q8_0")

    def test_resolve_model_id_with_a_matching_preference(self) -> None:
        client, _opener = client_with(lambda: json_response(MODELS_PAYLOAD))
        self.assertEqual(client.resolve_model_id("whisper-base"), "whisper-base")

    def test_resolve_model_id_refuses_to_invent_an_identifier(self) -> None:
        client, _opener = client_with(lambda: json_response(MODELS_PAYLOAD))
        with self.assertRaises(LocalAIError) as raised:
            client.resolve_model_id("qwen3-8b-that-is-not-installed")
        self.assertIn("not installed", str(raised.exception))


class EmptyModelListTests(unittest.TestCase):
    def test_empty_data_list_is_reachable_but_not_ready(self) -> None:
        client, _opener = client_with(lambda: json_response({"object": "list", "data": []}))
        status = client.check_connection()
        self.assertTrue(status.reachable)
        self.assertEqual(status.model_count, 0)
        self.assertFalse(status.models_ready)
        self.assertIsNotNone(status.hint)
        self.assertIn("no installed models", status.hint or "")

    def test_empty_model_list_returns_empty(self) -> None:
        client, _opener = client_with(lambda: json_response({"object": "list", "data": []}))
        self.assertEqual(client.list_models(), [])
        self.assertFalse(client.has_models())

    def test_missing_data_key_returns_empty(self) -> None:
        client, _opener = client_with(lambda: json_response({"object": "list"}))
        self.assertEqual(client.list_models(), [])

    def test_empty_body_returns_empty(self) -> None:
        client, _opener = client_with(lambda: FakeResponse(b"", status=200))
        self.assertEqual(client.list_models(), [])

    def test_resolve_model_id_raises_when_nothing_is_installed(self) -> None:
        client, _opener = client_with(lambda: json_response({"data": []}))
        with self.assertRaises(LocalAIError) as raised:
            client.resolve_model_id()
        self.assertIn("no installed models", str(raised.exception))


class ConnectionFailureTests(unittest.TestCase):
    def test_connection_refused_is_reported_not_raised(self) -> None:
        client, _opener = client_with(
            urllib.error.URLError(ConnectionRefusedError("Connection refused"))
        )
        status = client.check_connection()
        self.assertFalse(status.reachable)
        self.assertIsNone(status.status_code)
        self.assertIn("Connection refused", status.error or "")
        self.assertIn("never starts or", status.hint or "")

    def test_timeout_is_reported_not_raised(self) -> None:
        client, _opener = client_with(TimeoutError("timed out"))
        status = client.check_connection()
        self.assertFalse(status.reachable)
        self.assertIsNotNone(status.error)

    def test_dns_failure_is_reported_not_raised(self) -> None:
        client, _opener = client_with(urllib.error.URLError("nodename nor servname provided"))
        self.assertFalse(client.check_connection().reachable)

    def test_list_models_raises_on_transport_failure(self) -> None:
        client, _opener = client_with(urllib.error.URLError("refused"))
        with self.assertRaises(LocalAIError):
            client.list_models()

    def test_has_models_is_false_on_transport_failure(self) -> None:
        client, _opener = client_with(urllib.error.URLError("refused"))
        self.assertFalse(client.has_models())

    def test_http_error_status_is_surfaced(self) -> None:
        error = urllib.error.HTTPError(
            url="http://127.0.0.1:8080/v1/models",
            code=503,
            msg="Service Unavailable",
            hdrs=None,
            fp=io.BytesIO(b'{"error": "unavailable"}'),
        )
        client, _opener = client_with(error)
        status = client.check_connection()
        self.assertFalse(status.reachable)
        self.assertEqual(status.status_code, 503)
        self.assertIn("503", status.error or "")

    def test_http_error_makes_list_models_raise(self) -> None:
        error = urllib.error.HTTPError(
            url="http://127.0.0.1:8080/v1/models",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=io.BytesIO(b""),
        )
        client, _opener = client_with(error)
        with self.assertRaises(LocalAIError) as raised:
            client.list_models()
        self.assertIn("404", str(raised.exception))

    def test_generic_oserror_is_reported(self) -> None:
        client, _opener = client_with(OSError("network unreachable"))
        self.assertFalse(client.check_connection().reachable)


class MalformedResponseTests(unittest.TestCase):
    def test_non_json_body_raises(self) -> None:
        client, _opener = client_with(lambda: FakeResponse(b"<html>not json</html>"))
        with self.assertRaises(LocalAIError) as raised:
            client.list_models()
        self.assertIn("not valid JSON", str(raised.exception))

    def test_non_json_body_is_reported_by_check_connection(self) -> None:
        client, _opener = client_with(lambda: FakeResponse(b"<html>not json</html>"))
        self.assertFalse(client.check_connection().reachable)

    def test_data_of_the_wrong_type_raises(self) -> None:
        client, _opener = client_with(lambda: json_response({"data": "not-a-list"}))
        with self.assertRaises(LocalAIError) as raised:
            client.list_models()
        self.assertIn("expected a list", str(raised.exception))

    def test_entries_without_an_identifier_are_dropped(self) -> None:
        client, _opener = client_with(
            lambda: json_response({"data": [{"object": "model"}, {"id": ""}, {"id": "good"}, 42]})
        )
        self.assertEqual([model.id for model in client.list_models()], ["good"])

    def test_bare_list_of_strings_is_accepted(self) -> None:
        client, _opener = client_with(lambda: json_response(["alpha", "beta"]))
        self.assertEqual([model.id for model in client.list_models()], ["alpha", "beta"])

    def test_oversized_body_is_refused(self) -> None:
        huge = b'{"data": []}' + b" " * (2 * 1024 * 1024)
        client, _opener = client_with(lambda: FakeResponse(huge))
        with self.assertRaises(LocalAIError) as raised:
            client.list_models()
        self.assertIn("exceeded", str(raised.exception))


class ChatCompletionTests(unittest.TestCase):
    def test_posts_completion_and_returns_assistant_message(self) -> None:
        payload = {"choices": [{"message": {"role": "assistant", "content": "hello there"}}]}
        opener = SequenceOpener([json_response(MODELS_PAYLOAD), json_response(payload)])
        client = LocalAIClient()
        client._opener = opener  # noqa: SLF001
        result = client.chat_completion(
            [{"role": "user", "content": "hello"}], model="qwen3-4b-q8_0"
        )
        self.assertEqual(result["content"], "hello there")
        request = opener.requests[1]
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.full_url, "http://127.0.0.1:8080/v1/chat/completions")
        body = json.loads(request.data)
        self.assertEqual(body["model"], "qwen3-4b-q8_0")
        self.assertEqual(body["messages"][0]["content"], "hello")
        self.assertEqual(opener.timeouts[1], client.inference_timeout)

    def test_streams_visible_text_and_accumulates_tool_call(self) -> None:
        events = [
            {"choices": [{"delta": {"content": "hello"}}]},
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{\\"path\\":\\"x.py\\"}',
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
        ]
        stream = (
            b"".join(b"data: " + json.dumps(item).encode() + b"\n" for item in events)
            + b"data: [DONE]\n"
        )
        opener = SequenceOpener([json_response(MODELS_PAYLOAD), FakeResponse(stream)])
        client = LocalAIClient()
        client._opener = opener  # noqa: SLF001
        fragments: list[str] = []
        result = client.chat_completion(
            [{"role": "user", "content": "hello"}],
            model="qwen3-4b-q8_0",
            on_delta=fragments.append,
        )
        self.assertEqual(fragments, ["hello"])
        self.assertEqual(result["tool_calls"][0]["id"], "call_1")
        self.assertIn("read_file", result["tool_calls"][0]["function"]["name"])

    def test_prompt_is_redacted_from_audit(self) -> None:
        previous = get_audit_logger()
        logger = AuditLogger()
        set_audit_logger(logger)
        try:
            payload = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
            opener = SequenceOpener([json_response(MODELS_PAYLOAD), json_response(payload)])
            client = LocalAIClient()
            client._opener = opener  # noqa: SLF001
            secret_prompt = "private prompt content"
            client.chat_completion(
                [{"role": "user", "content": secret_prompt}], model="qwen3-4b-q8_0"
            )
            serialized = json.dumps(logger.entries()[-1].as_dict())
            self.assertNotIn(secret_prompt, serialized)
        finally:
            set_audit_logger(previous)


class LocalAIAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous = get_audit_logger()
        self.logger = AuditLogger()
        set_audit_logger(self.logger)

    def tearDown(self) -> None:
        set_audit_logger(self.previous)

    def test_check_connection_is_audited(self) -> None:
        client, _opener = client_with(lambda: json_response(MODELS_PAYLOAD))
        client.check_connection()
        entry = self.logger.entries()[-1]
        self.assertEqual(entry.tool_name, "localai.check_connection")
        self.assertEqual(entry.status, "success")
        self.assertIn("reachable=True", entry.result_summary)

    def test_list_models_is_audited(self) -> None:
        client, _opener = client_with(lambda: json_response(MODELS_PAYLOAD))
        client.list_models()
        entry = self.logger.entries()[-1]
        self.assertEqual(entry.tool_name, "localai.list_models")
        self.assertIn("2 items", entry.result_summary)

    def test_failure_is_audited(self) -> None:
        client, _opener = client_with(urllib.error.URLError("refused"))
        with self.assertRaises(LocalAIError):
            client.list_models()
        entry = self.logger.entries()[-1]
        self.assertEqual(entry.status, "error")
        self.assertIn("LocalAIError", entry.result_summary)


class ModelInfoTests(unittest.TestCase):
    def test_as_dict(self) -> None:
        model = ModelInfo(id="x", owned_by="localai", created=1)
        self.assertEqual(
            model.as_dict(), {"id": "x", "object": "model", "owned_by": "localai", "created": 1}
        )

    def test_audit_summary_contains_only_the_identifier(self) -> None:
        self.assertEqual(ModelInfo(id="x").audit_summary(), "model id=x")


if __name__ == "__main__":
    unittest.main()
