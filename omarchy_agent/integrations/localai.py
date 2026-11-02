"""Small, loopback-only LocalAI client for inspection and chat completion.

Constraints enforced here:

* The endpoint must be a loopback address. A non-loopback host is rejected
  before any socket is opened, so this client cannot be pointed at a remote
  service by configuration drift.
* Timeouts are short and explicit.
* Responses are parsed defensively: a malformed or truncated body produces a
  clear error, never an exception from deep inside the JSON parser.
* The model identifier is never assumed. Callers ask
  :meth:`LocalAIClient.list_models` and handle the empty case.
* Docker is never started, stopped or reconfigured, and the Docker socket is
  never touched.
"""

from __future__ import annotations

import ipaddress
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..security.audit import audited
from ..security.permissions import (
    Capability,
    PermissionPolicy,
    default_permission_policy,
)

__all__ = [
    "DEFAULT_BASE_URL",
    "ConnectionStatus",
    "LocalAIClient",
    "LocalAIError",
    "LocalAIConfigurationError",
    "ModelInfo",
]

DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_CONNECT_TIMEOUT = 2.0
DEFAULT_READ_TIMEOUT = 5.0
DEFAULT_INFERENCE_TIMEOUT = 180.0
MAX_RESPONSE_BYTES = 1 * 1024 * 1024
MAX_REQUEST_BYTES = 512 * 1024
LOOPBACK_HOSTNAMES = frozenset({"localhost", "localhost.localdomain", "ip6-localhost"})


class LocalAIError(RuntimeError):
    """Raised for LocalAI problems the caller is expected to report."""


class LocalAIConfigurationError(LocalAIError):
    """Raised when the configured endpoint violates the loopback-only rule."""


@dataclass(frozen=True)
class ModelInfo:
    """One entry from ``GET /v1/models``."""

    id: str
    object: str = "model"
    owned_by: str | None = None
    created: int | None = None
    raw_keys: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": self.object,
            "owned_by": self.owned_by,
            "created": self.created,
        }

    def audit_summary(self) -> str:
        return f"model id={self.id}"


@dataclass(frozen=True)
class ConnectionStatus:
    """Result of a connectivity probe. Never raises; always describes."""

    base_url: str
    reachable: bool
    status_code: int | None
    latency_ms: int | None
    model_count: int | None
    models_ready: bool
    error: str | None = None
    hint: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "reachable": self.reachable,
            "status_code": self.status_code,
            "latency_ms": self.latency_ms,
            "model_count": self.model_count,
            "models_ready": self.models_ready,
            "error": self.error,
            "hint": self.hint,
        }

    def audit_summary(self) -> str:
        return (
            f"localai {self.base_url}: reachable={self.reachable} "
            f"status={self.status_code} models={self.model_count}"
        )


def _is_loopback_host(host: str) -> bool:
    candidate = host.strip().strip("[]").lower()
    if not candidate:
        return False
    if candidate in LOOPBACK_HOSTNAMES:
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


class LocalAIClient:
    """A small client for inspection and chat on a LocalAI loopback instance.

    Inspection and inference share the same endpoint validation, transport
    limits, model discovery, permission policy, and audit boundary.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        read_timeout: float = DEFAULT_READ_TIMEOUT,
        inference_timeout: float = DEFAULT_INFERENCE_TIMEOUT,
        policy: PermissionPolicy | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        parsed = urllib.parse.urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"}:
            raise LocalAIConfigurationError(
                f"unsupported scheme {parsed.scheme!r}; expected http or https"
            )
        if not _is_loopback_host(parsed.hostname or ""):
            raise LocalAIConfigurationError(
                f"endpoint host {parsed.hostname!r} is not a loopback address; "
                "this agent only talks to LocalAI on 127.0.0.1 / ::1 / localhost"
            )
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self.connect_timeout = float(connect_timeout)
        self.read_timeout = float(read_timeout)
        self.inference_timeout = float(inference_timeout)
        self._policy = policy or default_permission_policy()
        # An opener with an empty ProxyHandler: a system proxy must never
        # intercept a loopback request.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    # -- low level -------------------------------------------------------

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def _get_json(self, path: str) -> tuple[int, Any]:
        """GET ``path`` and parse the body as JSON.

        Raises :class:`LocalAIError` for transport problems and for bodies that
        are not valid JSON objects.
        """
        request = urllib.request.Request(  # noqa: S310 - scheme validated in __init__
            self._url(path),
            method="GET",
            headers={"Accept": "application/json", "User-Agent": "omarchy-agent/0.1"},
        )
        try:
            with self._opener.open(request, timeout=self.read_timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
                status = int(getattr(response, "status", 0) or 0)
        except urllib.error.HTTPError as exc:
            # A 4xx/5xx still carries a status and possibly a JSON error body.
            try:
                raw = exc.read(MAX_RESPONSE_BYTES)
            except Exception:  # noqa: BLE001 - body is optional
                raw = b""
            finally:
                # HTTPError wraps a file object; closing it releases the socket
                # (or, in tests, the buffer) instead of waiting for the GC.
                exc.close()
            status = int(exc.code)
            payload: Any = None
            if raw:
                try:
                    payload = json.loads(raw.decode("utf-8", errors="replace"))
                except ValueError:
                    payload = None
            return status, payload
        except urllib.error.URLError as exc:
            raise LocalAIError(f"cannot reach LocalAI at {self.base_url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise LocalAIError(
                f"LocalAI at {self.base_url} did not respond within {self.read_timeout}s"
            ) from exc
        except OSError as exc:
            raise LocalAIError(f"network error talking to {self.base_url}: {exc}") from exc

        if len(raw) > MAX_RESPONSE_BYTES:
            raise LocalAIError(
                f"LocalAI response exceeded {MAX_RESPONSE_BYTES} bytes; refusing to parse"
            )
        if not raw:
            return status, None
        try:
            return status, json.loads(raw.decode("utf-8", errors="replace"))
        except ValueError as exc:
            raise LocalAIError(f"LocalAI returned a body that is not valid JSON: {exc}") from exc

    # -- public API ------------------------------------------------------

    @audited("localai.check_connection", Capability.LOCALAI_INSPECT)
    def check_connection(self) -> ConnectionStatus:
        """Probe the API. Returns a status object; does not raise on failure."""
        started = time.monotonic()
        try:
            status_code, payload = self._get_json("models")
        except LocalAIError as exc:
            return ConnectionStatus(
                base_url=self.base_url,
                reachable=False,
                status_code=None,
                latency_ms=int((time.monotonic() - started) * 1000),
                model_count=None,
                models_ready=False,
                error=str(exc),
                hint=(
                    "Confirm the LocalAI container is running and published on "
                    f"{self.host}:{self.port}. This agent never starts or "
                    "reconfigures Docker."
                ),
            )

        latency = int((time.monotonic() - started) * 1000)
        if status_code != 200:
            return ConnectionStatus(
                base_url=self.base_url,
                reachable=False,
                status_code=status_code,
                latency_ms=latency,
                model_count=None,
                models_ready=False,
                error=f"LocalAI answered with HTTP {status_code}",
                hint="The port is open but /v1/models did not return 200.",
            )

        models = _parse_model_payload(payload)
        return ConnectionStatus(
            base_url=self.base_url,
            reachable=True,
            status_code=status_code,
            latency_ms=latency,
            model_count=len(models),
            models_ready=bool(models),
            error=None,
            hint=None
            if models
            else (
                "The API is up but reports no installed models. Finish the "
                "gallery installation, then re-run this check. The model "
                "identifier is read from this endpoint, never guessed."
            ),
        )

    @audited("localai.list_models", Capability.LOCALAI_INSPECT)
    def list_models(self) -> list[ModelInfo]:
        """Return installed models. An empty list is a valid, expected result."""
        status_code, payload = self._get_json("models")
        if status_code != 200:
            raise LocalAIError(f"GET /v1/models returned HTTP {status_code}")
        return _parse_model_payload(payload)

    def has_models(self) -> bool:
        """Convenience wrapper: ``True`` when at least one model is installed."""
        try:
            return bool(self.list_models())
        except LocalAIError:
            return False

    def resolve_model_id(self, preferred: str | None = None) -> str:
        """Return a usable model id, read from the server.

        ``preferred`` is honoured only if the server actually reports it. With
        no preference the first reported model is returned. The identifier is
        never fabricated.
        """
        models = self.list_models()
        if not models:
            raise LocalAIError(
                "LocalAI reports no installed models; install one through the "
                "gallery before requesting inference"
            )
        if preferred is not None:
            for model in models:
                if model.id == preferred:
                    return model.id
            available = ", ".join(model.id for model in models[:10])
            raise LocalAIError(f"model {preferred!r} is not installed; available: {available}")
        return models[0].id

    # -- chat completion -------------------------------------------------

    @audited(
        "localai.chat_completion",
        Capability.LOCALAI_INFERENCE,
        redact_params=("messages",),
    )
    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_delta: Callable[[str], None] | None = None,
        on_usage: Callable[[dict[str, int | bool]], None] | None = None,
        max_tokens: int = 600,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        """Request a chat completion, optionally streaming visible answer text.

        Prompt messages are always redacted from audit records. Only narrowly
        supported generation options are exposed so arbitrary request fields
        cannot silently expand the integration's behavior.
        """
        self._policy.require(Capability.LOCALAI_INFERENCE)
        if not isinstance(messages, list) or not messages:
            raise LocalAIError("messages must be a non-empty list")
        if len(messages) > 40 or any(not isinstance(item, dict) for item in messages):
            raise LocalAIError("messages must contain at most 40 message objects")
        if not 1 <= int(max_tokens) <= 4096:
            raise LocalAIError("max_tokens must be between 1 and 4096")
        if not 0.0 <= float(temperature) <= 2.0:
            raise LocalAIError("temperature must be between 0 and 2")

        selected_model = self.resolve_model_id(model)
        body: dict[str, Any] = {
            "model": selected_model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "stream": on_delta is not None,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if on_delta is not None:
            # LocalAI returns exact request usage in the trailing SSE chunk.
            body["stream_options"] = {"include_usage": True}
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        if len(encoded) > MAX_REQUEST_BYTES:
            raise LocalAIError(f"chat request exceeded {MAX_REQUEST_BYTES} bytes; refusing to send")
        estimated_prompt_tokens = _estimate_tokens(encoded)
        if on_usage is not None:
            on_usage(
                {
                    "prompt_tokens": estimated_prompt_tokens,
                    "completion_tokens": 0,
                    "total_tokens": estimated_prompt_tokens,
                    "estimated": True,
                }
            )
        return self._post_chat_completion(
            encoded,
            on_delta=on_delta,
            on_usage=on_usage,
            estimated_prompt_tokens=estimated_prompt_tokens,
        )

    def _post_chat_completion(
        self,
        encoded: bytes,
        *,
        on_delta: Callable[[str], None] | None,
        on_usage: Callable[[dict[str, int | bool]], None] | None,
        estimated_prompt_tokens: int,
    ) -> dict[str, Any]:
        request = urllib.request.Request(  # noqa: S310 - scheme validated in __init__
            self._url("chat/completions"),
            data=encoded,
            method="POST",
            headers={
                "Accept": "text/event-stream, application/json"
                if on_delta is not None
                else "application/json",
                "Content-Type": "application/json",
                "User-Agent": "omarchy-agent/0.1",
            },
        )
        try:
            with self._opener.open(request, timeout=self.inference_timeout) as response:
                status = int(getattr(response, "status", 0) or 0)
                if status != 200:
                    raise LocalAIError(f"POST /v1/chat/completions returned HTTP {status}")
                if on_delta is None:
                    raw = response.read(MAX_RESPONSE_BYTES + 1)
                    if len(raw) > MAX_RESPONSE_BYTES:
                        raise LocalAIError(
                            f"LocalAI response exceeded {MAX_RESPONSE_BYTES} bytes; "
                            "refusing to parse"
                        )
                    payload = _parse_json_body(raw)
                    usage = (
                        _normalise_usage(payload.get("usage"))
                        if isinstance(payload, dict)
                        else None
                    )
                    if on_usage is not None:
                        on_usage(
                            usage
                            if usage is not None
                            else {
                                "prompt_tokens": estimated_prompt_tokens,
                                "completion_tokens": 0,
                                "total_tokens": estimated_prompt_tokens,
                                "estimated": True,
                            }
                        )
                    return _extract_completion_message(payload)
                return _read_streaming_completion(
                    response,
                    on_delta,
                    on_usage=on_usage,
                    estimated_prompt_tokens=estimated_prompt_tokens,
                )
        except urllib.error.HTTPError as exc:
            exc.close()
            raise LocalAIError(f"POST /v1/chat/completions returned HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise LocalAIError(f"cannot reach LocalAI at {self.base_url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise LocalAIError(
                f"LocalAI at {self.base_url} did not respond within {self.inference_timeout}s"
            ) from exc
        except OSError as exc:
            raise LocalAIError(f"network error talking to {self.base_url}: {exc}") from exc


def _parse_json_body(raw: bytes) -> Any:
    if not raw:
        raise LocalAIError("LocalAI returned an empty chat completion response")
    try:
        return json.loads(raw.decode("utf-8", errors="replace"))
    except ValueError as exc:
        raise LocalAIError(f"LocalAI returned invalid JSON: {exc}") from exc


def _extract_completion_message(payload: Any) -> dict[str, Any]:
    try:
        message = payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LocalAIError("LocalAI returned an invalid chat completion shape") from exc
    if not isinstance(message, dict):
        raise LocalAIError("LocalAI returned a non-object chat message")
    return message


def _normalise_usage(value: Any) -> dict[str, int | bool] | None:
    if not isinstance(value, dict):
        return None
    prompt = value.get("prompt_tokens")
    completion = value.get("completion_tokens")
    total = value.get("total_tokens")
    if not all(
        isinstance(item, int) and not isinstance(item, bool) and item >= 0
        for item in (prompt, completion)
    ):
        return None
    if not isinstance(total, int) or isinstance(total, bool) or total < 0:
        total = prompt + completion
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "estimated": False,
    }


def _estimate_tokens(value: bytes | str) -> int:
    """Return a visibly approximate byte-based token count without a tokenizer."""
    data = value if isinstance(value, bytes) else value.encode("utf-8")
    return _estimate_byte_count(len(data))


def _estimate_byte_count(byte_count: int) -> int:
    return max(0, (byte_count + 3) // 4)


def _read_streaming_completion(
    response: Any,
    on_delta: Callable[[str], None],
    *,
    on_usage: Callable[[dict[str, int | bool]], None] | None = None,
    estimated_prompt_tokens: int = 0,
) -> dict[str, Any]:
    """Collect OpenAI-compatible SSE chunks and stream visible text to a callback."""
    content: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    completion_chars = 0
    latest_usage: dict[str, int | bool] = {
        "prompt_tokens": estimated_prompt_tokens,
        "completion_tokens": 0,
        "total_tokens": estimated_prompt_tokens,
        "estimated": True,
    }
    total_bytes = 0
    saw_done = False
    while True:
        line = response.readline(MAX_RESPONSE_BYTES + 1)
        if not line:
            break
        total_bytes += len(line)
        if total_bytes > MAX_RESPONSE_BYTES:
            raise LocalAIError(
                f"LocalAI response exceeded {MAX_RESPONSE_BYTES} bytes; refusing to parse"
            )
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            saw_done = True
            break
        if not data:
            continue
        try:
            chunk = json.loads(data.decode("utf-8", errors="replace"))
        except ValueError as exc:
            raise LocalAIError("LocalAI sent malformed streaming JSON") from exc
        usage = _normalise_usage(chunk.get("usage")) if isinstance(chunk, dict) else None
        if usage is not None:
            latest_usage = usage
            if on_usage is not None:
                on_usage(latest_usage)
        choices = chunk.get("choices") if isinstance(chunk, dict) else None
        if choices == [] or choices is None:
            continue
        try:
            delta = choices[0].get("delta", {})
        except (KeyError, IndexError, TypeError) as exc:
            raise LocalAIError("LocalAI sent an invalid streaming response shape") from exc
        text = delta.get("content") if isinstance(delta, dict) else None
        if isinstance(text, str):
            content.append(text)
            completion_chars += len(text.encode("utf-8"))
            on_delta(text)
        calls = delta.get("tool_calls", []) if isinstance(delta, dict) else []
        if isinstance(calls, list):
            for call in calls:
                if not isinstance(call, dict) or not isinstance(call.get("index"), int):
                    continue
                index = call["index"]
                target = tool_calls.setdefault(
                    index,
                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if isinstance(call.get("id"), str):
                    target["id"] += call["id"]
                function = call.get("function", {})
                if isinstance(function, dict):
                    if isinstance(function.get("name"), str):
                        target["function"]["name"] += function["name"]
                    if isinstance(function.get("arguments"), str):
                        target["function"]["arguments"] += function["arguments"]
                        completion_chars += len(function["arguments"].encode("utf-8"))
                    if isinstance(function.get("name"), str):
                        completion_chars += len(function["name"].encode("utf-8"))
        if on_usage is not None and bool(latest_usage.get("estimated", True)):
            completion_tokens = _estimate_byte_count(completion_chars)
            latest_usage = {
                "prompt_tokens": estimated_prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": estimated_prompt_tokens + completion_tokens,
                "estimated": True,
            }
            on_usage(latest_usage)
    if not saw_done:
        raise LocalAIError("LocalAI streaming response ended before [DONE]")
    result: dict[str, Any] = {"role": "assistant", "content": "".join(content)}
    if tool_calls:
        result["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
    if on_usage is not None and bool(latest_usage.get("estimated", True)):
        completion_tokens = _estimate_byte_count(completion_chars)
        on_usage(
            {
                "prompt_tokens": estimated_prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": estimated_prompt_tokens + completion_tokens,
                "estimated": True,
            }
        )
    return result


def _parse_model_payload(payload: Any) -> list[ModelInfo]:
    """Extract models from an OpenAI-shaped ``/models`` body, defensively.

    Accepts ``{"data": [...]}`` and a bare list. Entries without a usable
    ``id`` are dropped rather than guessed at.
    """
    if payload is None:
        return []
    if isinstance(payload, dict):
        data = payload.get("data")
        if data is None:
            data = payload.get("models")
    else:
        data = payload
    if data is None:
        return []
    if not isinstance(data, list):
        raise LocalAIError(f"expected a list of models, got {type(data).__name__} from /v1/models")

    models: list[ModelInfo] = []
    for item in data:
        if isinstance(item, str):
            if item.strip():
                models.append(ModelInfo(id=item.strip()))
            continue
        if not isinstance(item, dict):
            continue
        identifier = item.get("id") or item.get("name")
        if not isinstance(identifier, str) or not identifier.strip():
            continue
        created = item.get("created")
        models.append(
            ModelInfo(
                id=identifier.strip(),
                object=str(item.get("object", "model")),
                owned_by=str(item["owned_by"]) if isinstance(item.get("owned_by"), str) else None,
                created=int(created) if isinstance(created, (int, float)) else None,
                raw_keys=tuple(sorted(str(key) for key in item)),
            )
        )
    return models
