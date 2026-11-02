"""Minimal LocalAI client.

Scope in v0.1: confirm the API answers and list the models it has installed.
Inference is intentionally *not* implemented -- the ``localai.inference``
capability is ``NOT_IMPLEMENTED`` in the permission policy, and
:meth:`LocalAIClient.chat_completion` refuses through that policy rather than
sending a request.

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
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
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
MAX_RESPONSE_BYTES = 1 * 1024 * 1024
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
    """A small, read-only client for a LocalAI instance on loopback.

    The public surface is deliberately narrow so that inference and tool
    calling can be added later as new methods guarded by the same policy,
    without changing how connectivity and model discovery work.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        read_timeout: float = DEFAULT_READ_TIMEOUT,
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
        request = urllib.request.Request(
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
        except (TimeoutError, socket.timeout) as exc:
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
            raise LocalAIError(
                f"model {preferred!r} is not installed; available: {available}"
            )
        return models[0].id

    # -- reserved for a later milestone ----------------------------------

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **_options: Any,
    ) -> Any:
        """Planned interface for inference and tool calling.

        This method exists so the integration boundary is fixed now: callers
        in a later milestone will not need to change shape. It refuses through
        the central permission policy, so enabling it is a one-line change in
        :mod:`omarchy_agent.security.permissions` plus the request body here.
        """
        self._policy.require(Capability.LOCALAI_INFERENCE)
        raise NotImplementedError(  # pragma: no cover - unreachable while denied
            "chat_completion is reserved for a later milestone"
        )


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
        raise LocalAIError(
            f"expected a list of models, got {type(data).__name__} from /v1/models"
        )

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
