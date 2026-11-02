"""Central audit logger and the ``@audited`` bridge decorator.

Every bridge call records: tool name, UTC timestamp, redacted arguments,
outcome, a short result summary and the duration in milliseconds.

Two hard rules hold everywhere in this module:

* File *contents* are never recorded -- only shapes and counts.
* Argument values whose name or shape suggests a secret are replaced with
  ``"[REDACTED]"`` before they reach any sink.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import json
import os
import re
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, TypeVar, runtime_checkable

from .permissions import (
    Capability,
    PermissionDenied,
    PermissionPolicy,
    default_permission_policy,
)

__all__ = [
    "AuditEntry",
    "AuditLogger",
    "SupportsAuditSummary",
    "audited",
    "get_audit_logger",
    "redact_arguments",
    "redact_text",
    "set_audit_logger",
    "summarize_value",
]

#: Argument / field names whose values are always redacted.
SENSITIVE_KEY_PATTERN = re.compile(
    r"(pass(word|wd|phrase)?|secret|token|api[_-]?key|access[_-]?key|"
    r"private[_-]?key|credential|session[_-]?id|cookie|auth(oriz\w+)?|"
    r"bearer|signature|salt|otp|pin)",
    re.IGNORECASE,
)

#: Value shapes that look like credentials regardless of the key name.
SENSITIVE_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bey[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}\b"),
)

REDACTED = "[REDACTED]"

#: Maximum characters kept for any single audited argument value.
MAX_ARGUMENT_CHARS = 256


@runtime_checkable
class SupportsAuditSummary(Protocol):
    """Objects that know how to describe themselves without leaking content."""

    def audit_summary(self) -> str: ...


def redact_text(text: str) -> str:
    """Replace credential-shaped substrings in ``text``."""
    for pattern in SENSITIVE_VALUE_PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


def _redact_value(key: str | None, value: Any, depth: int = 0) -> Any:
    if key is not None and SENSITIVE_KEY_PATTERN.search(key):
        return REDACTED
    if depth > 3:
        return f"<{type(value).__name__}>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, (str, os.PathLike)):
        text = redact_text(str(value))
        if len(text) > MAX_ARGUMENT_CHARS:
            text = f"{text[:MAX_ARGUMENT_CHARS]}...<truncated {len(text)} chars>"
        return text
    if isinstance(value, dict):
        return {str(k): _redact_value(str(k), v, depth + 1) for k, v in list(value.items())[:20]}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        redacted = [_redact_value(key, item, depth + 1) for item in items[:20]]
        if len(items) > 20:
            redacted.append(f"<+{len(items) - 20} more>")
        return redacted
    if is_dataclass(value) and not isinstance(value, type):
        return _redact_value(key, asdict(value), depth + 1)
    return f"<{type(value).__name__}>"


def redact_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``arguments`` safe to persist and display."""
    return {name: _redact_value(name, value) for name, value in arguments.items()}


def summarize_value(value: Any) -> str:
    """Describe a return value by shape, never by content."""
    if value is None:
        return "none"
    if isinstance(value, SupportsAuditSummary):
        return redact_text(value.audit_summary())
    if isinstance(value, bool):
        return f"bool={value}"
    if isinstance(value, (int, float)):
        return f"{type(value).__name__}={value}"
    if isinstance(value, str):
        return f"str<{len(value)} chars>"
    if isinstance(value, dict):
        keys = sorted(str(k) for k in list(value)[:12])
        return f"dict<{len(value)} keys: {', '.join(keys)}>"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}<{len(value)} items>"
    if is_dataclass(value) and not isinstance(value, type):
        return f"{type(value).__name__}<dataclass>"
    return f"<{type(value).__name__}>"


@dataclass(frozen=True)
class AuditEntry:
    """One immutable audit record."""

    tool_name: str
    timestamp: str
    arguments: dict[str, Any]
    status: str
    result_summary: str
    duration_ms: int
    capability: str | None = None
    task_id: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "timestamp": self.timestamp,
            "arguments": self.arguments,
            "status": self.status,
            "result_summary": self.result_summary,
            "duration_ms": self.duration_ms,
            "capability": self.capability,
            "task_id": self.task_id,
        }

    def as_line(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, default=str)

    def as_human(self) -> str:
        return (
            f"{self.timestamp}  {self.status.upper():<7} {self.tool_name:<28} "
            f"{self.duration_ms:>5}ms  {self.result_summary}"
        )


Sink = Callable[[AuditEntry], None]


@dataclass
class AuditLogger:
    """Collects audit entries in memory and fans them out to sinks.

    The in-memory ring buffer is always available (used by ``--json`` output
    and the tests). Additional sinks -- a JSONL file, the SQLite
    ``tool_executions`` table -- can be attached at runtime. A failing sink
    never breaks the audited call.
    """

    max_entries: int = 500
    sinks: list[Sink] = field(default_factory=list)
    _entries: deque[AuditEntry] = field(init=False, repr=False)
    sink_errors: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._entries = deque(maxlen=self.max_entries)

    # -- recording -------------------------------------------------------

    def record(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        status: str,
        result_summary: str,
        duration_ms: int,
        capability: Capability | None = None,
        task_id: int | None = None,
    ) -> AuditEntry:
        entry = AuditEntry(
            tool_name=tool_name,
            timestamp=datetime.now(UTC).isoformat(timespec="milliseconds"),
            arguments=redact_arguments(arguments),
            status=status,
            result_summary=redact_text(result_summary)[:400],
            duration_ms=duration_ms,
            capability=capability.value if capability else None,
            task_id=task_id,
        )
        self._entries.append(entry)
        for sink in list(self.sinks):
            try:
                sink(entry)
            except Exception as exc:  # noqa: BLE001 - a sink must never break a call
                self.sink_errors.append(f"{type(exc).__name__}: {exc}")
        return entry

    # -- reading ---------------------------------------------------------

    def entries(self, limit: int | None = None) -> list[AuditEntry]:
        items = list(self._entries)
        if limit is not None and limit >= 0:
            items = items[-limit:]
        return items

    def clear(self) -> None:
        self._entries.clear()
        self.sink_errors.clear()

    # -- sinks -----------------------------------------------------------

    def add_sink(self, sink: Sink) -> None:
        self.sinks.append(sink)

    def add_jsonl_file_sink(self, path: str | os.PathLike[str]) -> bool:
        """Append entries to ``path`` as JSON lines.

        The parent directory must already exist: this project never creates
        directories outside the project tree as a side effect.
        """
        target = Path(path)
        if not target.parent.is_dir():
            return False

        def sink(entry: AuditEntry) -> None:
            with open(target, "a", encoding="utf-8") as handle:
                handle.write(entry.as_line() + "\n")
            with contextlib.suppress(OSError):
                os.chmod(target, 0o600)

        self.add_sink(sink)
        return True


_LOGGER: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    """Return the process-wide audit logger, creating it on first use."""
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = AuditLogger()
    return _LOGGER


def set_audit_logger(logger: AuditLogger | None) -> None:
    """Replace the process-wide audit logger (used by tests and the CLI)."""
    global _LOGGER
    _LOGGER = logger


F = TypeVar("F", bound=Callable[..., Any])


def audited(
    tool_name: str,
    capability: Capability | None = None,
    *,
    policy: PermissionPolicy | None = None,
    redact_params: Sequence[str] | None = None,
) -> Callable[[F], F]:
    """Wrap a bridge function with a permission check, timing and auditing.

    The permission check runs *inside* the wrapper so a denial is itself
    audited with status ``denied``.
    """
    always_redact: frozenset[str] = frozenset(redact_params or ())

    def decorator(func: F) -> F:
        signature = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_audit_logger()
            effective_policy = policy or default_permission_policy()

            try:
                bound = signature.bind_partial(*args, **kwargs)
                bound.apply_defaults()
                raw_arguments = dict(bound.arguments)
            except TypeError:
                raw_arguments = {"<unbindable>": True}
            raw_arguments.pop("self", None)
            for name in always_redact:
                if name in raw_arguments:
                    raw_arguments[name] = REDACTED

            started = time.perf_counter()

            def elapsed_ms() -> int:
                return int((time.perf_counter() - started) * 1000)

            if capability is not None:
                try:
                    effective_policy.require(capability)
                except PermissionDenied as exc:
                    logger.record(
                        tool_name,
                        raw_arguments,
                        "denied",
                        f"{type(exc).__name__}: {exc.reason}",
                        elapsed_ms(),
                        capability,
                    )
                    raise

            try:
                result = func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - audited then re-raised
                logger.record(
                    tool_name,
                    raw_arguments,
                    "error",
                    f"{type(exc).__name__}: {exc}",
                    elapsed_ms(),
                    capability,
                )
                raise

            logger.record(
                tool_name,
                raw_arguments,
                "success",
                summarize_value(result),
                elapsed_ms(),
                capability,
            )
            return result

        return wrapper  # type: ignore[return-value]

    return decorator


def audit_entries_as_dicts(entries: Iterable[AuditEntry]) -> list[dict[str, Any]]:
    return [entry.as_dict() for entry in entries]
