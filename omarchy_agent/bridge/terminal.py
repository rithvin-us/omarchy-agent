"""Terminal command execution bridge.

Provides command execution inside the permitted workspace roots with explicit
timeouts and audited logging under :attr:`Capability.TERMINAL_EXEC`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..security.audit import audited, redact_text
from ..security.paths import PathPolicy, default_policy
from ..security.permissions import Capability

__all__ = [
    "CommandResult",
    "TerminalError",
    "execute_command",
]

MAX_OUTPUT_CHARS = 32_000


class TerminalError(RuntimeError):
    """Base exception for terminal execution failures."""


@dataclass(frozen=True)
class CommandResult:
    command: str
    cwd: str
    returncode: int
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "cwd": self.cwd,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_ms": self.duration_ms,
            "timed_out": self.timed_out,
        }

    def audit_summary(self) -> str:
        status = "timed out" if self.timed_out else f"exit {self.returncode}"
        return f"exec '{self.command[:60]}' in {self.cwd}: {status}"


@audited("terminal.execute", Capability.TERMINAL_EXEC)
def execute_command(
    command: str,
    *,
    cwd: str | os.PathLike[str] | None = None,
    timeout_seconds: float = 60.0,
    policy: PathPolicy | None = None,
) -> CommandResult:
    """Execute a shell command inside a permitted workspace directory."""
    clean_command = command.strip()
    if not clean_command:
        raise TerminalError("command string cannot be empty")

    active_policy = policy or default_policy()
    target_cwd = active_policy.resolve_within_roots(cwd or Path.cwd(), must_exist=True)
    if not target_cwd.is_dir():
        raise TerminalError(f"working directory is not a directory: {target_cwd}")

    shell = shutil.which("bash") or "/bin/sh"
    started = time.monotonic()
    timed_out = False
    returncode = -1
    stdout_text = ""
    stderr_text = ""

    try:
        completed = subprocess.run(  # noqa: S603 - audited command execution
            [shell, "-c", clean_command],
            cwd=str(target_cwd),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        returncode = completed.returncode
        stdout_text = completed.stdout or ""
        stderr_text = completed.stderr or ""
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        raw_stdout = exc.stdout or ""
        raw_stderr = exc.stderr or ""
        stdout_text = (
            raw_stdout if isinstance(raw_stdout, str) else raw_stdout.decode("utf-8", "replace")
        )
        stderr_text = (
            raw_stderr if isinstance(raw_stderr, str) else raw_stderr.decode("utf-8", "replace")
        )
    except OSError as exc:
        raise TerminalError(f"failed to spawn shell process: {exc}") from exc

    duration_ms = int((time.monotonic() - started) * 1000)

    # Bound output length and redact token patterns
    if len(stdout_text) > MAX_OUTPUT_CHARS:
        stdout_text = stdout_text[:MAX_OUTPUT_CHARS] + "\n... (truncated)"
    if len(stderr_text) > MAX_OUTPUT_CHARS:
        stderr_text = stderr_text[:MAX_OUTPUT_CHARS] + "\n... (truncated)"

    return CommandResult(
        command=clean_command,
        cwd=str(target_cwd),
        returncode=returncode,
        stdout=redact_text(stdout_text),
        stderr=redact_text(stderr_text),
        duration_ms=duration_ms,
        timed_out=timed_out,
    )
