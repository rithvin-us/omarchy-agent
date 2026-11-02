"""Central permission policy.

All capability decisions live here so that later milestones can add new
capabilities (terminal execution, desktop control, browser automation) by
flipping a single table entry instead of editing security checks scattered
across the codebase.

A capability is in exactly one of three states:

``ALLOWED``
    Implemented and permitted in this version.
``NOT_IMPLEMENTED``
    Planned for a later milestone. Calling it raises
    :class:`CapabilityNotImplemented`.
``DENIED``
    Deliberately forbidden by project policy, now and later.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "Capability",
    "CapabilityNotImplemented",
    "CapabilityState",
    "PermissionDenied",
    "PermissionPolicy",
    "default_permission_policy",
]


class CapabilityState(StrEnum):
    ALLOWED = "allowed"
    NOT_IMPLEMENTED = "not_implemented"
    DENIED = "denied"


class Capability(StrEnum):
    """Every action the agent can even ask about."""

    # --- implemented ---------------------------------------------------
    FS_READ = "filesystem.read"
    FS_LIST = "filesystem.list"
    FS_SEARCH = "filesystem.search"
    SYSTEM_INFO = "system.info"
    GIT_INSPECT = "git.inspect"
    LOCALAI_INSPECT = "localai.inspect"
    MEMORY_WRITE = "memory.write"
    MEMORY_READ = "memory.read"
    TASK_WRITE = "task.write"
    TASK_READ = "task.read"
    AUDIT_WRITE = "audit.write"

    # --- planned, not implemented -------------------------------------
    FS_WRITE = "filesystem.write"
    FS_DELETE = "filesystem.delete"
    TERMINAL_EXEC = "terminal.execute"
    LOCALAI_INFERENCE = "localai.inference"
    BROWSER_CONTROL = "browser.control"
    DESKTOP_INPUT = "desktop.input"
    DESKTOP_SCREENSHOT = "desktop.screenshot"
    APP_MANAGE = "application.manage"
    SCHEDULE_WRITE = "schedule.write"
    REMOTE_ACCESS = "remote.access"

    # --- forbidden by project policy -----------------------------------
    PRIVILEGE_ESCALATION = "system.privileged"
    PACKAGE_INSTALL = "system.package_install"
    DOCKER_CONTROL = "docker.control"
    SERVICE_INSTALL = "system.service_install"
    SECRET_READ = "secret.read"  # noqa: S105 - permission capability name
    NETWORK_BIND_PUBLIC = "network.bind_public"


_STATES: dict[Capability, CapabilityState] = {
    Capability.FS_READ: CapabilityState.ALLOWED,
    Capability.FS_LIST: CapabilityState.ALLOWED,
    Capability.FS_SEARCH: CapabilityState.ALLOWED,
    Capability.SYSTEM_INFO: CapabilityState.ALLOWED,
    Capability.GIT_INSPECT: CapabilityState.ALLOWED,
    Capability.LOCALAI_INSPECT: CapabilityState.ALLOWED,
    Capability.MEMORY_WRITE: CapabilityState.ALLOWED,
    Capability.MEMORY_READ: CapabilityState.ALLOWED,
    Capability.TASK_WRITE: CapabilityState.ALLOWED,
    Capability.TASK_READ: CapabilityState.ALLOWED,
    Capability.AUDIT_WRITE: CapabilityState.ALLOWED,
    Capability.FS_WRITE: CapabilityState.NOT_IMPLEMENTED,
    Capability.FS_DELETE: CapabilityState.NOT_IMPLEMENTED,
    Capability.TERMINAL_EXEC: CapabilityState.NOT_IMPLEMENTED,
    Capability.LOCALAI_INFERENCE: CapabilityState.ALLOWED,
    Capability.BROWSER_CONTROL: CapabilityState.NOT_IMPLEMENTED,
    Capability.DESKTOP_INPUT: CapabilityState.NOT_IMPLEMENTED,
    Capability.DESKTOP_SCREENSHOT: CapabilityState.NOT_IMPLEMENTED,
    Capability.APP_MANAGE: CapabilityState.NOT_IMPLEMENTED,
    Capability.SCHEDULE_WRITE: CapabilityState.NOT_IMPLEMENTED,
    Capability.REMOTE_ACCESS: CapabilityState.NOT_IMPLEMENTED,
    Capability.PRIVILEGE_ESCALATION: CapabilityState.DENIED,
    Capability.PACKAGE_INSTALL: CapabilityState.DENIED,
    Capability.DOCKER_CONTROL: CapabilityState.DENIED,
    Capability.SERVICE_INSTALL: CapabilityState.DENIED,
    Capability.SECRET_READ: CapabilityState.DENIED,
    Capability.NETWORK_BIND_PUBLIC: CapabilityState.DENIED,
}

_NOTES: dict[Capability, str] = {
    Capability.FS_READ: "Read text files inside the permitted workspace roots.",
    Capability.FS_LIST: "List directory entries inside the permitted roots.",
    Capability.FS_SEARCH: "Search file names and text content inside the roots.",
    Capability.SYSTEM_INFO: "Read-only OS, CPU, memory, storage, GPU and process info.",
    Capability.GIT_INSPECT: "Read-only git status, branch, log and diff statistics.",
    Capability.LOCALAI_INSPECT: "Query the local LocalAI API for health and model list.",
    Capability.MEMORY_WRITE: "Create and update the agent's own memory records.",
    Capability.MEMORY_READ: "Read the agent's own memory records.",
    Capability.TASK_WRITE: "Create and update the agent's own task records.",
    Capability.TASK_READ: "Read the agent's own task records.",
    Capability.AUDIT_WRITE: "Append tool-execution metadata to the audit log.",
    Capability.FS_WRITE: "Planned: create or modify workspace files.",
    Capability.FS_DELETE: "Planned: delete workspace files.",
    Capability.TERMINAL_EXEC: (
        "Planned: run an allowlisted command. Never a model-supplied shell string."
    ),
    Capability.LOCALAI_INFERENCE: "Run local chat completions and validated read-only tools.",
    Capability.BROWSER_CONTROL: "Planned: browser automation.",
    Capability.DESKTOP_INPUT: "Planned: mouse and keyboard control via Wayland tooling.",
    Capability.DESKTOP_SCREENSHOT: "Planned: screen capture via grim/slurp.",
    Capability.APP_MANAGE: "Planned: launch and focus desktop applications.",
    Capability.SCHEDULE_WRITE: "Planned: user-initiated scheduled workflows.",
    Capability.REMOTE_ACCESS: "Planned: access from a phone over a private network only.",
    Capability.PRIVILEGE_ESCALATION: "Forbidden: no sudo, no privileged system changes.",
    Capability.PACKAGE_INSTALL: "Forbidden: no pacman, no global package installation.",
    Capability.DOCKER_CONTROL: (
        "Forbidden: never start, stop or reconfigure containers; never touch the Docker socket."
    ),
    Capability.SERVICE_INSTALL: "Forbidden: no systemd units, cron jobs or background daemons.",
    Capability.SECRET_READ: "Forbidden: no SSH keys, credential stores or password databases.",
    Capability.NETWORK_BIND_PUBLIC: "Forbidden: never bind a listener to 0.0.0.0.",
}


class PermissionDenied(PermissionError):
    """Raised when a capability is forbidden by policy."""

    def __init__(self, capability: Capability, reason: str) -> None:
        super().__init__(f"{capability.value}: {reason}")
        self.capability = capability
        self.reason = reason


class CapabilityNotImplemented(PermissionDenied):
    """Raised when a capability is planned but not available in this version."""


@dataclass(frozen=True)
class CapabilityInfo:
    capability: Capability
    state: CapabilityState
    note: str

    def as_dict(self) -> dict[str, str]:
        return {
            "capability": self.capability.value,
            "state": self.state.value,
            "note": self.note,
        }


@dataclass(frozen=True)
class PermissionPolicy:
    """The capability table. Frozen; build a new one to change a decision."""

    states: dict[Capability, CapabilityState]

    def state_of(self, capability: Capability) -> CapabilityState:
        try:
            return self.states[capability]
        except KeyError:
            # Unknown capability: fail closed.
            return CapabilityState.DENIED

    def is_allowed(self, capability: Capability) -> bool:
        return self.state_of(capability) is CapabilityState.ALLOWED

    def require(self, capability: Capability) -> None:
        """Assert ``capability`` is permitted, else raise."""
        state = self.state_of(capability)
        if state is CapabilityState.ALLOWED:
            return
        note = _NOTES.get(capability, "no description available")
        if state is CapabilityState.NOT_IMPLEMENTED:
            raise CapabilityNotImplemented(capability, f"not implemented in this version -- {note}")
        raise PermissionDenied(capability, f"denied by project policy -- {note}")

    def inventory(self) -> list[CapabilityInfo]:
        """Return every capability with its state, for the CLI and docs."""
        return [
            CapabilityInfo(
                capability=capability,
                state=self.state_of(capability),
                note=_NOTES.get(capability, ""),
            )
            for capability in Capability
        ]

    def grouped_inventory(self) -> dict[str, list[dict[str, str]]]:
        grouped: dict[str, list[dict[str, str]]] = {state.value: [] for state in CapabilityState}
        for info in self.inventory():
            grouped[info.state.value].append(info.as_dict())
        return grouped


_DEFAULT = PermissionPolicy(states=dict(_STATES))


def default_permission_policy() -> PermissionPolicy:
    """Return the process-wide permission policy."""
    return _DEFAULT
