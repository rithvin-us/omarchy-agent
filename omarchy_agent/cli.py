"""Command-line interface for the Omarchy Agent.

Every command is read-only with two explicit exceptions, both of which say so
before they act:

* ``init`` creates the data directory and database (never overwrites one);
* ``tasks --create`` / ``memory --remember`` write the agent's own records.

Nothing here executes shell commands on behalf of a model, writes outside the
project tree and the agent's own data directory, or touches Docker.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

from . import __version__
from .bridge import filesystem as fs_bridge
from .bridge import git as git_bridge
from .bridge import system as system_bridge
from .core.task_manager import TaskManager, initialize_storage
from .integrations.localai import DEFAULT_BASE_URL, LocalAIClient, LocalAIError
from .memory.database import (
    SCHEMA_VERSION,
    Database,
    DatabaseError,
    DatabaseNotInitialized,
    default_data_directory,
    default_database_path,
)
from .security.audit import get_audit_logger
from .security.paths import PathPolicy, PathSecurityError, default_policy
from .security.permissions import (
    CapabilityState,
    PermissionDenied,
    default_permission_policy,
)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NOT_INITIALIZED = 2
EXIT_CHECK_FAILED = 3

#: External tools the agent may *use* if present. It never installs any of them.
OPTIONAL_TOOLS: tuple[tuple[str, str], ...] = (
    ("git", "repository inspection (implemented)"),
    ("uv", "python environment management (developer convenience)"),
    ("nvidia-smi", "GPU information (implemented, optional)"),
    ("hyprctl", "Hyprland introspection (planned milestone)"),
    ("grim", "screen capture (planned milestone)"),
    ("slurp", "region selection (planned milestone)"),
    ("wtype", "keyboard synthesis (planned milestone)"),
    ("docker", "present on the host; this agent never invokes it"),
)


# --------------------------------------------------------------------------
# formatting helpers
# --------------------------------------------------------------------------


def human_bytes(value: int | None) -> str:
    if value is None:
        return "-"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(size)} B"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TiB"  # pragma: no cover - loop always returns


def emit(data: Any, as_json: bool, renderer: Callable[[Any], None]) -> None:
    if as_json:
        print(json.dumps(data, indent=2, sort_keys=True, default=str))
    else:
        renderer(data)


def heading(text: str) -> None:
    print(f"\n{text}")
    print("-" * len(text))


def field(label: str, value: Any, width: int = 24) -> None:
    print(f"  {label:<{width}} {value}")


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def collect_environment() -> dict[str, Any]:
    """Gather safe environment facts. No environment variables are reported."""
    os_information = system_bridge.get_os_information()
    policy = default_policy()
    database_path = default_database_path()
    project_root = Path(__file__).resolve().parent.parent

    tools: list[dict[str, Any]] = []
    for name, purpose in OPTIONAL_TOOLS:
        location = shutil.which(name)
        tools.append(
            {
                "name": name,
                "available": location is not None,
                "path": location,
                "purpose": purpose,
            }
        )

    return {
        "agent_version": __version__,
        "python": {
            "version": os_information.python_version,
            "executable": os_information.python_executable,
            "implementation": sys.implementation.name,
        },
        "host": {
            "distribution": os_information.distribution,
            "distribution_version": os_information.distribution_version,
            "kernel_release": os_information.kernel_release,
            "architecture": os_information.architecture,
            "hostname": os_information.hostname,
            "desktop_session": os_information.desktop_session,
            "display_server": os_information.display_server,
            "is_omarchy": os_information.is_omarchy,
        },
        "tools": tools,
        "workspace_roots": policy.describe_roots(),
        "max_file_read_bytes": policy.max_file_bytes,
        "project_root": str(project_root),
        "database": {
            "path": str(database_path),
            "exists": database_path.exists(),
            "data_directory": str(default_data_directory()),
            "data_directory_exists": default_data_directory().is_dir(),
            "expected_schema_version": SCHEMA_VERSION,
        },
        "localai": {"base_url": DEFAULT_BASE_URL},
    }


def command_audit(args: argparse.Namespace) -> int:
    data = collect_environment()

    def render(payload: dict[str, Any]) -> None:
        heading(f"Omarchy Agent {payload['agent_version']} -- environment audit")
        field("python", f"{payload['python']['version']} ({payload['python']['executable']})")
        field("distribution", f"{payload['host']['distribution']} {payload['host']['distribution_version']}")
        field("kernel", payload["host"]["kernel_release"])
        field("architecture", payload["host"]["architecture"])
        field("session", f"{payload['host']['desktop_session']} / {payload['host']['display_server']}")

        heading("Tool availability (never installed by this agent)")
        for tool in payload["tools"]:
            mark = "yes" if tool["available"] else "no "
            field(f"[{mark}] {tool['name']}", tool["path"] or tool["purpose"], width=22)

        heading("Workspace policy")
        for root in payload["workspace_roots"]:
            field("permitted root", root)
        if not payload["workspace_roots"]:
            field("permitted root", "(none resolved -- ~/Projects missing?)")
        field("max file read", human_bytes(payload["max_file_read_bytes"]))

        heading("Persistent storage")
        field("database path", payload["database"]["path"])
        field("database exists", payload["database"]["exists"])
        field("data directory", payload["database"]["data_directory"])
        if not payload["database"]["exists"]:
            field("", "run 'python -m omarchy_agent init' to create it")

        heading("LocalAI")
        field("endpoint", payload["localai"]["base_url"])
        print()

    emit(data, args.json, render)
    return EXIT_OK


def command_capabilities(args: argparse.Namespace) -> int:
    policy = default_permission_policy()
    data = policy.grouped_inventory()

    def render(payload: dict[str, list[dict[str, str]]]) -> None:
        labels = {
            CapabilityState.ALLOWED.value: "Allowed in v0.1",
            CapabilityState.NOT_IMPLEMENTED.value: "Planned -- not implemented",
            CapabilityState.DENIED.value: "Denied by project policy",
        }
        for state, label in labels.items():
            heading(label)
            for item in payload.get(state, []):
                field(item["capability"], item["note"], width=28)
        print()

    emit(data, args.json, render)
    return EXIT_OK


def command_init(args: argparse.Namespace) -> int:
    target = Path(args.database).expanduser() if args.database else default_database_path()

    if not args.json:
        heading("Explicit storage initialization")
        print("  This command will create, if they do not already exist:")
        print(f"    directory : {target.parent}  (mode 0700)")
        print(f"    database  : {target}  (mode 0600)")
        print("  An existing database is opened and migrated forward only.")
        print("  Nothing else on the host is touched.")

    try:
        report = initialize_storage(target)
    except (DatabaseError, OSError) as exc:
        print(f"error: initialization failed: {exc}", file=sys.stderr)
        return EXIT_ERROR

    def render(payload: dict[str, Any]) -> None:
        heading("Result")
        field("database", payload["database_path"])
        field("directory created", payload["directory_created"])
        field("database created", payload["database_created"])
        field("already existed", payload["already_existed"])
        field("schema version", payload["schema_version"])
        field("integrity check", payload["integrity"])
        field("tables", ", ".join(payload["tables"]))
        if payload["already_existed"]:
            field("", "existing data was preserved")
        print()

    emit(report.as_dict(), args.json, render)
    return EXIT_OK


def command_system(args: argparse.Namespace) -> int:
    data = {
        "os": system_bridge.get_os_information().as_dict(),
        "cpu": system_bridge.get_cpu_information().as_dict(),
        "memory": system_bridge.get_memory_information().as_dict(),
        "storage": system_bridge.get_storage_information().as_dict(),
        "gpu": system_bridge.get_gpu_information().as_dict(),
        "processes": system_bridge.get_running_process_summary(top=args.top).as_dict(),
    }

    def render(payload: dict[str, Any]) -> None:
        operating_system = payload["os"]
        heading("Operating system")
        field("distribution", f"{operating_system['distribution']} {operating_system['distribution_version']}")
        field("kernel", f"{operating_system['kernel_name']} {operating_system['kernel_release']}")
        field("architecture", operating_system["architecture"])
        field("hostname", operating_system["hostname"])
        if operating_system["uptime_seconds"]:
            field("uptime", f"{operating_system['uptime_seconds'] / 3600:.1f} h")

        cpu = payload["cpu"]
        heading("CPU")
        field("model", cpu["model"])
        field("cores", f"{cpu['physical_cores']} physical / {cpu['logical_cores']} logical")
        if cpu["current_frequency_mhz"]:
            field("frequency", f"{cpu['current_frequency_mhz']:.0f} MHz avg")
        if cpu["max_frequency_mhz"]:
            field("max frequency", f"{cpu['max_frequency_mhz']:.0f} MHz")
        if cpu["load_average"]:
            field("load average", ", ".join(f"{value:.2f}" for value in cpu["load_average"]))
        if cpu["temperature_celsius"] is not None:
            field("temperature", f"{cpu['temperature_celsius']:.1f} C")

        memory = payload["memory"]
        heading("Memory")
        field("total", human_bytes(memory["total_bytes"]))
        field("used", f"{human_bytes(memory['used_bytes'])} ({memory['used_percent']:.1f}%)")
        field("available", human_bytes(memory["available_bytes"]))
        field("cached", human_bytes(memory["cached_bytes"]))
        if memory["swap_total_bytes"]:
            field(
                "swap",
                f"{human_bytes(memory['swap_used_bytes'])} / "
                f"{human_bytes(memory['swap_total_bytes'])} "
                f"({memory['swap_used_percent']:.1f}%)",
            )
        else:
            field("swap", "none configured")

        heading("Storage")
        for volume in payload["storage"]["volumes"]:
            field(
                volume["mountpoint"],
                f"{human_bytes(volume['used_bytes'])} / {human_bytes(volume['total_bytes'])} "
                f"({volume['used_percent']:.1f}%) {volume['filesystem']}"
                f"{' ro' if volume['read_only'] else ''}",
                width=28,
            )
        if not payload["storage"]["volumes"]:
            field("", "no real volumes detected")

        gpu = payload["gpu"]
        heading("GPU")
        if not gpu["available"]:
            field("status", gpu["detail"] or "unavailable")
        for device in gpu["devices"]:
            field(f"gpu {device['index']}", device["name"])
            field("  driver", device["driver_version"])
            if device["memory_total_mib"] is not None:
                field(
                    "  vram",
                    f"{device['memory_used_mib']} / {device['memory_total_mib']} MiB",
                )
            if device["utilization_percent"] is not None:
                field("  utilization", f"{device['utilization_percent']}%")
            if device["temperature_celsius"] is not None:
                field("  temperature", f"{device['temperature_celsius']} C")

        processes = payload["processes"]
        heading("Processes")
        field("total", processes["total"])
        field("threads", processes["thread_total"])
        field("by state", ", ".join(f"{k}={v}" for k, v in processes["by_state"].items()))
        print(f"\n  top {len(processes['top_by_memory'])} by resident memory:")
        for item in processes["top_by_memory"]:
            print(f"    {item['pid']:>8}  {human_bytes(item['resident_bytes']):>10}  {item['name']}")
        print(f"\n  note: {processes['note']}")
        print()

    emit(data, args.json, render)
    return EXIT_OK


def command_models(args: argparse.Namespace) -> int:
    try:
        client = LocalAIClient(args.endpoint)
    except LocalAIError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    status = client.check_connection()
    models: list[dict[str, Any]] = []
    error: str | None = None
    if status.reachable:
        try:
            models = [model.as_dict() for model in client.list_models()]
        except LocalAIError as exc:
            error = str(exc)

    data = {"connection": status.as_dict(), "models": models, "error": error}

    def render(payload: dict[str, Any]) -> None:
        connection = payload["connection"]
        heading("LocalAI")
        field("endpoint", connection["base_url"])
        field("reachable", connection["reachable"])
        field("status code", connection["status_code"])
        field("latency", f"{connection['latency_ms']} ms" if connection["latency_ms"] is not None else "-")
        if connection["error"]:
            field("error", connection["error"])
        if connection["hint"]:
            field("hint", connection["hint"])

        heading(f"Installed models ({len(payload['models'])})")
        if not payload["models"]:
            field("", "none reported by the server")
        for model in payload["models"]:
            field(model["id"], model["owned_by"] or "", width=40)
        if payload["error"]:
            field("error", payload["error"])
        print()

    emit(data, args.json, render)
    if not status.reachable:
        return EXIT_CHECK_FAILED
    return EXIT_OK


def _open_manager(args: argparse.Namespace, *, mirror_audit: bool = False) -> TaskManager:
    """Open the store for a command. ``create=False``: never creates anything.

    ``mirror_audit`` attaches the SQLite sink, so the audited calls this command
    makes are persisted as ``tool_executions`` rows and survive the process.
    """
    path = Path(args.database).expanduser() if args.database else None
    return TaskManager.open(path, create=False, mirror_audit_to_database=mirror_audit)


def command_tasks(args: argparse.Namespace) -> int:
    try:
        manager = _open_manager(args, mirror_audit=True)
    except DatabaseNotInitialized as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_NOT_INITIALIZED
    except DatabaseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    with manager:
        try:
            if args.create:
                created = manager.create_task(args.create, args.description or "")
                if not args.json:
                    print(f"created task {created.id}: {created.title}")
            if args.complete is not None:
                completed = manager.complete_task(args.complete)
                if not args.json:
                    print(f"task {completed.id} marked {completed.status}")

            tasks = manager.list_tasks(status=args.status, limit=args.limit)
            counts = manager.task_counts()
        except DatabaseError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_ERROR

        data = {
            "tasks": [task.as_dict() for task in tasks],
            "counts": counts,
            "database": str(manager.database.path),
        }

    def render(payload: dict[str, Any]) -> None:
        heading(f"Tasks ({len(payload['tasks'])} shown)")
        if not payload["tasks"]:
            field("", "no tasks recorded")
        for task in payload["tasks"]:
            print(f"  #{task['id']:<5} {task['status']:<12} {task['title']}")
            if task["description"]:
                print(f"         {task['description'].splitlines()[0][:90]}")
        heading("Counts by status")
        for status, total in sorted(payload["counts"].items()):
            field(status, total)
        if not payload["counts"]:
            field("", "none")
        print()

    emit(data, args.json, render)
    return EXIT_OK


def command_memory(args: argparse.Namespace) -> int:
    try:
        manager = _open_manager(args, mirror_audit=True)
    except DatabaseNotInitialized as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_NOT_INITIALIZED
    except DatabaseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    with manager:
        try:
            if args.remember:
                category, key, content = args.remember
                record = manager.remember(category, key, content, tags=args.tags or [])
                if not args.json:
                    print(f"stored memory {record.category}/{record.key} (id {record.id})")
            if args.search:
                records = manager.search_memories(args.search, limit=args.limit)
            else:
                records = manager.list_memories(
                    category=args.category, tag=args.tag, limit=args.limit
                )
        except DatabaseError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_ERROR

        data = {
            "memories": [record.as_dict() for record in records],
            "database": str(manager.database.path),
        }

    def render(payload: dict[str, Any]) -> None:
        heading(f"Memories ({len(payload['memories'])} shown)")
        if not payload["memories"]:
            field("", "no memories recorded")
        for record in payload["memories"]:
            tags = f" [{', '.join(record['tags'])}]" if record["tags"] else ""
            print(f"  #{record['id']:<5} {record['category']}/{record['key']}{tags}")
            first_line = record["content"].splitlines()[0] if record["content"] else ""
            print(f"         {first_line[:100]}")
            print(f"         updated {record['updated_at']}")
        print()

    emit(data, args.json, render)
    return EXIT_OK


def command_doctor(args: argparse.Namespace) -> int:
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str, severity: str = "error") -> None:
        checks.append(
            {"name": name, "ok": ok, "detail": detail, "severity": "ok" if ok else severity}
        )

    # 1. package structure
    package_root = Path(__file__).resolve().parent
    expected_modules = [
        "__main__.py",
        "cli.py",
        "core/task_manager.py",
        "bridge/filesystem.py",
        "bridge/system.py",
        "bridge/git.py",
        "memory/database.py",
        "integrations/localai.py",
        "security/paths.py",
        "security/permissions.py",
        "security/audit.py",
    ]
    missing = [name for name in expected_modules if not (package_root / name).is_file()]
    check(
        "package structure",
        not missing,
        "all modules present" if not missing else f"missing: {', '.join(missing)}",
    )

    # 2. workspace policy
    policy = default_policy()
    check(
        "workspace roots",
        bool(policy.roots),
        ", ".join(policy.describe_roots()) if policy.roots
        else "no permitted root resolved; ~/Projects not found",
    )

    # 3. filesystem bridge sanity against its own project directory
    project_root = package_root.parent
    if policy.is_within_roots(project_root):
        try:
            listing = fs_bridge.list_directory(project_root)
            check(
                "filesystem bridge",
                True,
                f"listed {len(listing.entries)} entries in {project_root}",
            )
        except (PathSecurityError, fs_bridge.FilesystemError, OSError) as exc:
            check("filesystem bridge", False, f"{type(exc).__name__}: {exc}")
    else:
        check(
            "filesystem bridge",
            True,
            f"skipped: {project_root} is outside the permitted roots",
            severity="warning",
        )

    # 4. traversal refusal still works
    try:
        fs_bridge.read_file("/etc/shadow")
        check("path traversal refusal", False, "reading /etc/shadow was NOT refused")
    except PathSecurityError:
        check("path traversal refusal", True, "paths outside the workspace are refused")
    except Exception as exc:  # noqa: BLE001
        check("path traversal refusal", True, f"refused with {type(exc).__name__}")

    # 5. permission policy
    permissions = default_permission_policy()
    try:
        permissions.require(_denied_probe())
        check("permission policy", False, "a denied capability was permitted")
    except PermissionDenied:
        inventory = permissions.grouped_inventory()
        check(
            "permission policy",
            True,
            f"{len(inventory['allowed'])} allowed, "
            f"{len(inventory['not_implemented'])} planned, "
            f"{len(inventory['denied'])} denied",
        )

    # 6. database
    database_path = Path(args.database).expanduser() if args.database else default_database_path()
    if not database_path.exists():
        check(
            "database",
            False,
            f"not initialized at {database_path}; run 'python -m omarchy_agent init'",
            severity="warning",
        )
        check("database permissions", False, "skipped: database missing", severity="warning")
    else:
        try:
            with Database.open(database_path, create=False) as database:
                info = database.info()
                integrity = database.integrity_check()
            check(
                "database",
                integrity == "ok" and info["schema_version"] == SCHEMA_VERSION,
                f"schema v{info['schema_version']} (expected v{SCHEMA_VERSION}), "
                f"integrity={integrity}, tables={len(info['tables'])}, "
                f"rows={sum(info['row_counts'].values())}",
            )
            mode = info["permissions"]
            check(
                "database permissions",
                mode in {"0o600", "0o640", "0o644"},
                f"file mode {mode} (0o600 expected)",
                severity="warning",
            )
        except DatabaseError as exc:
            check("database", False, f"{type(exc).__name__}: {exc}")
            check("database permissions", False, "skipped", severity="warning")

    # 7. git availability
    check(
        "git",
        git_bridge.git_available(),
        "git found on PATH" if git_bridge.git_available() else "git not installed",
        severity="warning",
    )

    # 8. LocalAI
    try:
        client = LocalAIClient(args.endpoint)
        status = client.check_connection()
        check(
            "localai api",
            status.reachable,
            f"HTTP {status.status_code} in {status.latency_ms} ms at {status.base_url}"
            if status.reachable
            else (status.error or "unreachable"),
        )
        check(
            "localai models",
            bool(status.models_ready),
            f"{status.model_count} model(s) installed"
            if status.models_ready
            else (status.hint or "no models installed"),
            severity="warning",
        )
    except LocalAIError as exc:
        check("localai api", False, f"{type(exc).__name__}: {exc}")
        check("localai models", False, "skipped", severity="warning")

    errors = [item for item in checks if not item["ok"] and item["severity"] == "error"]
    warnings = [item for item in checks if not item["ok"] and item["severity"] == "warning"]
    data = {
        "checks": checks,
        "errors": len(errors),
        "warnings": len(warnings),
        "passed": len([item for item in checks if item["ok"]]),
    }

    def render(payload: dict[str, Any]) -> None:
        heading("Doctor")
        for item in payload["checks"]:
            marker = "PASS" if item["ok"] else ("WARN" if item["severity"] == "warning" else "FAIL")
            print(f"  [{marker}] {item['name']:<24} {item['detail']}")
        print(
            f"\n  {payload['passed']} passed, {payload['warnings']} warning(s), "
            f"{payload['errors']} error(s)\n"
        )

    emit(data, args.json, render)
    return EXIT_CHECK_FAILED if errors else EXIT_OK


def _denied_probe():
    from .security.permissions import Capability

    return Capability.PRIVILEGE_ESCALATION


def command_files(args: argparse.Namespace) -> int:
    policy = default_policy() if not args.root else PathPolicy.from_paths([args.root])
    try:
        if args.action == "list":
            result: Any = fs_bridge.list_directory(args.path, policy=policy).as_dict()
        elif args.action == "read":
            result = fs_bridge.read_file(args.path, policy=policy).as_dict()
        elif args.action == "info":
            result = fs_bridge.get_file_info(args.path, policy=policy).as_dict()
        else:
            if not args.query:
                print("error: 'files search' requires --query", file=sys.stderr)
                return EXIT_ERROR
            result = fs_bridge.search_files(
                args.query, args.path, policy=policy, max_results=args.limit
            ).as_dict()
    except (PathSecurityError, fs_bridge.FilesystemError, PermissionDenied, OSError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR

    def render(payload: dict[str, Any]) -> None:
        if args.action == "read":
            heading(payload["path"])
            field("size", human_bytes(payload["size_bytes"]))
            field("lines", payload["line_count"])
            field("truncated", payload["truncated"])
            print()
            print(payload["content"])
        elif args.action == "list":
            heading(f"{payload['path']} ({len(payload['entries'])} entries)")
            for entry in payload["entries"]:
                note = f"  <- {entry['skip_reason']}" if entry["skip_reason"] else ""
                print(
                    f"  {entry['kind']:<10} {human_bytes(entry['size_bytes']):>10}  "
                    f"{entry['name']}{note}"
                )
            print()
        elif args.action == "info":
            heading(payload["path"])
            for key, value in payload.items():
                if key != "path":
                    field(key, value)
            print()
        else:
            heading(f"{len(payload['matches'])} match(es) for {payload['query']!r}")
            for match in payload["matches"]:
                where = f":{match['line_number']}" if match["line_number"] else ""
                print(f"  {match['path']}{where}")
                if match["line"]:
                    print(f"      {match['line']}")
            field("files scanned", payload["files_scanned"])
            field("timed out", payload["timed_out"])
            print()

    emit(result, args.json, render)
    return EXIT_OK


def command_git(args: argparse.Namespace) -> int:
    try:
        if args.action == "status":
            result: Any = git_bridge.get_repository_status(args.path).as_dict()
        elif args.action == "branch":
            result = git_bridge.get_current_branch(args.path).as_dict()
        elif args.action == "log":
            result = [commit.as_dict() for commit in git_bridge.get_recent_commits(args.path, limit=args.limit)]
        else:
            result = git_bridge.get_diff_summary(args.path, staged=args.staged).as_dict()
    except (PathSecurityError, git_bridge.GitError, PermissionDenied, OSError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR

    def render(payload: Any) -> None:
        if args.action == "log":
            heading(f"Recent commits ({len(payload)})")
            for commit in payload:
                print(f"  {commit['short_hash']}  {commit['date'][:19]}  {commit['author']}")
                print(f"            {commit['subject']}")
            print()
            return
        heading(f"git {args.action}")
        for key, value in payload.items():
            if key in {"entries", "files"}:
                continue
            field(key, value)
        for entry in payload.get("entries", []):
            print(f"    {entry['index_status']}{entry['worktree_status']}  {entry['path']}")
        for stat in payload.get("files", []):
            print(f"    +{stat['insertions']}/-{stat['deletions']}  {stat['path']}")
        print()

    emit(result, args.json, render)
    return EXIT_OK


def command_audit_log(args: argparse.Namespace) -> int:
    """Show audit rows persisted in the database."""
    try:
        manager = _open_manager(args)
    except DatabaseNotInitialized as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_NOT_INITIALIZED

    with manager:
        executions = manager.list_tool_executions(tool_name=args.tool, limit=args.limit)
        data = {"tool_executions": [item.as_dict() for item in executions]}

    def render(payload: dict[str, Any]) -> None:
        heading(f"Persisted tool executions ({len(payload['tool_executions'])})")
        if not payload["tool_executions"]:
            field("", "none recorded yet")
        for item in payload["tool_executions"]:
            print(
                f"  {item['created_at']}  {item['status']:<8} {item['tool_name']:<32} "
                f"{item['duration_ms']:>5}ms  {item['result_summary'][:70]}"
            )
        print()

    emit(data, args.json, render)
    return EXIT_OK


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omarchy-agent",
        description=(
            "Local, read-only inspection foundation for an Omarchy Linux AI agent. "
            "v0.1 can look at the workspace, the host and LocalAI, and can keep its "
            "own notes. It cannot run commands, change files or control the desktop."
        ),
        epilog="Run 'omarchy-agent doctor' first; 'omarchy-agent init' creates the database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"omarchy-agent {__version__}")
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON instead of text"
    )
    parser.add_argument(
        "--database",
        metavar="PATH",
        default=None,
        help=f"override the database path (default: {default_database_path()})",
    )
    parser.add_argument(
        "--endpoint",
        metavar="URL",
        default=DEFAULT_BASE_URL,
        help=f"LocalAI base URL, loopback only (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--show-audit",
        action="store_true",
        help="print this run's in-memory audit entries when the command finishes",
    )
    parser.add_argument(
        "--persist-audit",
        action="store_true",
        help=(
            "append this run's audit entries to the database as tool_executions "
            "rows (requires an initialised database; creates nothing)"
        ),
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    subparsers.add_parser(
        "audit", help="show safe environment information and tool availability"
    ).set_defaults(handler=command_audit)

    subparsers.add_parser(
        "doctor", help="verify structure, permissions, database and LocalAI connectivity"
    ).set_defaults(handler=command_doctor)

    subparsers.add_parser(
        "init", help="explicitly create the persistent database (never overwrites)"
    ).set_defaults(handler=command_init)

    system_parser = subparsers.add_parser(
        "system", help="show OS, CPU, memory, storage, GPU and process information"
    )
    system_parser.add_argument(
        "--top", type=int, default=10, help="processes to show by memory (default: 10)"
    )
    system_parser.set_defaults(handler=command_system)

    subparsers.add_parser("models", help="list models installed in LocalAI").set_defaults(
        handler=command_models
    )

    subparsers.add_parser(
        "capabilities", help="show the central permission policy"
    ).set_defaults(handler=command_capabilities)

    tasks_parser = subparsers.add_parser("tasks", help="list and manage persistent tasks")
    tasks_parser.add_argument("--status", default=None, help="filter by status")
    tasks_parser.add_argument("--limit", type=int, default=50, help="maximum rows (default: 50)")
    tasks_parser.add_argument("--create", metavar="TITLE", default=None, help="create a task")
    tasks_parser.add_argument(
        "--description", metavar="TEXT", default=None, help="description for --create"
    )
    tasks_parser.add_argument(
        "--complete", metavar="ID", type=int, default=None, help="mark a task done"
    )
    tasks_parser.set_defaults(handler=command_tasks)

    memory_parser = subparsers.add_parser("memory", help="list and manage stored memories")
    memory_parser.add_argument("--category", default=None, help="filter by category")
    memory_parser.add_argument("--tag", default=None, help="filter by tag")
    memory_parser.add_argument("--search", default=None, help="substring search")
    memory_parser.add_argument("--limit", type=int, default=50, help="maximum rows (default: 50)")
    memory_parser.add_argument(
        "--remember",
        nargs=3,
        metavar=("CATEGORY", "KEY", "CONTENT"),
        default=None,
        help="store one memory record",
    )
    memory_parser.add_argument(
        "--tags", nargs="*", default=None, metavar="TAG", help="tags for --remember"
    )
    memory_parser.set_defaults(handler=command_memory)

    files_parser = subparsers.add_parser(
        "files", help="read-only workspace inspection (list, read, info, search)"
    )
    files_parser.add_argument("action", choices=("list", "read", "info", "search"))
    files_parser.add_argument("path", help="target path inside a permitted workspace root")
    files_parser.add_argument("--query", default=None, help="search string for 'search'")
    files_parser.add_argument("--limit", type=int, default=50, help="maximum search matches")
    files_parser.add_argument(
        "--root", default=None, help="override the permitted root for this call"
    )
    files_parser.set_defaults(handler=command_files)

    git_parser = subparsers.add_parser(
        "git", help="read-only git inspection (status, branch, log, diff)"
    )
    git_parser.add_argument("action", choices=("status", "branch", "log", "diff"))
    git_parser.add_argument("path", help="path inside a git repository in a permitted root")
    git_parser.add_argument("--limit", type=int, default=10, help="commits for 'log'")
    git_parser.add_argument("--staged", action="store_true", help="diff the index for 'diff'")
    git_parser.set_defaults(handler=command_git)

    log_parser = subparsers.add_parser(
        "audit-log", help="show tool-execution metadata persisted in the database"
    )
    log_parser.add_argument("--tool", default=None, help="filter by tool name")
    log_parser.add_argument("--limit", type=int, default=25, help="maximum rows (default: 25)")
    log_parser.set_defaults(handler=command_audit_log)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        parser.print_help()
        return EXIT_OK

    try:
        exit_code = args.handler(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except PermissionDenied as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except PathSecurityError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if args.persist_audit and args.command not in {"tasks", "memory"}:
        # tasks/memory already mirror through a live sink; flushing them too
        # would double-record.
        try:
            with _open_manager(args) as manager:
                written = manager.flush_audit_log()
            if not args.json:
                print(f"\npersisted {written} audit entr{'y' if written == 1 else 'ies'}")
        except (DatabaseNotInitialized, DatabaseError) as exc:
            print(f"warning: could not persist audit entries: {exc}", file=sys.stderr)

    if args.show_audit:
        entries = get_audit_logger().entries()
        if args.json:
            print(json.dumps([entry.as_dict() for entry in entries], indent=2, default=str))
        else:
            heading(f"Audit entries this run ({len(entries)})")
            for entry in entries:
                print(f"  {entry.as_human()}")
            print()

    return exit_code
