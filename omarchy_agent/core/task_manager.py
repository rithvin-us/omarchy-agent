"""Task and memory orchestration.

:class:`TaskManager` is the thin layer the CLI (and later the MCP server and
the model loop) talk to. It owns the database handle, applies the permission
policy to every mutation, and optionally mirrors the audit log into the
``tool_executions`` table so that tool metadata survives a restart.

It deliberately holds no inference logic. When LocalAI inference is enabled in
a later milestone, the model loop will drive this class rather than replace it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..memory.database import (
    Approval,
    Conversation,
    Database,
    DatabaseError,
    MemoryRecord,
    Message,
    Task,
    ToolExecution,
    default_data_directory,
    default_database_path,
)
from ..security.audit import AuditLogger, audited, get_audit_logger
from ..security.permissions import Capability

__all__ = ["InitializationReport", "TaskManager", "initialize_storage"]


@dataclass(frozen=True)
class InitializationReport:
    """Outcome of an explicit ``init`` run."""

    database_path: str
    data_directory: str
    directory_created: bool
    database_created: bool
    already_existed: bool
    schema_version: int
    tables: list[str]
    row_counts: dict[str, int]
    integrity: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "database_path": self.database_path,
            "data_directory": self.data_directory,
            "directory_created": self.directory_created,
            "database_created": self.database_created,
            "already_existed": self.already_existed,
            "schema_version": self.schema_version,
            "tables": list(self.tables),
            "row_counts": dict(self.row_counts),
            "integrity": self.integrity,
        }

    def audit_summary(self) -> str:
        return (
            f"init {self.database_path}: created={self.database_created} "
            f"existed={self.already_existed} schema={self.schema_version}"
        )


def initialize_storage(
    path: str | os.PathLike[str] | None = None,
) -> InitializationReport:
    """Create the data directory and database **only when asked explicitly**.

    This is the single place in the project that is allowed to create anything
    outside the project tree, and it runs only from ``python -m omarchy_agent
    init``. An existing database is opened and migrated forward; it is never
    recreated, truncated or replaced.
    """
    target = Path(path).expanduser() if path is not None else default_database_path()
    data_directory = target.parent

    directory_existed = data_directory.is_dir()
    database_existed = target.exists()

    with Database.open(target, create=True) as database:
        report = InitializationReport(
            database_path=str(database.path),
            data_directory=str(data_directory),
            directory_created=not directory_existed,
            database_created=not database_existed,
            already_existed=database_existed,
            schema_version=database.schema_version,
            tables=database.table_names(),
            row_counts=database.row_counts(),
            integrity=database.integrity_check(),
        )
    return report


class TaskManager:
    """Permission-checked access to tasks, memories, conversations and audit rows."""

    def __init__(self, database: Database, *, audit_logger: AuditLogger | None = None) -> None:
        self._database = database
        self._audit = audit_logger or get_audit_logger()

    # -- lifecycle -------------------------------------------------------

    @classmethod
    def open(
        cls,
        path: str | os.PathLike[str] | None = None,
        *,
        create: bool = False,
        mirror_audit_to_database: bool = False,
        audit_logger: AuditLogger | None = None,
    ) -> TaskManager:
        """Open the store. ``create=False`` never touches the filesystem layout."""
        database = Database.open(path, create=create)
        manager = cls(database, audit_logger=audit_logger)
        if mirror_audit_to_database:
            manager._audit.add_sink(database.audit_sink())
        return manager

    @property
    def database(self) -> Database:
        return self._database

    def close(self) -> None:
        self._database.close()

    def __enter__(self) -> TaskManager:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- tasks -----------------------------------------------------------

    @audited("tasks.create_task", Capability.TASK_WRITE)
    def create_task(self, title: str, description: str = "") -> Task:
        return self._database.create_task(title, description)

    @audited("tasks.get_task", Capability.TASK_READ)
    def get_task(self, task_id: int) -> Task | None:
        return self._database.get_task(task_id)

    @audited("tasks.list_tasks", Capability.TASK_READ)
    def list_tasks(self, status: str | None = None, limit: int = 50) -> list[Task]:
        return self._database.list_tasks(status=status, limit=limit)

    @audited("tasks.update_task", Capability.TASK_WRITE)
    def update_task(
        self,
        task_id: int,
        *,
        title: str | None = None,
        description: str | None = None,
        status: str | None = None,
    ) -> Task:
        return self._database.update_task(
            task_id, title=title, description=description, status=status
        )

    @audited("tasks.set_task_status", Capability.TASK_WRITE)
    def set_task_status(self, task_id: int, status: str) -> Task:
        return self._database.update_task(task_id, status=status)

    @audited("tasks.complete_task", Capability.TASK_WRITE)
    def complete_task(self, task_id: int) -> Task:
        return self._database.complete_task(task_id)

    def task_counts(self) -> dict[str, int]:
        return self._database.count_tasks()

    # -- memories --------------------------------------------------------

    @audited("memory.remember", Capability.MEMORY_WRITE, redact_params=("content",))
    def remember(
        self,
        category: str,
        key: str,
        content: str,
        tags: list[str] | None = None,
    ) -> MemoryRecord:
        """Store a memory. Content is scrubbed and length-capped by the storage layer."""
        return self._database.remember(category, key, content, tags)

    @audited("memory.recall", Capability.MEMORY_READ)
    def recall(self, category: str, key: str) -> MemoryRecord | None:
        return self._database.get_memory(category, key)

    @audited("memory.list_memories", Capability.MEMORY_READ)
    def list_memories(
        self, category: str | None = None, tag: str | None = None, limit: int = 100
    ) -> list[MemoryRecord]:
        return self._database.list_memories(category=category, tag=tag, limit=limit)

    @audited("memory.search_memories", Capability.MEMORY_READ)
    def search_memories(self, query: str, limit: int = 50) -> list[MemoryRecord]:
        return self._database.search_memories(query, limit=limit)

    @audited("memory.forget", Capability.MEMORY_WRITE)
    def forget(self, category: str, key: str) -> bool:
        return self._database.forget(category, key)

    # -- conversations ---------------------------------------------------

    @audited("conversations.start", Capability.MEMORY_WRITE)
    def start_conversation(self, title: str) -> Conversation:
        return self._database.create_conversation(title)

    @audited("conversations.add_message", Capability.MEMORY_WRITE, redact_params=("content",))
    def add_message(self, conversation_id: int, role: str, content: str) -> Message:
        return self._database.add_message(conversation_id, role, content)

    @audited("conversations.get_messages", Capability.MEMORY_READ)
    def get_messages(self, conversation_id: int, limit: int = 200) -> list[Message]:
        return self._database.get_messages(conversation_id, limit=limit)

    @audited("conversations.list", Capability.MEMORY_READ)
    def list_conversations(self, limit: int = 50) -> list[Conversation]:
        return self._database.list_conversations(limit=limit)

    # -- tool execution metadata -----------------------------------------

    @audited("audit.record_tool_execution", Capability.AUDIT_WRITE)
    def record_tool_execution(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        result_summary: str = "",
        status: str = "success",
        duration_ms: int = 0,
        task_id: int | None = None,
    ) -> ToolExecution:
        return self._database.record_tool_execution(
            tool_name=tool_name,
            arguments=arguments,
            result_summary=result_summary,
            status=status,
            duration_ms=duration_ms,
            task_id=task_id,
        )

    @audited("audit.list_tool_executions", Capability.TASK_READ)
    def list_tool_executions(
        self, tool_name: str | None = None, limit: int = 50
    ) -> list[ToolExecution]:
        return self._database.list_tool_executions(tool_name=tool_name, limit=limit)

    def flush_audit_log(self, task_id: int | None = None) -> int:
        """Persist the in-memory audit entries as ``tool_executions`` rows.

        Returns the number of rows written. Used by the CLI so a one-shot
        command still leaves a durable trace without installing a sink that
        would double-record.
        """
        written = 0
        for entry in self._audit.entries():
            try:
                self._database.record_tool_execution(
                    tool_name=entry.tool_name,
                    arguments=entry.arguments,
                    result_summary=entry.result_summary,
                    status=entry.status,
                    duration_ms=entry.duration_ms,
                    task_id=entry.task_id if entry.task_id is not None else task_id,
                )
                written += 1
            except DatabaseError:
                continue
        return written

    # -- approvals -------------------------------------------------------

    @audited("approvals.request", Capability.AUDIT_WRITE)
    def request_approval(self, action: str, description: str = "") -> Approval:
        """Record that a capability needs the user's explicit go-ahead.

        v0.1 only records the request; nothing in this version acts on an
        approval, because no capability that would need one is implemented.
        """
        return self._database.create_approval(action, description)

    @audited("approvals.resolve", Capability.AUDIT_WRITE)
    def resolve_approval(self, approval_id: int, status: str) -> Approval:
        return self._database.resolve_approval(approval_id, status)

    @audited("approvals.list", Capability.TASK_READ)
    def list_approvals(self, status: str | None = None, limit: int = 50) -> list[Approval]:
        return self._database.list_approvals(status=status, limit=limit)

    # -- reporting -------------------------------------------------------

    def storage_report(self) -> dict[str, Any]:
        info = self._database.info()
        info["data_directory"] = str(default_data_directory())
        info["task_counts"] = self._database.count_tasks()
        return info
