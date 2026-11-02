"""Persistent storage for the Omarchy Agent.

Backed by SQLite from the standard library. Design rules:

* Initialization is **non-destructive**: an existing database file is opened
  and migrated forward, never recreated.
* All SQL is parameterized. No value is ever interpolated into a statement.
* Writes go through an explicit transaction.
* Timestamps are UTC ISO-8601 strings.
* Content is scrubbed for credential-shaped text and length-capped before it
  is stored, so the agent cannot quietly accumulate secrets or giant logs.
* The database file is created with ``0600`` and its directory with ``0700``.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..security.audit import SENSITIVE_KEY_PATTERN, redact_text

__all__ = [
    "DEFAULT_DATABASE_PATH",
    "SCHEMA_VERSION",
    "Approval",
    "Conversation",
    "Database",
    "DatabaseError",
    "DatabaseNotInitialized",
    "MemoryRecord",
    "Message",
    "Task",
    "TaskStatus",
    "ToolExecution",
    "default_data_directory",
    "default_database_path",
    "scrub_secrets",
]

SCHEMA_VERSION = 1

#: Largest piece of text the agent will persist in one record.
MAX_CONTENT_CHARS = 16 * 1024

#: Largest tool-execution result summary kept.
MAX_SUMMARY_CHARS = 1024

VALID_TASK_STATUSES: tuple[str, ...] = (
    "pending",
    "in_progress",
    "blocked",
    "done",
    "cancelled",
)

VALID_MESSAGE_ROLES: tuple[str, ...] = ("system", "user", "assistant", "tool")

VALID_APPROVAL_STATUSES: tuple[str, ...] = ("pending", "approved", "rejected", "expired")


class DatabaseError(RuntimeError):
    """Raised for storage problems the caller is expected to handle."""


class DatabaseNotInitialized(DatabaseError):
    """Raised when the database file does not exist and creation was not asked for."""


def default_data_directory() -> Path:
    """Return the agent's persistent data directory (never created here)."""
    override = os.environ.get("OMARCHY_AGENT_DATA_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_DATA_HOME", "").strip()
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return base / "omarchy-agent"


def default_database_path() -> Path:
    return default_data_directory() / "agent.sqlite3"


#: Evaluated lazily by callers; kept as a module constant for documentation.
DEFAULT_DATABASE_PATH = "~/.local/share/omarchy-agent/agent.sqlite3"


def utc_now() -> str:
    """UTC timestamp, ISO-8601, second precision."""
    return datetime.now(UTC).isoformat(timespec="seconds")


class TaskStatus:
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    DONE = "done"
    CANCELLED = "cancelled"


def scrub_secrets(text: str) -> str:
    """Remove credential-shaped text before persisting ``text``.

    Two passes: known token/key shapes (shared with the audit logger) and
    ``NAME=value`` assignments whose name looks like a secret.
    """
    cleaned = redact_text(text)
    scrubbed_lines: list[str] = []
    for line in cleaned.splitlines(keepends=False):
        name, separator, _value = line.partition("=")
        if separator and SENSITIVE_KEY_PATTERN.search(name):
            scrubbed_lines.append(f"{name}=[REDACTED]")
        else:
            scrubbed_lines.append(line)
    result = "\n".join(scrubbed_lines)
    if cleaned.endswith("\n") and not result.endswith("\n"):
        result += "\n"
    return result


def _prepare_content(content: str, limit: int = MAX_CONTENT_CHARS) -> str:
    if not isinstance(content, str):
        raise DatabaseError("content must be a string")
    cleaned = scrub_secrets(content)
    if len(cleaned) > limit:
        cleaned = f"{cleaned[:limit]}\n...[truncated, {len(cleaned)} chars original]"
    return cleaned


def _encode_json(value: Any, *, field: str) -> str:
    """Serialize ``value`` to JSON, rejecting anything non-serializable."""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise DatabaseError(f"{field} is not JSON-serializable: {exc}") from exc


def _decode_json(raw: str | None, *, field: str, default: Any) -> Any:
    if raw is None or raw == "":
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        # Corrupt row: never raise on read, report the default instead.
        return default


def _normalize_tags(tags: Sequence[str] | None) -> list[str]:
    if tags is None:
        return []
    if isinstance(tags, str):
        raise DatabaseError("tags must be a sequence of strings, not a single string")
    normalized: list[str] = []
    for tag in tags:
        if not isinstance(tag, str):
            raise DatabaseError("every tag must be a string")
        candidate = tag.strip()
        if candidate and candidate not in normalized:
            normalized.append(candidate)
    return normalized


# --------------------------------------------------------------------------
# row dataclasses
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Task:
    id: int
    title: str
    description: str
    status: str
    created_at: str
    updated_at: str
    completed_at: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Task":
        return cls(
            id=row["id"],
            title=row["title"],
            description=row["description"] or "",
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            completed_at=row["completed_at"],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
        }

    def audit_summary(self) -> str:
        return f"task id={self.id} status={self.status}"


@dataclass(frozen=True)
class MemoryRecord:
    id: int
    category: str
    key: str
    content: str
    tags: list[str]
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "MemoryRecord":
        return cls(
            id=row["id"],
            category=row["category"],
            key=row["key"],
            content=row["content"],
            tags=_decode_json(row["tags"], field="tags", default=[]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "key": self.key,
            "content": self.content,
            "tags": list(self.tags),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def audit_summary(self) -> str:
        return f"memory id={self.id} {self.category}/{self.key} <{len(self.content)} chars>"


@dataclass(frozen=True)
class Conversation:
    id: int
    title: str
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Conversation":
        return cls(
            id=row["id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def audit_summary(self) -> str:
        return f"conversation id={self.id}"


@dataclass(frozen=True)
class Message:
    id: int
    conversation_id: int
    role: str
    content: str
    created_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Message":
        return cls(
            id=row["id"],
            conversation_id=row["conversation_id"],
            role=row["role"],
            content=row["content"],
            created_at=row["created_at"],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
        }

    def audit_summary(self) -> str:
        return f"message id={self.id} role={self.role} <{len(self.content)} chars>"


@dataclass(frozen=True)
class ToolExecution:
    id: int
    task_id: int | None
    tool_name: str
    arguments: dict[str, Any]
    result_summary: str
    status: str
    created_at: str
    duration_ms: int

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ToolExecution":
        return cls(
            id=row["id"],
            task_id=row["task_id"],
            tool_name=row["tool_name"],
            arguments=_decode_json(row["arguments"], field="arguments", default={}),
            result_summary=row["result_summary"] or "",
            status=row["status"],
            created_at=row["created_at"],
            duration_ms=row["duration_ms"],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "result_summary": self.result_summary,
            "status": self.status,
            "created_at": self.created_at,
            "duration_ms": self.duration_ms,
        }

    def audit_summary(self) -> str:
        return f"tool_execution id={self.id} tool={self.tool_name} status={self.status}"


@dataclass(frozen=True)
class Approval:
    id: int
    action: str
    description: str
    status: str
    created_at: str
    resolved_at: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Approval":
        return cls(
            id=row["id"],
            action=row["action"],
            description=row["description"] or "",
            status=row["status"],
            created_at=row["created_at"],
            resolved_at=row["resolved_at"],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "action": self.action,
            "description": self.description,
            "status": self.status,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
        }

    def audit_summary(self) -> str:
        return f"approval id={self.id} status={self.status}"


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------

_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS conversations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    title      TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
    content         TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages (conversation_id, id);

CREATE TABLE IF NOT EXISTS tasks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    title        TEXT NOT NULL,
    description  TEXT NOT NULL DEFAULT '',
    status       TEXT NOT NULL DEFAULT 'pending'
                 CHECK (status IN ('pending', 'in_progress', 'blocked', 'done', 'cancelled')),
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks (status, id);

CREATE TABLE IF NOT EXISTS memories (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    category   TEXT NOT NULL,
    key        TEXT NOT NULL,
    content    TEXT NOT NULL,
    tags       TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (category, key)
);

CREATE INDEX IF NOT EXISTS idx_memories_category ON memories (category, key);

CREATE TABLE IF NOT EXISTS tool_executions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id        INTEGER REFERENCES tasks(id) ON DELETE SET NULL,
    tool_name      TEXT NOT NULL,
    arguments      TEXT NOT NULL DEFAULT '{}',
    result_summary TEXT NOT NULL DEFAULT '',
    status         TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    duration_ms    INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_tool_executions_tool
    ON tool_executions (tool_name, id);

CREATE TABLE IF NOT EXISTS approvals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    action      TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'approved', 'rejected', 'expired')),
    created_at  TEXT NOT NULL,
    resolved_at TEXT
);
"""

TABLE_NAMES: tuple[str, ...] = (
    "conversations",
    "messages",
    "tasks",
    "memories",
    "tool_executions",
    "approvals",
)


def _migrate_to_v1(connection: sqlite3.Connection) -> None:
    connection.executescript(_SCHEMA_V1)


#: Ordered migration steps. Each entry upgrades from ``version - 1`` to
#: ``version``. Steps must be additive so an existing database survives.
MIGRATIONS: tuple[tuple[int, Any], ...] = ((1, _migrate_to_v1),)


# --------------------------------------------------------------------------
# the database
# --------------------------------------------------------------------------


class Database:
    """A connection to the agent's SQLite store.

    Use as a context manager::

        with Database.open(path, create=True) as db:
            db.create_task("inspect repo")
    """

    def __init__(self, connection: sqlite3.Connection, path: Path) -> None:
        self._connection = connection
        self._path = path

    # -- lifecycle -------------------------------------------------------

    @classmethod
    def open(
        cls,
        path: str | os.PathLike[str] | None = None,
        *,
        create: bool = False,
        busy_timeout_ms: int = 5000,
    ) -> "Database":
        """Open (and optionally create) the database.

        ``create=False`` -- the default -- never touches the filesystem layout:
        a missing file raises :class:`DatabaseNotInitialized`. ``create=True``
        is reached only from the explicit ``init`` command.
        """
        target = Path(path).expanduser() if path is not None else default_database_path()
        existed = target.exists()

        if not existed and not create:
            raise DatabaseNotInitialized(
                f"database not found at {target}; run 'python -m omarchy_agent init' first"
            )

        if create and not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(target.parent, 0o700)
            except OSError:
                pass

        try:
            connection = sqlite3.connect(
                target,
                timeout=busy_timeout_ms / 1000,
                isolation_level=None,  # explicit transaction control
            )
        except sqlite3.Error as exc:
            raise DatabaseError(f"cannot open database at {target}: {exc}") from exc

        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")

        database = cls(connection, target)
        if not existed:
            database._restrict_permissions()
        try:
            database._migrate()
        except BaseException:
            # Never leak the connection when the schema is unusable.
            database.close()
            raise
        return database

    def _restrict_permissions(self) -> None:
        for candidate in (
            self._path,
            self._path.with_name(self._path.name + "-wal"),
            self._path.with_name(self._path.name + "-shm"),
        ):
            try:
                if candidate.exists():
                    os.chmod(candidate, 0o600)
            except OSError:
                pass

    def _migrate(self) -> None:
        """Apply pending migrations. Never drops or rewrites existing data."""
        current = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
        if current > SCHEMA_VERSION:
            raise DatabaseError(
                f"database schema version {current} is newer than this build "
                f"supports ({SCHEMA_VERSION}); upgrade the agent"
            )
        for version, step in MIGRATIONS:
            if version <= current:
                continue
            with self.transaction():
                step(self._connection)
            # PRAGMA cannot be parameterized; the value is an int literal here.
            self._connection.execute(f"PRAGMA user_version = {int(version)}")

    @property
    def path(self) -> Path:
        return self._path

    @property
    def schema_version(self) -> int:
        return int(self._connection.execute("PRAGMA user_version").fetchone()[0])

    def close(self) -> None:
        try:
            self._connection.close()
        except sqlite3.Error:
            pass

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- transactions ----------------------------------------------------

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a block inside one transaction; roll back on any exception.

        ``in_transaction`` is re-checked before COMMIT and ROLLBACK because
        some statements -- ``executescript`` in a migration, for one -- commit
        on their own, leaving no open transaction to close.
        """
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield self._connection
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise
        if self._connection.in_transaction:
            self._connection.execute("COMMIT")

    # -- tasks -----------------------------------------------------------

    def create_task(
        self, title: str, description: str = "", status: str = TaskStatus.PENDING
    ) -> Task:
        clean_title = _prepare_content(title, limit=512).strip()
        if not clean_title:
            raise DatabaseError("task title must not be empty")
        if status not in VALID_TASK_STATUSES:
            raise DatabaseError(
                f"invalid task status {status!r}; expected one of {VALID_TASK_STATUSES}"
            )
        now = utc_now()
        with self.transaction() as connection:
            cursor = connection.execute(
                "INSERT INTO tasks (title, description, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (clean_title, _prepare_content(description), status, now, now),
            )
            task_id = int(cursor.lastrowid or 0)
        task = self.get_task(task_id)
        if task is None:  # pragma: no cover - insert just succeeded
            raise DatabaseError("task insert did not produce a row")
        return task

    def get_task(self, task_id: int) -> Task | None:
        row = self._connection.execute(
            "SELECT * FROM tasks WHERE id = ?", (int(task_id),)
        ).fetchone()
        return Task.from_row(row) if row else None

    def list_tasks(self, status: str | None = None, limit: int = 50) -> list[Task]:
        if status is not None and status not in VALID_TASK_STATUSES:
            raise DatabaseError(f"invalid task status {status!r}")
        if status is None:
            rows = self._connection.execute(
                "SELECT * FROM tasks ORDER BY id DESC LIMIT ?", (int(limit),)
            ).fetchall()
        else:
            rows = self._connection.execute(
                "SELECT * FROM tasks WHERE status = ? ORDER BY id DESC LIMIT ?",
                (status, int(limit)),
            ).fetchall()
        return [Task.from_row(row) for row in rows]

    def update_task(
        self,
        task_id: int,
        *,
        title: str | None = None,
        description: str | None = None,
        status: str | None = None,
    ) -> Task:
        existing = self.get_task(task_id)
        if existing is None:
            raise DatabaseError(f"no task with id {task_id}")
        if status is not None and status not in VALID_TASK_STATUSES:
            raise DatabaseError(f"invalid task status {status!r}")

        now = utc_now()
        new_title = existing.title if title is None else _prepare_content(title, 512).strip()
        if not new_title:
            raise DatabaseError("task title must not be empty")
        new_description = (
            existing.description if description is None else _prepare_content(description)
        )
        new_status = existing.status if status is None else status
        completed_at = existing.completed_at
        if new_status == TaskStatus.DONE and completed_at is None:
            completed_at = now
        if new_status != TaskStatus.DONE:
            completed_at = None

        with self.transaction() as connection:
            connection.execute(
                "UPDATE tasks SET title = ?, description = ?, status = ?, "
                "updated_at = ?, completed_at = ? WHERE id = ?",
                (
                    new_title,
                    new_description,
                    new_status,
                    now,
                    completed_at,
                    int(task_id),
                ),
            )
        updated = self.get_task(task_id)
        if updated is None:  # pragma: no cover
            raise DatabaseError("task vanished during update")
        return updated

    def complete_task(self, task_id: int) -> Task:
        return self.update_task(task_id, status=TaskStatus.DONE)

    def count_tasks(self) -> dict[str, int]:
        rows = self._connection.execute(
            "SELECT status, COUNT(*) AS total FROM tasks GROUP BY status"
        ).fetchall()
        return {row["status"]: int(row["total"]) for row in rows}

    # -- memories --------------------------------------------------------

    def remember(
        self,
        category: str,
        key: str,
        content: str,
        tags: Sequence[str] | None = None,
    ) -> MemoryRecord:
        """Create or update one memory record, keyed by ``(category, key)``."""
        clean_category = category.strip()
        clean_key = key.strip()
        if not clean_category or not clean_key:
            raise DatabaseError("memory category and key must both be non-empty")
        clean_content = _prepare_content(content)
        encoded_tags = _encode_json(_normalize_tags(tags), field="tags")
        now = utc_now()
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO memories (category, key, content, tags, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (category, key) DO UPDATE SET "
                "content = excluded.content, tags = excluded.tags, updated_at = excluded.updated_at",
                (clean_category, clean_key, clean_content, encoded_tags, now, now),
            )
        record = self.get_memory(clean_category, clean_key)
        if record is None:  # pragma: no cover
            raise DatabaseError("memory upsert did not produce a row")
        return record

    def get_memory(self, category: str, key: str) -> MemoryRecord | None:
        row = self._connection.execute(
            "SELECT * FROM memories WHERE category = ? AND key = ?",
            (category.strip(), key.strip()),
        ).fetchone()
        return MemoryRecord.from_row(row) if row else None

    def list_memories(
        self, category: str | None = None, tag: str | None = None, limit: int = 100
    ) -> list[MemoryRecord]:
        if category is None:
            rows = self._connection.execute(
                "SELECT * FROM memories ORDER BY updated_at DESC, id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        else:
            rows = self._connection.execute(
                "SELECT * FROM memories WHERE category = ? "
                "ORDER BY updated_at DESC, id DESC LIMIT ?",
                (category.strip(), int(limit)),
            ).fetchall()
        records = [MemoryRecord.from_row(row) for row in rows]
        if tag is not None:
            records = [record for record in records if tag in record.tags]
        return records

    def search_memories(self, query: str, limit: int = 50) -> list[MemoryRecord]:
        """Substring search over key and content, parameterized with LIKE."""
        pattern = f"%{query}%"
        rows = self._connection.execute(
            "SELECT * FROM memories WHERE key LIKE ? ESCAPE '\\' OR content LIKE ? ESCAPE '\\' "
            "ORDER BY updated_at DESC LIMIT ?",
            (pattern, pattern, int(limit)),
        ).fetchall()
        return [MemoryRecord.from_row(row) for row in rows]

    def forget(self, category: str, key: str) -> bool:
        """Delete one of the agent's own memory records. Returns True if removed."""
        with self.transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM memories WHERE category = ? AND key = ?",
                (category.strip(), key.strip()),
            )
            return cursor.rowcount > 0

    # -- conversations and messages --------------------------------------

    def create_conversation(self, title: str) -> Conversation:
        clean_title = _prepare_content(title, 512).strip() or "untitled"
        now = utc_now()
        with self.transaction() as connection:
            cursor = connection.execute(
                "INSERT INTO conversations (title, created_at, updated_at) VALUES (?, ?, ?)",
                (clean_title, now, now),
            )
            conversation_id = int(cursor.lastrowid or 0)
        conversation = self.get_conversation(conversation_id)
        if conversation is None:  # pragma: no cover
            raise DatabaseError("conversation insert did not produce a row")
        return conversation

    def get_conversation(self, conversation_id: int) -> Conversation | None:
        row = self._connection.execute(
            "SELECT * FROM conversations WHERE id = ?", (int(conversation_id),)
        ).fetchone()
        return Conversation.from_row(row) if row else None

    def list_conversations(self, limit: int = 50) -> list[Conversation]:
        rows = self._connection.execute(
            "SELECT * FROM conversations ORDER BY updated_at DESC, id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [Conversation.from_row(row) for row in rows]

    def add_message(self, conversation_id: int, role: str, content: str) -> Message:
        if role not in VALID_MESSAGE_ROLES:
            raise DatabaseError(
                f"invalid message role {role!r}; expected one of {VALID_MESSAGE_ROLES}"
            )
        if self.get_conversation(conversation_id) is None:
            raise DatabaseError(f"no conversation with id {conversation_id}")
        now = utc_now()
        with self.transaction() as connection:
            cursor = connection.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                (int(conversation_id), role, _prepare_content(content), now),
            )
            message_id = int(cursor.lastrowid or 0)
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, int(conversation_id)),
            )
        row = self._connection.execute(
            "SELECT * FROM messages WHERE id = ?", (message_id,)
        ).fetchone()
        return Message.from_row(row)

    def get_messages(self, conversation_id: int, limit: int = 200) -> list[Message]:
        rows = self._connection.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id ASC LIMIT ?",
            (int(conversation_id), int(limit)),
        ).fetchall()
        return [Message.from_row(row) for row in rows]

    # -- tool executions -------------------------------------------------

    def record_tool_execution(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        result_summary: str = "",
        status: str = "success",
        duration_ms: int = 0,
        task_id: int | None = None,
    ) -> ToolExecution:
        """Persist tool-execution *metadata*. Never file contents."""
        encoded_arguments = _encode_json(arguments or {}, field="arguments")
        summary = scrub_secrets(result_summary)[:MAX_SUMMARY_CHARS]
        now = utc_now()
        with self.transaction() as connection:
            cursor = connection.execute(
                "INSERT INTO tool_executions "
                "(task_id, tool_name, arguments, result_summary, status, created_at, duration_ms) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    int(task_id) if task_id is not None else None,
                    str(tool_name),
                    encoded_arguments,
                    summary,
                    str(status),
                    now,
                    int(duration_ms),
                ),
            )
            execution_id = int(cursor.lastrowid or 0)
        row = self._connection.execute(
            "SELECT * FROM tool_executions WHERE id = ?", (execution_id,)
        ).fetchone()
        return ToolExecution.from_row(row)

    def list_tool_executions(
        self, tool_name: str | None = None, limit: int = 50
    ) -> list[ToolExecution]:
        if tool_name is None:
            rows = self._connection.execute(
                "SELECT * FROM tool_executions ORDER BY id DESC LIMIT ?", (int(limit),)
            ).fetchall()
        else:
            rows = self._connection.execute(
                "SELECT * FROM tool_executions WHERE tool_name = ? ORDER BY id DESC LIMIT ?",
                (tool_name, int(limit)),
            ).fetchall()
        return [ToolExecution.from_row(row) for row in rows]

    # -- approvals -------------------------------------------------------

    def create_approval(self, action: str, description: str = "") -> Approval:
        clean_action = action.strip()
        if not clean_action:
            raise DatabaseError("approval action must not be empty")
        now = utc_now()
        with self.transaction() as connection:
            cursor = connection.execute(
                "INSERT INTO approvals (action, description, status, created_at) "
                "VALUES (?, ?, 'pending', ?)",
                (clean_action, _prepare_content(description), now),
            )
            approval_id = int(cursor.lastrowid or 0)
        row = self._connection.execute(
            "SELECT * FROM approvals WHERE id = ?", (approval_id,)
        ).fetchone()
        return Approval.from_row(row)

    def resolve_approval(self, approval_id: int, status: str) -> Approval:
        if status not in VALID_APPROVAL_STATUSES or status == "pending":
            raise DatabaseError(
                f"invalid approval resolution {status!r}; "
                "expected 'approved', 'rejected' or 'expired'"
            )
        now = utc_now()
        with self.transaction() as connection:
            cursor = connection.execute(
                "UPDATE approvals SET status = ?, resolved_at = ? WHERE id = ?",
                (status, now, int(approval_id)),
            )
            if cursor.rowcount == 0:
                raise DatabaseError(f"no approval with id {approval_id}")
        row = self._connection.execute(
            "SELECT * FROM approvals WHERE id = ?", (int(approval_id),)
        ).fetchone()
        return Approval.from_row(row)

    def list_approvals(self, status: str | None = None, limit: int = 50) -> list[Approval]:
        if status is None:
            rows = self._connection.execute(
                "SELECT * FROM approvals ORDER BY id DESC LIMIT ?", (int(limit),)
            ).fetchall()
        else:
            if status not in VALID_APPROVAL_STATUSES:
                raise DatabaseError(f"invalid approval status {status!r}")
            rows = self._connection.execute(
                "SELECT * FROM approvals WHERE status = ? ORDER BY id DESC LIMIT ?",
                (status, int(limit)),
            ).fetchall()
        return [Approval.from_row(row) for row in rows]

    # -- introspection ---------------------------------------------------

    def table_names(self) -> list[str]:
        rows = self._connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        ).fetchall()
        return [row["name"] for row in rows]

    def row_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        present = set(self.table_names())
        for table in TABLE_NAMES:
            if table not in present:
                continue
            # ``table`` comes from the module-level constant, never from input.
            row = self._connection.execute(f"SELECT COUNT(*) AS total FROM {table}").fetchone()
            counts[table] = int(row["total"])
        return counts

    def integrity_check(self) -> str:
        row = self._connection.execute("PRAGMA integrity_check").fetchone()
        return str(row[0])

    def info(self) -> dict[str, Any]:
        try:
            mode = oct(self._path.stat().st_mode & 0o777)
            size = self._path.stat().st_size
        except OSError:
            mode, size = "unknown", 0
        return {
            "path": str(self._path),
            "exists": self._path.exists(),
            "size_bytes": size,
            "permissions": mode,
            "schema_version": self.schema_version,
            "expected_schema_version": SCHEMA_VERSION,
            "foreign_keys": bool(
                self._connection.execute("PRAGMA foreign_keys").fetchone()[0]
            ),
            "journal_mode": str(
                self._connection.execute("PRAGMA journal_mode").fetchone()[0]
            ),
            "tables": self.table_names(),
            "row_counts": self.row_counts(),
        }

    # -- audit sink ------------------------------------------------------

    def audit_sink(self, task_id: int | None = None) -> Any:
        """Return a callable suitable for :meth:`AuditLogger.add_sink`."""

        def sink(entry: Any) -> None:
            self.record_tool_execution(
                tool_name=entry.tool_name,
                arguments=entry.arguments,
                result_summary=entry.result_summary,
                status=entry.status,
                duration_ms=entry.duration_ms,
                task_id=entry.task_id if entry.task_id is not None else task_id,
            )

        return sink
