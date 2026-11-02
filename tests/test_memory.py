"""Tests for the SQLite persistence layer and the task manager.

Every test uses its own temporary directory and database file; the real
``~/.local/share/omarchy-agent`` is never touched.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from omarchy_agent.core.task_manager import TaskManager, initialize_storage
from omarchy_agent.memory.database import (
    SCHEMA_VERSION,
    Database,
    DatabaseError,
    DatabaseNotInitialized,
    TaskStatus,
    scrub_secrets,
)
from omarchy_agent.security.audit import AuditLogger, get_audit_logger, set_audit_logger


class DatabaseTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self._temporary.name)
        self.path = self.directory / "agent.sqlite3"
        self.database = Database.open(self.path, create=True)

    def tearDown(self) -> None:
        self.database.close()
        self._temporary.cleanup()


class InitializationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self._temporary.name)
        self.path = self.directory / "nested" / "agent.sqlite3"

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_creates_database_and_directory(self) -> None:
        report = initialize_storage(self.path)
        self.assertTrue(self.path.exists())
        self.assertTrue(report.database_created)
        self.assertTrue(report.directory_created)
        self.assertFalse(report.already_existed)
        self.assertEqual(report.schema_version, SCHEMA_VERSION)
        self.assertEqual(report.integrity, "ok")

    def test_creates_every_required_table(self) -> None:
        report = initialize_storage(self.path)
        for table in (
            "conversations",
            "messages",
            "tasks",
            "memories",
            "tool_executions",
            "approvals",
        ):
            with self.subTest(table=table):
                self.assertIn(table, report.tables)

    def test_directory_is_private(self) -> None:
        initialize_storage(self.path)
        self.assertEqual(oct(self.path.parent.stat().st_mode & 0o777), "0o700")

    def test_database_file_is_private(self) -> None:
        initialize_storage(self.path)
        self.assertEqual(oct(self.path.stat().st_mode & 0o777), "0o600")

    def test_second_initialization_is_non_destructive(self) -> None:
        initialize_storage(self.path)
        with Database.open(self.path) as database:
            task = database.create_task("survive the reinit", "must still be here")
            record = database.remember("project", "note", "important content")
        task_id, memory_id = task.id, record.id

        report = initialize_storage(self.path)
        self.assertFalse(report.database_created)
        self.assertTrue(report.already_existed)
        self.assertEqual(report.row_counts["tasks"], 1)
        self.assertEqual(report.row_counts["memories"], 1)

        with Database.open(self.path) as database:
            survived = database.get_task(task_id)
            self.assertIsNotNone(survived)
            self.assertEqual(survived.title, "survive the reinit")
            self.assertEqual(database.get_memory("project", "note").id, memory_id)

    def test_opening_a_missing_database_without_create_raises(self) -> None:
        with self.assertRaises(DatabaseNotInitialized):
            Database.open(self.path, create=False)
        self.assertFalse(self.path.exists())
        self.assertFalse(self.path.parent.exists())

    def test_pragmas_are_set(self) -> None:
        initialize_storage(self.path)
        with Database.open(self.path) as database:
            info = database.info()
        self.assertTrue(info["foreign_keys"])
        self.assertEqual(info["journal_mode"].lower(), "wal")

    def test_schema_version_is_recorded(self) -> None:
        initialize_storage(self.path)
        with Database.open(self.path) as database:
            self.assertEqual(database.schema_version, SCHEMA_VERSION)

    def test_a_newer_schema_is_refused(self) -> None:
        initialize_storage(self.path)
        connection = sqlite3.connect(self.path)
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 5}")
        connection.close()
        with self.assertRaises(DatabaseError) as raised:
            Database.open(self.path)
        self.assertIn("newer than this build", str(raised.exception))


class PersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.path = Path(self._temporary.name) / "agent.sqlite3"

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_data_survives_close_and_reopen(self) -> None:
        with Database.open(self.path, create=True) as database:
            task = database.create_task("persisted task")
            database.remember("facts", "gpu", "RTX 4050 with 6 GB VRAM")
            conversation = database.create_conversation("first chat")
            database.add_message(conversation.id, "user", "hello")

        with Database.open(self.path, create=False) as database:
            self.assertEqual(database.get_task(task.id).title, "persisted task")
            self.assertIn("RTX 4050", database.get_memory("facts", "gpu").content)
            self.assertEqual(len(database.get_messages(conversation.id)), 1)
            self.assertEqual(database.row_counts()["messages"], 1)

    def test_many_reopens_keep_a_single_schema(self) -> None:
        for _ in range(3):
            with Database.open(self.path, create=True) as database:
                database.create_task("one more")
        with Database.open(self.path) as database:
            self.assertEqual(database.row_counts()["tasks"], 3)
            self.assertEqual(database.schema_version, SCHEMA_VERSION)


class TaskCrudTests(DatabaseTestCase):
    def test_create_and_read(self) -> None:
        task = self.database.create_task("inspect repository", "look at git status")
        self.assertEqual(task.status, TaskStatus.PENDING)
        self.assertEqual(task.title, "inspect repository")
        self.assertIsNone(task.completed_at)
        self.assertTrue(task.created_at.endswith("+00:00"))
        fetched = self.database.get_task(task.id)
        self.assertEqual(fetched, task)

    def test_update_fields(self) -> None:
        task = self.database.create_task("original")
        updated = self.database.update_task(
            task.id, title="renamed", description="new description"
        )
        self.assertEqual(updated.title, "renamed")
        self.assertEqual(updated.description, "new description")
        self.assertEqual(updated.id, task.id)

    def test_status_transitions_and_completion_timestamp(self) -> None:
        task = self.database.create_task("do the thing")
        in_progress = self.database.update_task(task.id, status=TaskStatus.IN_PROGRESS)
        self.assertIsNone(in_progress.completed_at)
        done = self.database.complete_task(task.id)
        self.assertEqual(done.status, TaskStatus.DONE)
        self.assertIsNotNone(done.completed_at)
        reopened = self.database.update_task(task.id, status=TaskStatus.PENDING)
        self.assertIsNone(reopened.completed_at)

    def test_list_and_filter(self) -> None:
        self.database.create_task("a")
        second = self.database.create_task("b")
        self.database.complete_task(second.id)
        self.assertEqual(len(self.database.list_tasks()), 2)
        self.assertEqual(len(self.database.list_tasks(status=TaskStatus.DONE)), 1)
        self.assertEqual(len(self.database.list_tasks(limit=1)), 1)

    def test_counts_by_status(self) -> None:
        self.database.create_task("a")
        self.database.create_task("b")
        self.database.complete_task(self.database.create_task("c").id)
        counts = self.database.count_tasks()
        self.assertEqual(counts[TaskStatus.PENDING], 2)
        self.assertEqual(counts[TaskStatus.DONE], 1)

    def test_rejects_empty_title(self) -> None:
        with self.assertRaises(DatabaseError):
            self.database.create_task("   ")

    def test_rejects_invalid_status(self) -> None:
        with self.assertRaises(DatabaseError):
            self.database.create_task("x", status="exploded")
        task = self.database.create_task("y")
        with self.assertRaises(DatabaseError):
            self.database.update_task(task.id, status="exploded")

    def test_update_of_missing_task_raises(self) -> None:
        with self.assertRaises(DatabaseError):
            self.database.update_task(9999, status=TaskStatus.DONE)

    def test_get_missing_task_returns_none(self) -> None:
        self.assertIsNone(self.database.get_task(4242))

    def test_check_constraint_blocks_direct_bad_status(self) -> None:
        with self.assertRaises(sqlite3.IntegrityError):
            with self.database.transaction() as connection:
                connection.execute(
                    "INSERT INTO tasks (title, description, status, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    ("direct", "", "nonsense", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
                )


class MemoryCrudTests(DatabaseTestCase):
    def test_create_and_read(self) -> None:
        record = self.database.remember(
            "preferences", "editor", "uses neovim", tags=["editor", "tooling"]
        )
        self.assertEqual(record.category, "preferences")
        self.assertEqual(record.key, "editor")
        self.assertEqual(record.tags, ["editor", "tooling"])
        self.assertEqual(self.database.get_memory("preferences", "editor"), record)

    def test_upsert_updates_in_place(self) -> None:
        first = self.database.remember("facts", "ram", "14 GB")
        second = self.database.remember("facts", "ram", "14 GB usable", tags=["hardware"])
        self.assertEqual(first.id, second.id)
        self.assertEqual(second.content, "14 GB usable")
        self.assertEqual(second.tags, ["hardware"])
        self.assertEqual(self.database.row_counts()["memories"], 1)

    def test_list_and_filter_by_category_and_tag(self) -> None:
        self.database.remember("facts", "a", "one", tags=["x"])
        self.database.remember("facts", "b", "two", tags=["y"])
        self.database.remember("prefs", "c", "three", tags=["x"])
        self.assertEqual(len(self.database.list_memories()), 3)
        self.assertEqual(len(self.database.list_memories(category="facts")), 2)
        self.assertEqual(len(self.database.list_memories(tag="x")), 2)

    def test_search(self) -> None:
        self.database.remember("facts", "gpu", "NVIDIA RTX 4050 laptop")
        self.database.remember("facts", "cpu", "AMD Ryzen 7 7735HS")
        self.assertEqual(len(self.database.search_memories("Ryzen")), 1)
        self.assertEqual(len(self.database.search_memories("gpu")), 1)
        self.assertEqual(len(self.database.search_memories("nothing-here")), 0)

    def test_forget(self) -> None:
        self.database.remember("facts", "temp", "delete me")
        self.assertTrue(self.database.forget("facts", "temp"))
        self.assertIsNone(self.database.get_memory("facts", "temp"))
        self.assertFalse(self.database.forget("facts", "temp"))

    def test_rejects_empty_category_or_key(self) -> None:
        with self.assertRaises(DatabaseError):
            self.database.remember("", "key", "content")
        with self.assertRaises(DatabaseError):
            self.database.remember("category", "  ", "content")

    def test_rejects_a_string_as_tags(self) -> None:
        with self.assertRaises(DatabaseError):
            self.database.remember("facts", "k", "v", tags="not-a-list")  # type: ignore[arg-type]

    def test_rejects_non_string_tag(self) -> None:
        with self.assertRaises(DatabaseError):
            self.database.remember("facts", "k", "v", tags=[1, 2])  # type: ignore[list-item]

    def test_tags_are_deduplicated_and_trimmed(self) -> None:
        record = self.database.remember("facts", "k", "v", tags=[" a ", "a", "b", ""])
        self.assertEqual(record.tags, ["a", "b"])

    def test_long_content_is_truncated(self) -> None:
        record = self.database.remember("facts", "big", "z" * 40_000)
        self.assertLess(len(record.content), 20_000)
        self.assertIn("truncated", record.content)


class SecretScrubbingTests(DatabaseTestCase):
    def test_token_shaped_values_are_scrubbed(self) -> None:
        record = self.database.remember(
            "notes", "leak", "the key is sk-abcdefghijklmnopqrstuvwxyz123456 do not share"
        )
        self.assertNotIn("sk-abcdefghijklmnopqrstuvwxyz123456", record.content)
        self.assertIn("[REDACTED]", record.content)

    def test_assignment_lines_with_secret_names_are_scrubbed(self) -> None:
        content = "HOST=127.0.0.1\nAPI_KEY=super-secret\nPASSWORD=hunter2\nPORT=8080"
        record = self.database.remember("notes", "env-dump", content)
        self.assertIn("HOST=127.0.0.1", record.content)
        self.assertIn("PORT=8080", record.content)
        self.assertNotIn("super-secret", record.content)
        self.assertNotIn("hunter2", record.content)

    def test_private_key_blocks_are_scrubbed(self) -> None:
        record = self.database.remember(
            "notes", "key", "-----BEGIN RSA PRIVATE KEY-----\nMIIEow...\n"
        )
        self.assertNotIn("BEGIN RSA PRIVATE KEY", record.content)

    def test_scrub_secrets_keeps_harmless_text(self) -> None:
        text = "nothing sensitive here, just prose about 127.0.0.1 and port 8080"
        self.assertEqual(scrub_secrets(text), text)


class SqlInjectionTests(DatabaseTestCase):
    def test_injection_in_memory_key_is_stored_as_data(self) -> None:
        hostile_key = "'; DROP TABLE memories; --"
        record = self.database.remember("attack", hostile_key, "payload")
        self.assertEqual(record.key, hostile_key)
        self.assertIn("memories", self.database.table_names())
        self.assertEqual(self.database.get_memory("attack", hostile_key).content, "payload")

    def test_injection_in_task_title_is_stored_as_data(self) -> None:
        hostile = "x'); DELETE FROM tasks; --"
        task = self.database.create_task(hostile)
        self.assertEqual(task.title, hostile)
        self.assertIn("tasks", self.database.table_names())
        self.assertEqual(self.database.row_counts()["tasks"], 1)

    def test_injection_in_search_is_inert(self) -> None:
        self.database.remember("facts", "safe", "ordinary content")
        for hostile in ("' OR 1=1 --", "%", "_", "'; DROP TABLE memories; --"):
            with self.subTest(hostile=hostile):
                self.database.search_memories(hostile)
                self.assertIn("memories", self.database.table_names())
        self.assertEqual(self.database.row_counts()["memories"], 1)

    def test_wildcards_in_search_are_not_expanded_into_sql(self) -> None:
        self.database.remember("facts", "a", "alpha")
        self.database.remember("facts", "b", "beta")
        # A bare '%' is a LIKE wildcard, so it legitimately matches both rows;
        # the point is that the statement still executes as one parameterized
        # query and the table survives.
        self.assertEqual(len(self.database.search_memories("%")), 2)
        self.assertEqual(len(self.database.search_memories("alpha")), 1)

    def test_injection_in_status_filter_is_rejected(self) -> None:
        with self.assertRaises(DatabaseError):
            self.database.list_tasks(status="pending' OR '1'='1")
        self.assertIn("tasks", self.database.table_names())

    def test_injection_in_tool_name_is_stored_as_data(self) -> None:
        hostile = "tool'); DROP TABLE tool_executions; --"
        execution = self.database.record_tool_execution(hostile, {"a": 1}, "ok", "success", 5)
        self.assertEqual(execution.tool_name, hostile)
        self.assertIn("tool_executions", self.database.table_names())


class ConversationTests(DatabaseTestCase):
    def test_create_and_append_messages(self) -> None:
        conversation = self.database.create_conversation("planning session")
        first = self.database.add_message(conversation.id, "user", "what is on disk?")
        second = self.database.add_message(conversation.id, "assistant", "let me look")
        messages = self.database.get_messages(conversation.id)
        self.assertEqual([message.id for message in messages], [first.id, second.id])
        self.assertEqual(messages[0].role, "user")

    def test_adding_a_message_touches_the_conversation(self) -> None:
        conversation = self.database.create_conversation("chat")
        self.database.add_message(conversation.id, "user", "hi")
        refreshed = self.database.get_conversation(conversation.id)
        self.assertGreaterEqual(refreshed.updated_at, conversation.updated_at)

    def test_rejects_unknown_role(self) -> None:
        conversation = self.database.create_conversation("chat")
        with self.assertRaises(DatabaseError):
            self.database.add_message(conversation.id, "root", "escalate")

    def test_rejects_message_for_missing_conversation(self) -> None:
        with self.assertRaises(DatabaseError):
            self.database.add_message(9999, "user", "orphan")

    def test_foreign_key_is_enforced_on_direct_insert(self) -> None:
        with self.assertRaises(sqlite3.IntegrityError):
            with self.database.transaction() as connection:
                connection.execute(
                    "INSERT INTO messages (conversation_id, role, content, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (4242, "user", "orphan", "2026-01-01T00:00:00+00:00"),
                )

    def test_deleting_a_conversation_cascades(self) -> None:
        conversation = self.database.create_conversation("chat")
        self.database.add_message(conversation.id, "user", "hi")
        with self.database.transaction() as connection:
            connection.execute("DELETE FROM conversations WHERE id = ?", (conversation.id,))
        self.assertEqual(self.database.row_counts()["messages"], 0)


class ToolExecutionAndApprovalTests(DatabaseTestCase):
    def test_records_tool_metadata(self) -> None:
        execution = self.database.record_tool_execution(
            "filesystem.read_file",
            {"path": "/home/user/Projects/app/main.py"},
            "read 120 bytes",
            "success",
            7,
        )
        self.assertEqual(execution.tool_name, "filesystem.read_file")
        self.assertEqual(execution.arguments["path"], "/home/user/Projects/app/main.py")
        self.assertEqual(execution.duration_ms, 7)

    def test_links_to_a_task_and_survives_task_deletion(self) -> None:
        task = self.database.create_task("linked")
        execution = self.database.record_tool_execution("t", {}, "ok", "success", 1, task.id)
        self.assertEqual(execution.task_id, task.id)
        with self.database.transaction() as connection:
            connection.execute("DELETE FROM tasks WHERE id = ?", (task.id,))
        refreshed = self.database.list_tool_executions()[0]
        self.assertIsNone(refreshed.task_id)

    def test_rejects_non_serializable_arguments(self) -> None:
        with self.assertRaises(DatabaseError):
            self.database.record_tool_execution("t", {"obj": object()}, "ok", "success", 1)

    def test_result_summary_is_scrubbed(self) -> None:
        execution = self.database.record_tool_execution(
            "t", {}, "saw ghp_abcdefghijklmnopqrstuvwxyz1234 in output", "success", 1
        )
        self.assertNotIn("ghp_abcdefghijklmnopqrstuvwxyz1234", execution.result_summary)

    def test_approval_lifecycle(self) -> None:
        approval = self.database.create_approval("terminal.execute", "run a build")
        self.assertEqual(approval.status, "pending")
        self.assertIsNone(approval.resolved_at)
        resolved = self.database.resolve_approval(approval.id, "rejected")
        self.assertEqual(resolved.status, "rejected")
        self.assertIsNotNone(resolved.resolved_at)
        self.assertEqual(len(self.database.list_approvals(status="rejected")), 1)

    def test_approval_cannot_be_resolved_to_pending(self) -> None:
        approval = self.database.create_approval("x")
        with self.assertRaises(DatabaseError):
            self.database.resolve_approval(approval.id, "pending")

    def test_resolving_a_missing_approval_raises(self) -> None:
        with self.assertRaises(DatabaseError):
            self.database.resolve_approval(9999, "approved")


class TransactionTests(DatabaseTestCase):
    def test_rollback_on_exception(self) -> None:
        with self.assertRaises(RuntimeError):
            with self.database.transaction() as connection:
                connection.execute(
                    "INSERT INTO tasks (title, description, status, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    ("rolled back", "", "pending", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
                )
                raise RuntimeError("abort")
        self.assertEqual(self.database.row_counts()["tasks"], 0)

    def test_commit_on_success(self) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO tasks (title, description, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("committed", "", "pending", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
            )
        self.assertEqual(self.database.row_counts()["tasks"], 1)


class TaskManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.path = Path(self._temporary.name) / "agent.sqlite3"
        self.previous = get_audit_logger()
        self.logger = AuditLogger()
        set_audit_logger(self.logger)
        self.manager = TaskManager.open(self.path, create=True)

    def tearDown(self) -> None:
        self.manager.close()
        set_audit_logger(self.previous)
        self._temporary.cleanup()

    def test_task_lifecycle_through_the_manager(self) -> None:
        task = self.manager.create_task("audit the host", "run the system command")
        self.assertEqual(self.manager.get_task(task.id).title, "audit the host")
        self.manager.set_task_status(task.id, TaskStatus.IN_PROGRESS)
        done = self.manager.complete_task(task.id)
        self.assertEqual(done.status, TaskStatus.DONE)
        self.assertEqual(len(self.manager.list_tasks()), 1)

    def test_memory_lifecycle_through_the_manager(self) -> None:
        self.manager.remember("facts", "gpu", "RTX 4050", tags=["hardware"])
        self.assertEqual(self.manager.recall("facts", "gpu").content, "RTX 4050")
        self.assertEqual(len(self.manager.list_memories(tag="hardware")), 1)
        self.assertEqual(len(self.manager.search_memories("4050")), 1)
        self.assertTrue(self.manager.forget("facts", "gpu"))

    def test_memory_content_is_not_logged(self) -> None:
        self.manager.remember("facts", "secret-note", "SENSITIVE-BODY-TEXT-77")
        for entry in self.logger.entries():
            self.assertNotIn("SENSITIVE-BODY-TEXT-77", entry.as_line())

    def test_manager_calls_are_audited(self) -> None:
        self.manager.create_task("audited task")
        names = [entry.tool_name for entry in self.logger.entries()]
        self.assertIn("tasks.create_task", names)

    def test_audit_mirroring_writes_tool_execution_rows(self) -> None:
        manager = TaskManager.open(
            self.path, create=True, mirror_audit_to_database=True, audit_logger=self.logger
        )
        with manager:
            manager.create_task("mirrored")
            rows = manager.list_tool_executions()
        self.assertTrue(any(row.tool_name == "tasks.create_task" for row in rows))

    def test_flush_audit_log_persists_entries(self) -> None:
        self.manager.create_task("flushed")
        written = self.manager.flush_audit_log()
        self.assertGreater(written, 0)
        rows = self.manager.list_tool_executions()
        self.assertTrue(rows)

    def test_storage_report(self) -> None:
        report = self.manager.storage_report()
        self.assertEqual(report["schema_version"], SCHEMA_VERSION)
        self.assertEqual(report["path"], str(self.path))
        self.assertIn("task_counts", report)

    def test_opening_without_create_requires_an_existing_file(self) -> None:
        missing = Path(self._temporary.name) / "other" / "agent.sqlite3"
        with self.assertRaises(DatabaseNotInitialized):
            TaskManager.open(missing, create=False)


class DefaultLocationTests(unittest.TestCase):
    def test_data_directory_honours_the_override(self) -> None:
        from omarchy_agent.memory.database import default_data_directory, default_database_path

        previous = os.environ.get("OMARCHY_AGENT_DATA_DIR")
        try:
            os.environ["OMARCHY_AGENT_DATA_DIR"] = "/tmp/omarchy-agent-test-dir"
            self.assertEqual(
                default_data_directory(), Path("/tmp/omarchy-agent-test-dir")
            )
            self.assertEqual(
                default_database_path(),
                Path("/tmp/omarchy-agent-test-dir/agent.sqlite3"),
            )
        finally:
            if previous is None:
                os.environ.pop("OMARCHY_AGENT_DATA_DIR", None)
            else:
                os.environ["OMARCHY_AGENT_DATA_DIR"] = previous

    def test_default_path_is_under_local_share(self) -> None:
        from omarchy_agent.memory.database import default_database_path

        previous_data = os.environ.pop("OMARCHY_AGENT_DATA_DIR", None)
        previous_xdg = os.environ.pop("XDG_DATA_HOME", None)
        try:
            self.assertEqual(
                default_database_path(),
                Path.home() / ".local" / "share" / "omarchy-agent" / "agent.sqlite3",
            )
        finally:
            if previous_data is not None:
                os.environ["OMARCHY_AGENT_DATA_DIR"] = previous_data
            if previous_xdg is not None:
                os.environ["XDG_DATA_HOME"] = previous_xdg


if __name__ == "__main__":
    unittest.main()
