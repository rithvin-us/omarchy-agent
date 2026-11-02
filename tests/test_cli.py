"""Tests for the command-line interface.

LocalAI is always mocked here so the suite never depends on a running server,
and every command is pointed at a temporary database.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any
from unittest import mock

from omarchy_agent.cli import EXIT_NOT_INITIALIZED, EXIT_OK, build_parser, human_bytes, main
from omarchy_agent.integrations.localai import ConnectionStatus, ModelInfo


class FakeClient:
    """Stands in for :class:`LocalAIClient` with a reachable, model-less server."""

    def __init__(self, base_url: str = "http://127.0.0.1:8080/v1", **_kwargs: Any) -> None:
        self.base_url = base_url

    def check_connection(self) -> ConnectionStatus:
        return ConnectionStatus(
            base_url=self.base_url,
            reachable=True,
            status_code=200,
            latency_ms=3,
            model_count=1,
            models_ready=True,
        )

    def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(id="qwen3-4b-q8_0", owned_by="localai")]


def run_cli(arguments: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with mock.patch("omarchy_agent.cli.LocalAIClient", FakeClient):
        with redirect_stdout(out), redirect_stderr(err):
            code = main(arguments)
    return code, out.getvalue(), err.getvalue()


class CliTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.database = Path(self._temporary.name) / "data" / "agent.sqlite3"

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def arguments(self, *parts: str) -> list[str]:
        return ["--database", str(self.database), *parts]


class HelpAndVersionTests(CliTestCase):
    def test_no_command_prints_help(self) -> None:
        code, out, _err = run_cli([])
        self.assertEqual(code, EXIT_OK)
        self.assertIn("usage:", out)

    def test_parser_exposes_every_documented_command(self) -> None:
        parser = build_parser()
        actions = [
            action for action in parser._actions if action.dest == "command"  # noqa: SLF001
        ]
        choices = set(actions[0].choices or {})
        for command in (
            "audit",
            "doctor",
            "init",
            "system",
            "models",
            "tasks",
            "memory",
        ):
            with self.subTest(command=command):
                self.assertIn(command, choices)

    def test_help_exits_cleanly(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            run_cli(["--help"])
        self.assertEqual(raised.exception.code, 0)


class AuditCommandTests(CliTestCase):
    def test_text_output(self) -> None:
        code, out, _err = run_cli(self.arguments("audit"))
        self.assertEqual(code, EXIT_OK)
        self.assertIn("environment audit", out)
        self.assertIn("Tool availability", out)
        self.assertIn("Workspace policy", out)

    def test_json_output_is_valid(self) -> None:
        code, out, _err = run_cli(["--json", "audit"])
        self.assertEqual(code, EXIT_OK)
        payload = json.loads(out)
        self.assertIn("python", payload)
        self.assertIn("workspace_roots", payload)
        self.assertEqual(payload["max_file_read_bytes"], 128 * 1024)

    def test_no_environment_variables_are_dumped(self) -> None:
        _code, out, _err = run_cli(["--json", "audit"])
        payload = json.loads(out)
        self.assertNotIn("environ", payload)
        serialized = json.dumps(payload)
        for marker in ("PATH=", "HOME=", "SSH_AUTH_SOCK"):
            self.assertNotIn(marker, serialized)


class CapabilitiesCommandTests(CliTestCase):
    def test_lists_all_three_groups(self) -> None:
        code, out, _err = run_cli(["capabilities"])
        self.assertEqual(code, EXIT_OK)
        self.assertIn("Allowed in v0.1", out)
        self.assertIn("Planned -- not implemented", out)
        self.assertIn("Denied by project policy", out)

    def test_json_output(self) -> None:
        _code, out, _err = run_cli(["--json", "capabilities"])
        payload = json.loads(out)
        self.assertEqual(set(payload), {"allowed", "not_implemented", "denied"})


class InitCommandTests(CliTestCase):
    def test_creates_the_database_and_announces_what_it_will_do(self) -> None:
        code, out, _err = run_cli(self.arguments("init"))
        self.assertEqual(code, EXIT_OK)
        self.assertTrue(self.database.exists())
        self.assertIn("will create", out)
        self.assertIn("mode 0700", out)

    def test_second_run_preserves_data(self) -> None:
        run_cli(self.arguments("init"))
        run_cli(self.arguments("tasks", "--create", "keep me"))
        code, out, _err = run_cli(["--json", "--database", str(self.database), "init"])
        self.assertEqual(code, EXIT_OK)
        payload = json.loads(out)
        self.assertTrue(payload["already_existed"])
        self.assertFalse(payload["database_created"])
        self.assertEqual(payload["row_counts"]["tasks"], 1)


class SystemCommandTests(CliTestCase):
    def test_text_output_has_every_section(self) -> None:
        code, out, _err = run_cli(["system", "--top", "3"])
        self.assertEqual(code, EXIT_OK)
        for section in ("Operating system", "CPU", "Memory", "Storage", "GPU", "Processes"):
            with self.subTest(section=section):
                self.assertIn(section, out)

    def test_json_output(self) -> None:
        _code, out, _err = run_cli(["--json", "system"])
        payload = json.loads(out)
        self.assertEqual(set(payload), {"os", "cpu", "memory", "storage", "gpu", "processes"})
        self.assertGreater(payload["memory"]["total_bytes"], 0)


class ModelsCommandTests(CliTestCase):
    def test_lists_models(self) -> None:
        code, out, _err = run_cli(["models"])
        self.assertEqual(code, EXIT_OK)
        self.assertIn("qwen3-4b-q8_0", out)

    def test_json_output(self) -> None:
        _code, out, _err = run_cli(["--json", "models"])
        payload = json.loads(out)
        self.assertTrue(payload["connection"]["reachable"])
        self.assertEqual(payload["models"][0]["id"], "qwen3-4b-q8_0")


class TasksAndMemoryCommandTests(CliTestCase):
    def test_tasks_requires_initialization(self) -> None:
        code, _out, err = run_cli(self.arguments("tasks"))
        self.assertEqual(code, EXIT_NOT_INITIALIZED)
        self.assertIn("init", err)

    def test_memory_requires_initialization(self) -> None:
        code, _out, err = run_cli(self.arguments("memory"))
        self.assertEqual(code, EXIT_NOT_INITIALIZED)
        self.assertIn("init", err)

    def test_task_create_list_and_complete(self) -> None:
        run_cli(self.arguments("init"))
        code, out, _err = run_cli(self.arguments("tasks", "--create", "inspect the host"))
        self.assertEqual(code, EXIT_OK)
        self.assertIn("created task", out)
        self.assertIn("inspect the host", out)

        _code, out, _err = run_cli(["--json", "--database", str(self.database), "tasks"])
        payload = json.loads(out)
        task_id = payload["tasks"][0]["id"]

        code, out, _err = run_cli(self.arguments("tasks", "--complete", str(task_id)))
        self.assertEqual(code, EXIT_OK)
        self.assertIn("done", out)

    def test_memory_remember_and_list(self) -> None:
        run_cli(self.arguments("init"))
        code, out, _err = run_cli(
            self.arguments("memory", "--remember", "facts", "gpu", "RTX 4050", "--tags", "hardware")
        )
        self.assertEqual(code, EXIT_OK)
        self.assertIn("stored memory facts/gpu", out)

        _code, out, _err = run_cli(
            ["--json", "--database", str(self.database), "memory", "--category", "facts"]
        )
        payload = json.loads(out)
        self.assertEqual(payload["memories"][0]["content"], "RTX 4050")
        self.assertEqual(payload["memories"][0]["tags"], ["hardware"])

    def test_audit_log_command(self) -> None:
        run_cli(self.arguments("init"))
        run_cli(self.arguments("tasks", "--create", "logged"))
        code, out, _err = run_cli(self.arguments("audit-log"))
        self.assertEqual(code, EXIT_OK)
        self.assertIn("Persisted tool executions", out)


class FilesAndGitCommandTests(CliTestCase):
    def test_files_list_inside_an_explicit_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "hello.txt").write_text("hi\n", encoding="utf-8")
            code, out, _err = run_cli(["files", "list", str(root), "--root", str(root)])
            self.assertEqual(code, EXIT_OK)
            self.assertIn("hello.txt", out)

    def test_files_read_refuses_outside_the_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            code, _out, err = run_cli(
                ["files", "read", "/etc/passwd", "--root", str(Path(directory).resolve())]
            )
            self.assertNotEqual(code, EXIT_OK)
            self.assertIn("PathSecurityError", err)

    def test_files_search_requires_a_query(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            code, _out, err = run_cli(["files", "search", str(root), "--root", str(root)])
            self.assertNotEqual(code, EXIT_OK)
            self.assertIn("--query", err)

    def test_git_status_outside_a_repository_fails_cleanly(self) -> None:
        code, _out, err = run_cli(["git", "status", "/etc"])
        self.assertNotEqual(code, EXIT_OK)
        self.assertIn("error:", err)


class DoctorCommandTests(CliTestCase):
    def test_reports_checks_and_warns_about_a_missing_database(self) -> None:
        code, out, _err = run_cli(self.arguments("doctor"))
        self.assertIn("Doctor", out)
        self.assertIn("package structure", out)
        self.assertIn("path traversal refusal", out)
        self.assertIn("permission policy", out)
        self.assertIn("WARN", out)
        self.assertIn(code, (EXIT_OK, 3))

    def test_passes_after_initialization(self) -> None:
        run_cli(self.arguments("init"))
        code, out, _err = run_cli(self.arguments("doctor"))
        self.assertEqual(code, EXIT_OK, msg=out)
        self.assertIn("0 error(s)", out)

    def test_json_output(self) -> None:
        run_cli(self.arguments("init"))
        _code, out, _err = run_cli(["--json", "--database", str(self.database), "doctor"])
        payload = json.loads(out)
        self.assertEqual(payload["errors"], 0)
        self.assertTrue(payload["checks"])


class PersistAuditFlagTests(CliTestCase):
    def test_bridge_audit_entries_are_persisted(self) -> None:
        run_cli(self.arguments("init"))
        code, out, _err = run_cli(self.arguments("--persist-audit", "system", "--top", "1"))
        self.assertEqual(code, EXIT_OK)
        self.assertIn("persisted", out)

        _code, out, _err = run_cli(
            ["--json", "--database", str(self.database), "audit-log"]
        )
        payload = json.loads(out)
        names = {row["tool_name"] for row in payload["tool_executions"]}
        self.assertIn("system.get_cpu_information", names)

    def test_warns_instead_of_creating_a_database(self) -> None:
        code, _out, err = run_cli(self.arguments("--persist-audit", "system", "--top", "1"))
        self.assertEqual(code, EXIT_OK)
        self.assertIn("could not persist audit entries", err)
        self.assertFalse(self.database.exists())

    def test_tasks_mirror_their_own_audit_entries(self) -> None:
        run_cli(self.arguments("init"))
        run_cli(self.arguments("tasks", "--create", "mirrored task"))
        _code, out, _err = run_cli(
            ["--json", "--database", str(self.database), "audit-log"]
        )
        payload = json.loads(out)
        names = {row["tool_name"] for row in payload["tool_executions"]}
        self.assertIn("tasks.create_task", names)


class ShowAuditFlagTests(CliTestCase):
    def test_audit_entries_can_be_printed(self) -> None:
        code, out, _err = run_cli(["--show-audit", "system", "--top", "1"])
        self.assertEqual(code, EXIT_OK)
        self.assertIn("Audit entries this run", out)
        self.assertIn("system.get_cpu_information", out)


class HelperTests(unittest.TestCase):
    def test_human_bytes(self) -> None:
        self.assertEqual(human_bytes(0), "0 B")
        self.assertEqual(human_bytes(None), "-")
        self.assertEqual(human_bytes(1536), "1.5 KiB")
        self.assertEqual(human_bytes(1024 * 1024), "1.0 MiB")


if __name__ == "__main__":
    unittest.main()
