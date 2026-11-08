"""Socket-free tests for repository context and read-only model tools."""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from omarchy_agent.core.chat import ChatSession
from omarchy_agent.core.terminal_ui import TerminalUI
from omarchy_agent.security.paths import PathPolicy


class ChatSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "README.md").write_text("Widget parser lives in parser.py\n", encoding="utf-8")
        (self.root / "parser.py").write_text(
            "def parse_widget(value):\n    return value\n", encoding="utf-8"
        )
        self.policy = PathPolicy.from_paths([self.root])
        self.policy_patch = mock.patch(
            "omarchy_agent.core.chat.default_policy", return_value=self.policy
        )
        self.policy_patch.start()
        self.addCleanup(self.policy_patch.stop)
        self.session = ChatSession(object(), self.root)  # type: ignore[arg-type]

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_selects_bounded_relevant_context(self) -> None:
        context, paths = self.session.context_for("explain parser widget")
        self.assertIn("parse_widget", context)
        self.assertTrue(any(path.endswith("parser.py") for path in paths))
        self.assertEqual(paths, self.session.latest_context_files)

    def test_read_tool_uses_repository_path_policy(self) -> None:
        result = self.session._dispatch_tool(
            "read_file", '{"path": "' + str(self.root / "parser.py") + '"}'
        )
        self.assertIn("parse_widget", result["result"]["content"])

    def test_read_tool_refuses_path_outside_repository(self) -> None:
        result = self.session._dispatch_tool("read_file", '{"path": "/etc/passwd"}')
        self.assertIn("error", result)

    def test_unknown_tool_is_never_dispatched(self) -> None:
        result = self.session._dispatch_tool("delete_file", '{"path":"x"}')
        self.assertIn("unknown tool", result["error"])

    def _approving_session(self) -> ChatSession:
        return ChatSession(object(), self.root, on_confirm_command=lambda *_: True)  # type: ignore[arg-type]

    def test_write_and_edit_are_refused_without_approval(self) -> None:
        write = self.session._dispatch_tool("write_file", '{"path": "n.txt", "content": "x"}')
        edit = self.session._dispatch_tool(
            "edit_file", '{"path": "parser.py", "target": "def parse_widget", "replacement": "y"}'
        )
        self.assertIn("rejected", write["error"])
        self.assertIn("rejected", edit["error"])
        self.assertFalse((self.root / "n.txt").exists())
        self.assertIn("parse_widget", (self.root / "parser.py").read_text(encoding="utf-8"))

    def test_approval_prompt_does_not_leak_file_body(self) -> None:
        seen: list[str] = []
        session = ChatSession(
            object(),  # type: ignore[arg-type]
            self.root,
            on_confirm_command=lambda text, _cwd: seen.append(text) or False,
        )
        session._dispatch_tool("write_file", '{"path": "n.txt", "content": "SECRETBODY"}')
        self.assertIn("n.txt", seen[0])
        self.assertNotIn("SECRETBODY", seen[0])

    def test_write_file_tool_creates_file(self) -> None:
        payload = '{"path": "new_file.txt", "content": "hello world"}'
        result = self._approving_session()._dispatch_tool("write_file", payload)
        self.assertIn("result", result)
        self.assertTrue((self.root / "new_file.txt").exists())
        self.assertEqual((self.root / "new_file.txt").read_text(encoding="utf-8"), "hello world")

    def test_write_file_tool_refuses_outside_repo(self) -> None:
        result = self.session._dispatch_tool(
            "write_file", '{"path": "/etc/danger.txt", "content": "danger"}'
        )
        self.assertIn("error", result)

    def test_edit_file_tool_modifies_file(self) -> None:
        payload = (
            '{"path": "parser.py", "target": "def parse_widget", '
            '"replacement": "def custom_widget"}'
        )
        result = self._approving_session()._dispatch_tool("edit_file", payload)
        self.assertIn("result", result)
        self.assertIn("custom_widget", (self.root / "parser.py").read_text(encoding="utf-8"))

    def test_edit_file_tool_fails_if_target_not_found(self) -> None:
        payload = '{"path": "parser.py", "target": "missing_token", "replacement": "new_token"}'
        result = self.session._dispatch_tool("edit_file", payload)
        self.assertIn("error", result)

    def test_execute_command_tool_confirmed(self) -> None:
        session = ChatSession(
            object(),  # type: ignore[arg-type]
            self.root,
            on_confirm_command=lambda cmd, cwd: True,
        )
        result = session._dispatch_tool("execute_command", '{"command": "echo test_ok"}')
        self.assertIn("result", result)
        self.assertEqual(result["result"]["returncode"], 0)
        self.assertIn("test_ok", result["result"]["stdout"])

    def test_execute_command_tool_rejected(self) -> None:
        session = ChatSession(
            object(),  # type: ignore[arg-type]
            self.root,
            on_confirm_command=lambda cmd, cwd: False,
        )
        result = session._dispatch_tool("execute_command", '{"command": "echo test_fail"}')
        self.assertIn("error", result)
        self.assertIn("rejected", result["error"])

    def test_plan_turn_creates_tasks_in_manager(self) -> None:
        manager = mock.MagicMock()
        created_task = mock.MagicMock(id=1, title="Step 1: Test", status="pending")
        manager.database.create_task.return_value = created_task

        plan_output = (
            "Here is the plan:\n"
            "```json\n"
            '[{"title": "Step 1: Test", "description": "Execute test"}]\n'
            "```"
        )
        with mock.patch.object(self.session, "run_turn", return_value=plan_output):
            answer, tasks = self.session.plan_turn("Test goal", manager=manager)
            self.assertEqual(answer, plan_output)
            self.assertEqual(tasks, [created_task])
            manager.database.create_task.assert_called_once_with(
                title="Step 1: Test", description="Execute test"
            )

    def test_invalid_and_extra_tool_arguments_are_rejected(self) -> None:
        malformed = self.session._dispatch_tool("read_file", "not json")
        extra = self.session._dispatch_tool(
            "read_file", '{"path":"' + str(self.root / "parser.py") + '","write":true}'
        )
        self.assertIn("valid JSON", malformed["error"])
        self.assertIn("unexpected", extra["error"])


class TerminalUITests(unittest.TestCase):
    def setUp(self) -> None:
        self.stream = io.StringIO()
        self.ui = TerminalUI(stream=self.stream)

    def test_header_non_tty(self) -> None:
        self.ui.header("test-model", "/tmp/repo")
        output = self.stream.getvalue()
        self.assertIn("Omarchy Agent · LocalAI", output)
        self.assertIn("test-model", output)
        self.assertIn("/help · /files · /clear · /exit", output)

    def test_status_messages(self) -> None:
        self.ui.status("tool: read_file parser.py")
        output = self.stream.getvalue()
        self.assertIn("Calling read-only tool · read_file parser.py", output)
        self.ui.status("context: parser.py")
        self.assertEqual(self.ui._phase, "Repository context ready · parser.py")
        self.ui.status("tool result ready: done")
        self.assertEqual(self.ui._phase, "Read-only tool finished · done")
        self.ui.status("response complete")
        self.assertEqual(self.ui._phase, "Formatting the answer")

    def test_write_delta_and_finish_turn(self) -> None:
        self.ui.begin_turn()
        self.ui.write_delta("hello from model\n")
        self.ui.finish_turn("hello from model\n")
        output = self.stream.getvalue()
        self.assertIn("hello from model", output)
        self.assertIn("tokens", output)

    def test_update_usage(self) -> None:
        self.ui.update_usage(
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "estimated": False}
        )
        self.assertEqual(self.ui._usage["prompt_tokens"], 10)
        self.assertEqual(self.ui._usage["total_tokens"], 15)
        self.assertFalse(self.ui._usage["estimated"])

    def test_end_turn_resets_state(self) -> None:
        self.ui.begin_turn()
        self.ui.write_delta("partial response")
        self.ui.end_turn()
        self.assertEqual(self.ui._answer, "")

    def test_error_writes_to_stderr(self) -> None:
        stderr = io.StringIO()
        with mock.patch("sys.stderr", stderr):
            self.ui.error(ValueError("bad value"))
        self.assertIn("error: ValueError: bad value", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()


class BrowserToolTests(unittest.TestCase):
    def test_non_http_url_is_refused(self) -> None:
        from omarchy_agent.bridge.browser import BrowserError, validate_url

        with self.assertRaises(BrowserError):
            validate_url("file:///etc/passwd")
        self.assertEqual(validate_url("https://example.com"), "https://example.com")
