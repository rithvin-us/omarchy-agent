"""Tests for the path policy, the permission policy and the audit logger."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from omarchy_agent.security.audit import (
    REDACTED,
    AuditLogger,
    audited,
    get_audit_logger,
    redact_arguments,
    redact_text,
    set_audit_logger,
    summarize_value,
)
from omarchy_agent.security.paths import (
    MAX_FILE_BYTES,
    PathPolicy,
    PathSecurityError,
    default_workspace_roots,
)
from omarchy_agent.security.permissions import (
    Capability,
    CapabilityNotImplemented,
    CapabilityState,
    PermissionDenied,
    default_permission_policy,
)


class PathPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name).resolve()
        self.policy = PathPolicy.from_paths([self.root])
        (self.root / "src").mkdir()
        (self.root / "src" / "main.py").write_text("print('hello')\n", encoding="utf-8")

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_allows_path_inside_root(self) -> None:
        resolved = self.policy.resolve_within_roots(self.root / "src" / "main.py")
        self.assertEqual(resolved, self.root / "src" / "main.py")

    def test_allows_the_root_itself(self) -> None:
        self.assertEqual(self.policy.resolve_within_roots(self.root), self.root)

    def test_rejects_path_outside_root(self) -> None:
        with self.assertRaises(PathSecurityError):
            self.policy.resolve_within_roots("/etc/passwd")

    def test_rejects_dotdot_traversal(self) -> None:
        attacks = [
            self.root / ".." / "escaped.txt",
            self.root / "src" / ".." / ".." / "escaped.txt",
            f"{self.root}/src/../../../../etc/passwd",
        ]
        for attack in attacks:
            with self.subTest(attack=str(attack)):
                with self.assertRaises(PathSecurityError):
                    self.policy.resolve_within_roots(attack, must_exist=False)

    def test_rejects_nul_byte_in_path(self) -> None:
        with self.assertRaises(PathSecurityError):
            self.policy.resolve_within_roots(f"{self.root}/src\x00/main.py", must_exist=False)

    def test_rejects_symlink_escaping_the_root(self) -> None:
        with tempfile.TemporaryDirectory() as outside:
            secret = Path(outside) / "outside.txt"
            secret.write_text("outside data", encoding="utf-8")
            link = self.root / "escape-link"
            link.symlink_to(secret)
            with self.assertRaises(PathSecurityError):
                self.policy.resolve_within_roots(link)

    def test_rejects_symlinked_directory_escaping_the_root(self) -> None:
        with tempfile.TemporaryDirectory() as outside:
            link = self.root / "outside-dir"
            link.symlink_to(Path(outside), target_is_directory=True)
            with self.assertRaises(PathSecurityError):
                self.policy.resolve_within_roots(link / "anything.txt", must_exist=False)

    def test_allows_symlink_that_stays_inside_the_root(self) -> None:
        link = self.root / "inside-link"
        link.symlink_to(self.root / "src" / "main.py")
        self.assertEqual(
            self.policy.resolve_within_roots(link), self.root / "src" / "main.py"
        )

    def test_rejects_path_inside_blocked_directory(self) -> None:
        git_directory = self.root / ".git"
        git_directory.mkdir()
        (git_directory / "config").write_text("[core]\n", encoding="utf-8")
        with self.assertRaises(PathSecurityError):
            self.policy.resolve_within_roots(git_directory / "config")

    def test_rejects_blocked_directory_itself(self) -> None:
        (self.root / "node_modules").mkdir()
        with self.assertRaises(PathSecurityError):
            self.policy.resolve_within_roots(self.root / "node_modules")

    def test_blocked_directory_names(self) -> None:
        for name in (".git", "node_modules", ".venv", "venv", "__pycache__", "build", ".cache"):
            with self.subTest(name=name):
                self.assertTrue(PathPolicy.is_blocked_directory_name(name))
        for name in ("src", "omarchy_agent", "tests", "docs"):
            with self.subTest(name=name):
                self.assertFalse(PathPolicy.is_blocked_directory_name(name))

    def test_sensitive_file_names(self) -> None:
        sensitive = [
            ".env",
            ".env.local",
            ".env.production",
            "id_rsa",
            "id_ed25519",
            "server.pem",
            "private.key",
            ".netrc",
            ".git-credentials",
            "credentials.json",
            "vault.kdbx",
            "logins.json",
            "key4.db",
            ".npmrc",
        ]
        for name in sensitive:
            with self.subTest(name=name):
                self.assertTrue(PathPolicy.is_sensitive_file_name(name))

    def test_example_configuration_files_are_allowed(self) -> None:
        harmless = [".env.example", ".env.sample", ".env.template", "config.yaml.example"]
        for name in harmless:
            with self.subTest(name=name):
                self.assertFalse(PathPolicy.is_sensitive_file_name(name))

    def test_sensitive_file_is_refused_by_the_policy(self) -> None:
        secret = self.root / ".env"
        secret.write_text("API_KEY=abcdef\n", encoding="utf-8")
        with self.assertRaises(PathSecurityError):
            self.policy.resolve_within_roots(secret)

    def test_missing_path_is_refused_when_existence_required(self) -> None:
        with self.assertRaises(PathSecurityError):
            self.policy.resolve_within_roots(self.root / "nope.txt")
        self.assertTrue(
            self.policy.resolve_within_roots(self.root / "nope.txt", must_exist=False)
        )

    def test_empty_root_set_refuses_everything(self) -> None:
        empty = PathPolicy(roots=())
        with self.assertRaises(PathSecurityError):
            empty.resolve_within_roots(self.root)

    def test_default_max_file_size(self) -> None:
        self.assertEqual(MAX_FILE_BYTES, 128 * 1024)
        self.assertEqual(self.policy.max_file_bytes, 128 * 1024)

    def test_default_roots_honour_environment_override(self) -> None:
        previous = os.environ.get("OMARCHY_AGENT_WORKSPACE_ROOTS")
        try:
            os.environ["OMARCHY_AGENT_WORKSPACE_ROOTS"] = str(self.root)
            self.assertEqual(default_workspace_roots(), (self.root,))
            os.environ["OMARCHY_AGENT_WORKSPACE_ROOTS"] = "/nonexistent-root-xyz"
            self.assertEqual(default_workspace_roots(), ())
        finally:
            if previous is None:
                os.environ.pop("OMARCHY_AGENT_WORKSPACE_ROOTS", None)
            else:
                os.environ["OMARCHY_AGENT_WORKSPACE_ROOTS"] = previous


class PermissionPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = default_permission_policy()

    def test_implemented_capabilities_are_allowed(self) -> None:
        for capability in (
            Capability.FS_READ,
            Capability.FS_LIST,
            Capability.FS_SEARCH,
            Capability.SYSTEM_INFO,
            Capability.GIT_INSPECT,
            Capability.LOCALAI_INSPECT,
            Capability.MEMORY_WRITE,
            Capability.TASK_WRITE,
            Capability.AUDIT_WRITE,
        ):
            with self.subTest(capability=capability):
                self.assertTrue(self.policy.is_allowed(capability))
                self.policy.require(capability)

    def test_planned_capabilities_raise_not_implemented(self) -> None:
        for capability in (
            Capability.TERMINAL_EXEC,
            Capability.FS_WRITE,
            Capability.FS_DELETE,
            Capability.BROWSER_CONTROL,
            Capability.DESKTOP_INPUT,
            Capability.DESKTOP_SCREENSHOT,
            Capability.LOCALAI_INFERENCE,
        ):
            with self.subTest(capability=capability):
                self.assertFalse(self.policy.is_allowed(capability))
                with self.assertRaises(CapabilityNotImplemented):
                    self.policy.require(capability)

    def test_forbidden_capabilities_are_denied(self) -> None:
        for capability in (
            Capability.PRIVILEGE_ESCALATION,
            Capability.PACKAGE_INSTALL,
            Capability.DOCKER_CONTROL,
            Capability.SERVICE_INSTALL,
            Capability.SECRET_READ,
            Capability.NETWORK_BIND_PUBLIC,
        ):
            with self.subTest(capability=capability):
                with self.assertRaises(PermissionDenied) as raised:
                    self.policy.require(capability)
                self.assertNotIsInstance(raised.exception, CapabilityNotImplemented)

    def test_denied_is_not_reported_as_not_implemented(self) -> None:
        self.assertIs(
            self.policy.state_of(Capability.DOCKER_CONTROL), CapabilityState.DENIED
        )
        self.assertIs(
            self.policy.state_of(Capability.TERMINAL_EXEC), CapabilityState.NOT_IMPLEMENTED
        )

    def test_every_capability_has_a_state(self) -> None:
        inventory = self.policy.inventory()
        self.assertEqual(len(inventory), len(list(Capability)))
        for info in inventory:
            self.assertIn(info.state, tuple(CapabilityState))
            self.assertTrue(info.note, f"{info.capability} has no description")

    def test_grouped_inventory_has_all_three_groups(self) -> None:
        grouped = self.policy.grouped_inventory()
        self.assertEqual(set(grouped), {"allowed", "not_implemented", "denied"})
        self.assertTrue(grouped["allowed"])
        self.assertTrue(grouped["not_implemented"])
        self.assertTrue(grouped["denied"])


class RedactionTests(unittest.TestCase):
    def test_sensitive_argument_names_are_redacted(self) -> None:
        redacted = redact_arguments(
            {
                "password": "hunter2",
                "api_key": "abc123",
                "ACCESS_TOKEN": "xyz",
                "private_key": "-----BEGIN RSA PRIVATE KEY-----",
                "path": "/home/user/Projects/app",
                "limit": 10,
            }
        )
        for key in ("password", "api_key", "ACCESS_TOKEN", "private_key"):
            with self.subTest(key=key):
                self.assertEqual(redacted[key], REDACTED)
        self.assertEqual(redacted["path"], "/home/user/Projects/app")
        self.assertEqual(redacted["limit"], 10)

    def test_credential_shaped_values_are_redacted_by_shape(self) -> None:
        samples = [
            "sk-abcdefghijklmnopqrstuvwxyz123456",
            "ghp_abcdefghijklmnopqrstuvwxyz1234",
            "xoxb-1234567890-abcdefghijklm",
            "AKIAIOSFODNN7EXAMPLE",
            "-----BEGIN OPENSSH PRIVATE KEY-----",
        ]
        for sample in samples:
            with self.subTest(sample=sample[:12]):
                self.assertIn(REDACTED, redact_text(f"value is {sample} here"))

    def test_nested_structures_are_redacted(self) -> None:
        redacted = redact_arguments({"config": {"token": "secret-value", "host": "127.0.0.1"}})
        self.assertEqual(redacted["config"]["token"], REDACTED)
        self.assertEqual(redacted["config"]["host"], "127.0.0.1")

    def test_long_values_are_truncated(self) -> None:
        redacted = redact_arguments({"blob": "a" * 5000})
        self.assertLess(len(redacted["blob"]), 400)
        self.assertIn("truncated", redacted["blob"])

    def test_summarize_value_never_includes_string_content(self) -> None:
        summary = summarize_value("super secret file body")
        self.assertNotIn("secret", summary)
        self.assertIn("chars", summary)


class AuditLoggerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous = get_audit_logger()
        self.logger = AuditLogger(max_entries=50)
        set_audit_logger(self.logger)

    def tearDown(self) -> None:
        set_audit_logger(self.previous)

    def test_successful_call_is_recorded(self) -> None:
        @audited("test.add", Capability.SYSTEM_INFO)
        def add(left: int, right: int) -> int:
            return left + right

        self.assertEqual(add(2, 3), 5)
        entries = self.logger.entries()
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry.tool_name, "test.add")
        self.assertEqual(entry.status, "success")
        self.assertEqual(entry.arguments, {"left": 2, "right": 3})
        self.assertEqual(entry.capability, Capability.SYSTEM_INFO.value)
        self.assertGreaterEqual(entry.duration_ms, 0)
        self.assertTrue(entry.timestamp.endswith("+00:00"))

    def test_failing_call_is_recorded_as_error_and_reraises(self) -> None:
        @audited("test.boom", Capability.SYSTEM_INFO)
        def boom() -> None:
            raise ValueError("bad input")

        with self.assertRaises(ValueError):
            boom()
        entry = self.logger.entries()[-1]
        self.assertEqual(entry.status, "error")
        self.assertIn("ValueError", entry.result_summary)

    def test_denied_capability_is_recorded_as_denied(self) -> None:
        @audited("test.install", Capability.PACKAGE_INSTALL)
        def install() -> None:  # pragma: no cover - must never run
            raise AssertionError("the denied function body executed")

        with self.assertRaises(PermissionDenied):
            install()
        entry = self.logger.entries()[-1]
        self.assertEqual(entry.status, "denied")
        self.assertEqual(entry.capability, Capability.PACKAGE_INSTALL.value)

    def test_not_implemented_capability_is_recorded_as_denied(self) -> None:
        @audited("test.exec", Capability.TERMINAL_EXEC)
        def execute(command: str) -> None:  # pragma: no cover - must never run
            raise AssertionError("the planned function body executed")

        with self.assertRaises(CapabilityNotImplemented):
            execute("rm -rf /")
        entry = self.logger.entries()[-1]
        self.assertEqual(entry.status, "denied")
        self.assertEqual(entry.arguments["command"], "rm -rf /")

    def test_secret_arguments_are_redacted_in_the_log(self) -> None:
        @audited("test.login", Capability.SYSTEM_INFO)
        def login(user: str, password: str) -> bool:
            return bool(user and password)

        login("alice", "hunter2")
        entry = self.logger.entries()[-1]
        self.assertEqual(entry.arguments["password"], REDACTED)
        self.assertNotIn("hunter2", entry.as_line())

    def test_explicitly_redacted_parameters(self) -> None:
        @audited("test.store", Capability.MEMORY_WRITE, redact_params=("content",))
        def store(key: str, content: str) -> int:
            return len(content)

        store("note", "this body must not be logged")
        entry = self.logger.entries()[-1]
        self.assertEqual(entry.arguments["content"], REDACTED)
        self.assertNotIn("must not be logged", entry.as_line())

    def test_ring_buffer_is_bounded(self) -> None:
        logger = AuditLogger(max_entries=5)
        for index in range(20):
            logger.record(f"tool.{index}", {}, "success", "ok", 1)
        self.assertEqual(len(logger.entries()), 5)
        self.assertEqual(logger.entries()[-1].tool_name, "tool.19")

    def test_jsonl_sink_writes_one_line_per_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "audit.log"
            logger = AuditLogger()
            self.assertTrue(logger.add_jsonl_file_sink(target))
            logger.record("tool.one", {"path": "/tmp/x"}, "success", "ok", 3)
            logger.record("tool.two", {"token": "abc"}, "denied", "nope", 1)
            lines = target.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            second = json.loads(lines[1])
            self.assertEqual(second["arguments"]["token"], REDACTED)
            self.assertEqual(oct(target.stat().st_mode & 0o777), "0o600")

    def test_jsonl_sink_refuses_a_missing_directory(self) -> None:
        logger = AuditLogger()
        self.assertFalse(
            logger.add_jsonl_file_sink("/nonexistent-directory-xyz/audit.log")
        )

    def test_a_broken_sink_does_not_break_the_call(self) -> None:
        def broken(_entry: object) -> None:
            raise RuntimeError("sink exploded")

        self.logger.add_sink(broken)

        @audited("test.ok", Capability.SYSTEM_INFO)
        def ok() -> str:
            return "fine"

        self.assertEqual(ok(), "fine")
        self.assertTrue(self.logger.sink_errors)


if __name__ == "__main__":
    unittest.main()
