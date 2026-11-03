"""Tests for the terminal execution bridge."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from omarchy_agent.bridge.terminal import TerminalError, execute_command
from omarchy_agent.security.paths import PathPolicy, PathSecurityError


class TerminalBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.policy = PathPolicy.from_paths([self.root])

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_executes_simple_shell_command(self) -> None:
        res = execute_command("echo hello_omarchy", cwd=self.root, policy=self.policy)
        self.assertEqual(res.returncode, 0)
        self.assertIn("hello_omarchy", res.stdout)
        self.assertFalse(res.timed_out)
        self.assertEqual(res.cwd, str(self.root))

    def test_captures_nonzero_returncode(self) -> None:
        res = execute_command("exit 42", cwd=self.root, policy=self.policy)
        self.assertEqual(res.returncode, 42)

    def test_refuses_empty_command(self) -> None:
        with self.assertRaises(TerminalError):
            execute_command("   ", cwd=self.root, policy=self.policy)

    def test_refuses_working_directory_outside_roots(self) -> None:
        with self.assertRaises(PathSecurityError):
            execute_command("ls", cwd="/etc", policy=self.policy)

    def test_handles_timeout(self) -> None:
        res = execute_command("sleep 5", cwd=self.root, timeout_seconds=0.1, policy=self.policy)
        self.assertTrue(res.timed_out)

    def test_redacts_sensitive_token_in_output(self) -> None:
        res = execute_command(
            "echo token sk-1234567890abcdef1234567890abcdef", cwd=self.root, policy=self.policy
        )
        self.assertNotIn("sk-1234567890abcdef1234567890abcdef", res.stdout)
        self.assertIn("[REDACTED]", res.stdout)


if __name__ == "__main__":
    unittest.main()

