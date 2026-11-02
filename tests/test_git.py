"""Tests for the read-only git bridge.

A throwaway repository is built in a temporary directory with the real ``git``
binary (skipped entirely if git is absent). Timeout and failure paths are
mocked so they never depend on timing.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from omarchy_agent.bridge import git as git_bridge
from omarchy_agent.bridge.git import (
    GitError,
    NotARepository,
    get_current_branch,
    get_diff_summary,
    get_recent_commits,
    get_repository_status,
    git_available,
)
from omarchy_agent.security.audit import AuditLogger, get_audit_logger, set_audit_logger
from omarchy_agent.security.paths import PathPolicy, PathSecurityError

GIT_MISSING = not git_available()

COMMIT_ENVIRONMENT = {
    "GIT_AUTHOR_NAME": "Test Author",
    "GIT_AUTHOR_EMAIL": "author@example.invalid",
    "GIT_COMMITTER_NAME": "Test Author",
    "GIT_COMMITTER_EMAIL": "author@example.invalid",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
}


def run_setup_git(directory: Path, arguments: list[str]) -> str:
    """Run a git command for test *setup* only, never through the bridge."""
    environment = {**os.environ, **COMMIT_ENVIRONMENT}
    completed = subprocess.run(  # noqa: S603
        ["git", *arguments],  # noqa: S607
        cwd=str(directory),
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
        env=environment,
    )
    return completed.stdout


@unittest.skipIf(GIT_MISSING, "git is not installed")
class GitBridgeTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name).resolve()
        self.repository = self.root / "demo-repo"
        self.repository.mkdir()
        self.policy = PathPolicy.from_paths([self.root])

        self.previous_logger = get_audit_logger()
        self.logger = AuditLogger()
        set_audit_logger(self.logger)

        run_setup_git(self.repository, ["init", "-b", "main", "--quiet"])
        (self.repository / "README.md").write_text("# demo\n", encoding="utf-8")
        run_setup_git(self.repository, ["add", "README.md"])
        run_setup_git(self.repository, ["commit", "--quiet", "-m", "initial commit"])
        (self.repository / "second.txt").write_text("one\ntwo\n", encoding="utf-8")
        run_setup_git(self.repository, ["add", "second.txt"])
        run_setup_git(self.repository, ["commit", "--quiet", "-m", "add second file"])

    def tearDown(self) -> None:
        set_audit_logger(self.previous_logger)
        self._temporary.cleanup()


class RepositoryStatusTests(GitBridgeTestCase):
    def test_clean_repository(self) -> None:
        status = get_repository_status(self.repository, policy=self.policy)
        self.assertEqual(status.root, str(self.repository))
        self.assertEqual(status.branch, "main")
        self.assertFalse(status.detached)
        self.assertTrue(status.clean)
        self.assertEqual(status.staged_count, 0)
        self.assertEqual(status.unstaged_count, 0)
        self.assertEqual(status.untracked_count, 0)
        self.assertIsNotNone(status.head_commit)

    def test_detects_untracked_file(self) -> None:
        (self.repository / "scratch.txt").write_text("hi\n", encoding="utf-8")
        status = get_repository_status(self.repository, policy=self.policy)
        self.assertFalse(status.clean)
        self.assertEqual(status.untracked_count, 1)
        self.assertTrue(any(entry.untracked for entry in status.entries))

    def test_detects_unstaged_modification(self) -> None:
        (self.repository / "README.md").write_text("# demo changed\n", encoding="utf-8")
        status = get_repository_status(self.repository, policy=self.policy)
        self.assertEqual(status.unstaged_count, 1)
        self.assertEqual(status.staged_count, 0)

    def test_detects_staged_modification(self) -> None:
        (self.repository / "README.md").write_text("# demo staged\n", encoding="utf-8")
        run_setup_git(self.repository, ["add", "README.md"])
        status = get_repository_status(self.repository, policy=self.policy)
        self.assertEqual(status.staged_count, 1)

    def test_works_from_a_subdirectory(self) -> None:
        nested = self.repository / "src" / "deep"
        nested.mkdir(parents=True)
        (nested / "mod.py").write_text("x = 1\n", encoding="utf-8")
        status = get_repository_status(nested, policy=self.policy)
        self.assertEqual(status.root, str(self.repository))

    def test_works_when_given_a_file_path(self) -> None:
        status = get_repository_status(self.repository / "README.md", policy=self.policy)
        self.assertEqual(status.root, str(self.repository))

    def test_rejects_non_repository(self) -> None:
        plain = self.root / "not-a-repo"
        plain.mkdir()
        with self.assertRaises(NotARepository):
            get_repository_status(plain, policy=self.policy)

    def test_rejects_path_outside_permitted_roots(self) -> None:
        with self.assertRaises(PathSecurityError):
            get_repository_status("/etc", policy=self.policy)

    def test_rejects_missing_path(self) -> None:
        with self.assertRaises(PathSecurityError):
            get_repository_status(self.root / "nope", policy=self.policy)


class BranchTests(GitBridgeTestCase):
    def test_reports_current_branch(self) -> None:
        info = get_current_branch(self.repository, policy=self.policy)
        self.assertEqual(info.branch, "main")
        self.assertFalse(info.detached)
        self.assertIsNotNone(info.head_commit)
        self.assertIsNone(info.upstream)

    def test_reports_a_new_branch(self) -> None:
        run_setup_git(self.repository, ["checkout", "--quiet", "-b", "feature/x"])
        info = get_current_branch(self.repository, policy=self.policy)
        self.assertEqual(info.branch, "feature/x")

    def test_reports_detached_head(self) -> None:
        head = run_setup_git(self.repository, ["rev-parse", "HEAD~1"]).strip()
        run_setup_git(self.repository, ["checkout", "--quiet", head])
        info = get_current_branch(self.repository, policy=self.policy)
        self.assertTrue(info.detached)
        self.assertEqual(info.branch, "HEAD")


class RecentCommitsTests(GitBridgeTestCase):
    def test_returns_commits_newest_first(self) -> None:
        commits = get_recent_commits(self.repository, policy=self.policy)
        self.assertEqual(len(commits), 2)
        self.assertEqual(commits[0].subject, "add second file")
        self.assertEqual(commits[1].subject, "initial commit")
        self.assertEqual(commits[0].author, "Test Author")
        self.assertEqual(len(commits[0].hash), 40)
        self.assertTrue(commits[0].date)

    def test_limit_is_honoured(self) -> None:
        commits = get_recent_commits(self.repository, limit=1, policy=self.policy)
        self.assertEqual(len(commits), 1)

    def test_author_email_is_not_returned(self) -> None:
        commits = get_recent_commits(self.repository, policy=self.policy)
        for commit in commits:
            self.assertNotIn("@", commit.author)
            self.assertNotIn("example.invalid", str(commit.as_dict()))

    def test_empty_repository_returns_no_commits(self) -> None:
        empty = self.root / "empty-repo"
        empty.mkdir()
        run_setup_git(empty, ["init", "-b", "main", "--quiet"])
        self.assertEqual(get_recent_commits(empty, policy=self.policy), [])


class DiffSummaryTests(GitBridgeTestCase):
    def test_empty_diff_on_clean_tree(self) -> None:
        summary = get_diff_summary(self.repository, policy=self.policy)
        self.assertEqual(summary.files_changed, 0)
        self.assertEqual(summary.insertions, 0)
        self.assertEqual(summary.deletions, 0)

    def test_counts_unstaged_changes(self) -> None:
        (self.repository / "second.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
        summary = get_diff_summary(self.repository, policy=self.policy)
        self.assertEqual(summary.files_changed, 1)
        self.assertEqual(summary.insertions, 1)
        self.assertEqual(summary.deletions, 0)
        self.assertEqual(summary.files[0].path, "second.txt")

    def test_counts_staged_changes(self) -> None:
        (self.repository / "second.txt").write_text("replaced\n", encoding="utf-8")
        run_setup_git(self.repository, ["add", "second.txt"])
        unstaged = get_diff_summary(self.repository, policy=self.policy)
        staged = get_diff_summary(self.repository, staged=True, policy=self.policy)
        self.assertEqual(unstaged.files_changed, 0)
        self.assertEqual(staged.files_changed, 1)
        self.assertTrue(staged.staged)

    def test_marks_binary_files(self) -> None:
        (self.repository / "blob.bin").write_bytes(b"\x00\x01\x02\x03")
        run_setup_git(self.repository, ["add", "blob.bin"])
        summary = get_diff_summary(self.repository, staged=True, policy=self.policy)
        binary = [item for item in summary.files if item.binary]
        self.assertTrue(binary)
        self.assertIsNone(binary[0].insertions)

    def test_does_not_return_diff_text(self) -> None:
        (self.repository / "second.txt").write_text("secret-line-content\n", encoding="utf-8")
        summary = get_diff_summary(self.repository, policy=self.policy)
        self.assertNotIn("secret-line-content", str(summary.as_dict()))


class FailureHandlingTests(GitBridgeTestCase):
    def test_timeout_becomes_a_git_error(self) -> None:
        with (
            mock.patch(
                "subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd=["git", "status"], timeout=0.01),
            ),
            self.assertRaises(GitError) as raised,
        ):
            get_repository_status(self.repository, policy=self.policy, timeout=0.01)
        self.assertIn("timed out", str(raised.exception))

    def test_oserror_becomes_a_git_error(self) -> None:
        with (
            mock.patch("subprocess.run", side_effect=OSError("no exec")),
            self.assertRaises(GitError),
        ):
            get_repository_status(self.repository, policy=self.policy)

    def test_nonzero_exit_becomes_a_git_error(self) -> None:
        # The first call resolves the repository root successfully; the second
        # (``git status``) fails, which is the path under test.
        resolved = subprocess.CompletedProcess(
            args=["git"], returncode=0, stdout=f"{self.repository}\n", stderr=""
        )
        failed = subprocess.CompletedProcess(
            args=["git"], returncode=128, stdout="", stderr="fatal: bad revision\n"
        )
        with (
            mock.patch("subprocess.run", side_effect=[resolved, failed]),
            self.assertRaises(GitError) as raised,
        ):
            get_repository_status(self.repository, policy=self.policy)
        self.assertIn("bad revision", str(raised.exception))

    def test_unresolvable_repository_is_reported(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["git"], returncode=128, stdout="", stderr="fatal: not a git repository\n"
        )
        with (
            mock.patch("subprocess.run", return_value=completed),
            self.assertRaises(NotARepository),
        ):
            get_repository_status(self.repository, policy=self.policy)

    def test_missing_git_binary_is_reported(self) -> None:
        with (
            mock.patch.object(git_bridge.shutil, "which", return_value=None),
            self.assertRaises(git_bridge.GitNotAvailable),
        ):
            get_repository_status(self.repository, policy=self.policy)

    def test_timeout_is_audited_as_an_error(self) -> None:
        with (
            mock.patch(
                "subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd=["git", "status"], timeout=0.01),
            ),
            self.assertRaises(GitError),
        ):
            get_repository_status(self.repository, policy=self.policy, timeout=0.01)
        entry = self.logger.entries()[-1]
        self.assertEqual(entry.status, "error")
        self.assertIn("GitError", entry.result_summary)


class HardeningTests(GitBridgeTestCase):
    def test_repository_hooks_are_disabled(self) -> None:
        with mock.patch("subprocess.run", wraps=subprocess.run) as spy:
            get_current_branch(self.repository, policy=self.policy)
        argv = spy.call_args_list[0][0][0]
        self.assertIn("core.hooksPath=", argv)
        self.assertIn("core.fsmonitor=false", argv)
        self.assertIn("--no-optional-locks", argv)

    def test_no_shell_is_used(self) -> None:
        with mock.patch("subprocess.run", wraps=subprocess.run) as spy:
            get_current_branch(self.repository, policy=self.policy)
        for call in spy.call_args_list:
            self.assertNotIn("shell", call.kwargs)
            self.assertIsInstance(call.args[0], list)

    def test_environment_blocks_credential_prompts(self) -> None:
        with mock.patch("subprocess.run", wraps=subprocess.run) as spy:
            get_current_branch(self.repository, policy=self.policy)
        environment = spy.call_args_list[0].kwargs["env"]
        self.assertEqual(environment["GIT_TERMINAL_PROMPT"], "0")
        self.assertEqual(environment["GIT_CONFIG_NOSYSTEM"], "1")
        self.assertEqual(environment["GIT_ASKPASS"], "")
        self.assertNotIn("SSH_AUTH_SOCK", environment)

    def test_a_hook_is_not_executed(self) -> None:
        hooks = self.repository / ".git" / "hooks"
        hooks.mkdir(exist_ok=True)
        marker = self.root / "hook-ran.marker"
        hook = hooks / "post-index-change"
        hook.write_text(f"#!/bin/sh\ntouch {marker}\n", encoding="utf-8")
        hook.chmod(0o755)
        get_repository_status(self.repository, policy=self.policy)
        self.assertFalse(marker.exists(), "a repository hook was executed")


class GitAuditTests(GitBridgeTestCase):
    def test_calls_are_audited(self) -> None:
        get_repository_status(self.repository, policy=self.policy)
        get_current_branch(self.repository, policy=self.policy)
        get_recent_commits(self.repository, policy=self.policy)
        get_diff_summary(self.repository, policy=self.policy)
        names = [entry.tool_name for entry in self.logger.entries()]
        self.assertEqual(
            names,
            [
                "git.get_repository_status",
                "git.get_current_branch",
                "git.get_recent_commits",
                "git.get_diff_summary",
            ],
        )


if __name__ == "__main__":
    unittest.main()
