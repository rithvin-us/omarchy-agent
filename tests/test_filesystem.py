"""Tests for the read-only filesystem bridge."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from omarchy_agent.bridge.filesystem import (
    BinaryFileRejected,
    FilesystemError,
    FileTooLarge,
    get_file_info,
    list_directory,
    read_file,
    search_files,
)
from omarchy_agent.security.audit import AuditLogger, get_audit_logger, set_audit_logger
from omarchy_agent.security.paths import PathPolicy, PathSecurityError


class FilesystemBridgeTestCase(unittest.TestCase):
    """Builds a realistic throwaway project tree under a temporary root."""

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self._outside = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name).resolve()
        self.outside = Path(self._outside.name).resolve()
        self.policy = PathPolicy.from_paths([self.root])

        self.previous_logger = get_audit_logger()
        self.logger = AuditLogger()
        set_audit_logger(self.logger)

        # a normal project
        (self.root / "src").mkdir()
        (self.root / "src" / "main.py").write_text(
            "def main():\n    print('omarchy')\n", encoding="utf-8"
        )
        (self.root / "README.md").write_text("# demo project\n\nhello\n", encoding="utf-8")
        (self.root / ".env.example").write_text("API_URL=http://127.0.0.1\n", encoding="utf-8")

        # things the bridge must refuse or skip
        (self.root / ".env").write_text("API_KEY=super-secret-value\n", encoding="utf-8")
        (self.root / "id_rsa").write_text("-----BEGIN RSA PRIVATE KEY-----\n", encoding="utf-8")
        (self.root / ".git").mkdir()
        (self.root / ".git" / "config").write_text("[core]\n", encoding="utf-8")
        (self.root / "node_modules").mkdir()
        (self.root / "node_modules" / "left-pad.js").write_text("// pad\n", encoding="utf-8")
        (self.root / "__pycache__").mkdir()

        # a binary file
        (self.root / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x01")

        # an oversized text file (just over the 128 KiB policy limit)
        (self.root / "huge.log").write_text("x" * (129 * 1024), encoding="utf-8")

        # a file outside the root, plus symlinks pointing at it
        self.outside_file = self.outside / "outside-secret.txt"
        self.outside_file.write_text("data that must never be read\n", encoding="utf-8")
        (self.root / "escape-file").symlink_to(self.outside_file)
        (self.root / "escape-dir").symlink_to(self.outside, target_is_directory=True)

    def tearDown(self) -> None:
        set_audit_logger(self.previous_logger)
        self._temporary.cleanup()
        self._outside.cleanup()


class ListDirectoryTests(FilesystemBridgeTestCase):
    def test_lists_normal_project_entries(self) -> None:
        listing = list_directory(self.root, policy=self.policy)
        names = {entry.name for entry in listing.entries}
        self.assertIn("src", names)
        self.assertIn("README.md", names)
        self.assertEqual(listing.path, str(self.root))

    def test_blocked_directories_are_marked_not_readable(self) -> None:
        listing = list_directory(self.root, policy=self.policy)
        by_name = {entry.name: entry for entry in listing.entries}
        for name in (".git", "node_modules", "__pycache__"):
            with self.subTest(name=name):
                self.assertFalse(by_name[name].readable)
                self.assertEqual(by_name[name].skip_reason, "blocked directory")

    def test_sensitive_files_are_marked_not_readable(self) -> None:
        listing = list_directory(self.root, policy=self.policy)
        by_name = {entry.name: entry for entry in listing.entries}
        self.assertEqual(by_name[".env"].skip_reason, "may contain secrets")
        self.assertEqual(by_name["id_rsa"].skip_reason, "may contain secrets")
        self.assertIsNone(by_name[".env.example"].skip_reason)

    def test_escaping_symlinks_are_marked_not_readable(self) -> None:
        listing = list_directory(self.root, policy=self.policy)
        by_name = {entry.name: entry for entry in listing.entries}
        for name in ("escape-file", "escape-dir"):
            with self.subTest(name=name):
                self.assertEqual(by_name[name].kind, "symlink")
                self.assertEqual(
                    by_name[name].skip_reason, "symlink points outside the workspace"
                )

    def test_file_sizes_and_timestamps_are_reported(self) -> None:
        listing = list_directory(self.root / "src", policy=self.policy)
        entry = next(item for item in listing.entries if item.name == "main.py")
        self.assertGreater(entry.size_bytes or 0, 0)
        self.assertIsNotNone(entry.modified_at)

    def test_rejects_directory_outside_the_root(self) -> None:
        with self.assertRaises(PathSecurityError):
            list_directory(self.outside, policy=self.policy)

    def test_rejects_a_file_argument(self) -> None:
        with self.assertRaises(FilesystemError):
            list_directory(self.root / "README.md", policy=self.policy)

    def test_limit_truncates_the_listing(self) -> None:
        listing = list_directory(self.root, policy=self.policy, limit=2)
        self.assertEqual(len(listing.entries), 2)
        self.assertTrue(listing.truncated)


class ReadFileTests(FilesystemBridgeTestCase):
    def test_reads_a_normal_text_file(self) -> None:
        result = read_file(self.root / "src" / "main.py", policy=self.policy)
        self.assertIn("omarchy", result.content)
        self.assertEqual(result.encoding, "utf-8")
        self.assertEqual(result.line_count, 2)
        self.assertFalse(result.truncated)

    def test_reads_an_example_env_file(self) -> None:
        result = read_file(self.root / ".env.example", policy=self.policy)
        self.assertIn("API_URL", result.content)

    def test_refuses_dotdot_traversal(self) -> None:
        with self.assertRaises(PathSecurityError):
            read_file(f"{self.root}/src/../../etc/passwd", policy=self.policy)

    def test_refuses_absolute_path_outside_the_root(self) -> None:
        for target in ("/etc/passwd", "/etc/shadow", str(self.outside_file)):
            with self.subTest(target=target):
                with self.assertRaises(PathSecurityError):
                    read_file(target, policy=self.policy)

    def test_refuses_symlink_escape(self) -> None:
        with self.assertRaises(PathSecurityError):
            read_file(self.root / "escape-file", policy=self.policy)

    def test_refuses_file_through_symlinked_directory(self) -> None:
        with self.assertRaises(PathSecurityError):
            read_file(self.root / "escape-dir" / "outside-secret.txt", policy=self.policy)

    def test_refuses_sensitive_files(self) -> None:
        for name in (".env", "id_rsa"):
            with self.subTest(name=name):
                with self.assertRaises(PathSecurityError):
                    read_file(self.root / name, policy=self.policy)

    def test_refuses_file_inside_blocked_directory(self) -> None:
        with self.assertRaises(PathSecurityError):
            read_file(self.root / ".git" / "config", policy=self.policy)
        with self.assertRaises(PathSecurityError):
            read_file(self.root / "node_modules" / "left-pad.js", policy=self.policy)

    def test_rejects_oversized_file(self) -> None:
        with self.assertRaises(FileTooLarge) as raised:
            read_file(self.root / "huge.log", policy=self.policy)
        self.assertEqual(raised.exception.limit, 128 * 1024)
        self.assertGreater(raised.exception.size, 128 * 1024)

    def test_rejects_binary_file(self) -> None:
        with self.assertRaises(BinaryFileRejected):
            read_file(self.root / "logo.png", policy=self.policy)

    def test_rejects_directory(self) -> None:
        with self.assertRaises(FilesystemError):
            read_file(self.root / "src", policy=self.policy)

    def test_rejects_missing_file(self) -> None:
        with self.assertRaises(PathSecurityError):
            read_file(self.root / "does-not-exist.txt", policy=self.policy)

    def test_custom_byte_limit_is_honoured(self) -> None:
        with self.assertRaises(FileTooLarge):
            read_file(self.root / "README.md", policy=self.policy, max_bytes=3)

    def test_file_content_is_never_written_to_the_audit_log(self) -> None:
        secret_marker = "UNIQUE-MARKER-9f3a2c"
        target = self.root / "notes.txt"
        target.write_text(f"some notes {secret_marker}\n", encoding="utf-8")
        result = read_file(target, policy=self.policy)
        self.assertIn(secret_marker, result.content)
        for entry in self.logger.entries():
            self.assertNotIn(secret_marker, entry.as_line())

    def test_audit_entry_describes_the_read(self) -> None:
        read_file(self.root / "README.md", policy=self.policy)
        entry = self.logger.entries()[-1]
        self.assertEqual(entry.tool_name, "filesystem.read_file")
        self.assertEqual(entry.status, "success")
        self.assertIn("bytes", entry.result_summary)

    def test_refusal_is_audited_as_an_error(self) -> None:
        with self.assertRaises(PathSecurityError):
            read_file("/etc/passwd", policy=self.policy)
        entry = self.logger.entries()[-1]
        self.assertEqual(entry.status, "error")
        self.assertIn("PathSecurityError", entry.result_summary)


class FileInfoTests(FilesystemBridgeTestCase):
    def test_reports_metadata_for_a_normal_file(self) -> None:
        info = get_file_info(self.root / "README.md", policy=self.policy)
        self.assertEqual(info.kind, "file")
        self.assertEqual(info.name, "README.md")
        self.assertGreater(info.size_bytes, 0)
        self.assertTrue(info.readable)
        self.assertFalse(info.sensitive)
        self.assertTrue(info.within_workspace)

    def test_reports_metadata_for_a_directory(self) -> None:
        info = get_file_info(self.root / "src", policy=self.policy)
        self.assertEqual(info.kind, "directory")

    def test_flags_sensitive_files_without_reading_them(self) -> None:
        info = get_file_info(self.root / ".env", policy=self.policy)
        self.assertTrue(info.sensitive)
        self.assertFalse(info.readable)

    def test_flags_blocked_directories(self) -> None:
        info = get_file_info(self.root / "node_modules", policy=self.policy)
        self.assertTrue(info.sensitive)

    def test_reports_symlink_target(self) -> None:
        link = self.root / "inside-link"
        link.symlink_to(self.root / "README.md")
        info = get_file_info(link, policy=self.policy)
        self.assertTrue(info.is_symlink)
        self.assertEqual(info.symlink_target, str(self.root / "README.md"))

    def test_rejects_path_outside_the_root(self) -> None:
        with self.assertRaises(PathSecurityError):
            get_file_info("/etc/passwd", policy=self.policy)

    def test_rejects_escaping_symlink(self) -> None:
        with self.assertRaises(PathSecurityError):
            get_file_info(self.root / "escape-file", policy=self.policy)

    def test_rejects_oversized_file_as_not_readable(self) -> None:
        info = get_file_info(self.root / "huge.log", policy=self.policy)
        self.assertFalse(info.readable)


class SearchFilesTests(FilesystemBridgeTestCase):
    def test_finds_content_matches(self) -> None:
        results = search_files("omarchy", self.root, policy=self.policy)
        content_matches = [match for match in results.matches if match.match_type == "content"]
        self.assertTrue(content_matches)
        self.assertTrue(any("main.py" in match.path for match in content_matches))
        self.assertTrue(all(match.line_number for match in content_matches))

    def test_finds_name_matches(self) -> None:
        results = search_files("README", self.root, policy=self.policy)
        self.assertTrue(any(match.match_type == "name" for match in results.matches))

    def test_search_is_case_insensitive(self) -> None:
        self.assertTrue(search_files("OMARCHY", self.root, policy=self.policy).matches)

    def test_query_is_literal_not_a_regex(self) -> None:
        (self.root / "regex.txt").write_text("literal a+b here\n", encoding="utf-8")
        results = search_files("a+b", self.root, policy=self.policy)
        self.assertTrue(any("regex.txt" in match.path for match in results.matches))
        self.assertFalse(search_files(".*", self.root, policy=self.policy).matches)

    def test_does_not_search_blocked_directories(self) -> None:
        results = search_files("pad", self.root, policy=self.policy)
        self.assertFalse(any("node_modules" in match.path for match in results.matches))
        self.assertGreater(results.directories_skipped, 0)

    def test_does_not_search_sensitive_files(self) -> None:
        results = search_files("super-secret-value", self.root, policy=self.policy)
        self.assertEqual(results.matches, [])
        self.assertGreater(results.files_skipped, 0)

    def test_does_not_follow_symlinks_out_of_the_root(self) -> None:
        results = search_files("must never be read", self.root, policy=self.policy)
        self.assertEqual(results.matches, [])

    def test_skips_binary_files(self) -> None:
        results = search_files("IHDR", self.root, policy=self.policy)
        self.assertFalse(
            any(match.match_type == "content" and "logo.png" in match.path for match in results.matches)
        )

    def test_respects_max_results(self) -> None:
        bulk = self.root / "bulk"
        bulk.mkdir()
        for index in range(40):
            (bulk / f"file{index}.txt").write_text("needle\n", encoding="utf-8")
        results = search_files("needle", bulk, policy=self.policy, max_results=5)
        self.assertEqual(len(results.matches), 5)
        self.assertTrue(results.truncated)

    def test_timeout_is_reported_not_raised(self) -> None:
        bulk = self.root / "slow"
        bulk.mkdir()
        for index in range(60):
            (bulk / f"file{index}.txt").write_text("content\n" * 100, encoding="utf-8")
        results = search_files(
            "content", bulk, policy=self.policy, timeout_seconds=0.0, max_results=10_000
        )
        self.assertIsInstance(results.timed_out, bool)
        self.assertGreaterEqual(results.duration_ms, 0)

    def test_rejects_root_outside_the_policy(self) -> None:
        with self.assertRaises(PathSecurityError):
            search_files("anything", self.outside, policy=self.policy)

    def test_rejects_empty_query(self) -> None:
        with self.assertRaises(FilesystemError):
            search_files("   ", self.root, policy=self.policy)

    def test_credential_shaped_match_lines_are_redacted(self) -> None:
        (self.root / "leak.txt").write_text(
            "token sk-abcdefghijklmnopqrstuvwxyz123456 appears here\n", encoding="utf-8"
        )
        results = search_files("appears here", self.root, policy=self.policy)
        lines = [match.line for match in results.matches if match.line]
        self.assertTrue(lines)
        for line in lines:
            self.assertNotIn("sk-abcdefghijklmnopqrstuvwxyz123456", line)


if __name__ == "__main__":
    unittest.main()
