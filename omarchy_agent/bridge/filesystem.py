"""Read-only filesystem bridge.

Every public function here is a *tool*: it takes plain arguments, returns a
dataclass that knows how to describe itself for the audit log, and is wrapped
by :func:`omarchy_agent.security.audit.audited`. That shape is what a later
milestone will expose over MCP without changing this module.

v0.1 is strictly read-only. There is no write, move, delete or chmod path in
this file, and the ``filesystem.write`` capability is marked
``NOT_IMPLEMENTED`` in the permission policy.
"""

from __future__ import annotations

import os
import stat
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..security.audit import audited, redact_text
from ..security.paths import PathPolicy, PathSecurityError, default_policy
from ..security.permissions import Capability

__all__ = [
    "BinaryFileRejected",
    "DirectoryEntry",
    "DirectoryListing",
    "FileContent",
    "FileInfo",
    "FileTooLarge",
    "FilesystemError",
    "SearchMatch",
    "SearchResults",
    "get_file_info",
    "list_directory",
    "read_file",
    "search_files",
]

#: Bytes sampled when deciding whether a file is binary.
BINARY_SNIFF_BYTES = 8192

#: Hard ceiling on entries returned by one ``list_directory`` call.
DEFAULT_LISTING_LIMIT = 1000

#: Defaults for ``search_files``.
DEFAULT_SEARCH_MAX_RESULTS = 100
DEFAULT_SEARCH_TIMEOUT_SECONDS = 5.0
DEFAULT_SEARCH_MAX_FILES = 20_000
MAX_SEARCH_LINE_CHARS = 200


class FilesystemError(RuntimeError):
    """Base class for filesystem bridge refusals that are not policy breaches."""


class FileTooLarge(FilesystemError):
    def __init__(self, path: Path, size: int, limit: int) -> None:
        super().__init__(
            f"file is {size} bytes which exceeds the {limit} byte read limit: {path}"
        )
        self.path = str(path)
        self.size = size
        self.limit = limit


class BinaryFileRejected(FilesystemError):
    def __init__(self, path: Path, reason: str) -> None:
        super().__init__(f"refusing to read binary file ({reason}): {path}")
        self.path = str(path)
        self.reason = reason


def _resolve_policy(policy: PathPolicy | None) -> PathPolicy:
    return policy if policy is not None else default_policy()


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, UTC).isoformat(timespec="seconds")


def _entry_kind(path: Path, *, follow: bool = False) -> str:
    try:
        info = path.stat() if follow else path.lstat()
    except OSError:
        return "unknown"
    mode = info.st_mode
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISFIFO(mode):
        return "fifo"
    if stat.S_ISSOCK(mode):
        return "socket"
    if stat.S_ISBLK(mode) or stat.S_ISCHR(mode):
        return "device"
    return "other"


# --------------------------------------------------------------------------
# result types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectoryEntry:
    name: str
    path: str
    kind: str
    size_bytes: int | None
    modified_at: str | None
    readable: bool
    skip_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "kind": self.kind,
            "size_bytes": self.size_bytes,
            "modified_at": self.modified_at,
            "readable": self.readable,
            "skip_reason": self.skip_reason,
        }


@dataclass(frozen=True)
class DirectoryListing:
    path: str
    entries: list[DirectoryEntry]
    total_entries: int
    truncated: bool
    skipped: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "entries": [entry.as_dict() for entry in self.entries],
            "total_entries": self.total_entries,
            "truncated": self.truncated,
            "skipped": self.skipped,
        }

    def audit_summary(self) -> str:
        return (
            f"listing of {self.path}: {len(self.entries)} entries "
            f"({self.skipped} skipped, truncated={self.truncated})"
        )


@dataclass(frozen=True)
class FileContent:
    path: str
    size_bytes: int
    encoding: str
    line_count: int
    truncated: bool
    content: str = field(repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "encoding": self.encoding,
            "line_count": self.line_count,
            "truncated": self.truncated,
            "content": self.content,
        }

    def audit_summary(self) -> str:
        """Describe the read without ever revealing the file's content."""
        return (
            f"read {self.path}: {self.size_bytes} bytes, {self.line_count} lines, "
            f"encoding={self.encoding}, truncated={self.truncated}"
        )


@dataclass(frozen=True)
class FileInfo:
    path: str
    name: str
    kind: str
    size_bytes: int
    permissions: str
    owner_uid: int
    group_gid: int
    modified_at: str
    created_at: str
    accessed_at: str
    is_symlink: bool
    symlink_target: str | None
    readable: bool
    within_workspace: bool
    sensitive: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "name": self.name,
            "kind": self.kind,
            "size_bytes": self.size_bytes,
            "permissions": self.permissions,
            "owner_uid": self.owner_uid,
            "group_gid": self.group_gid,
            "modified_at": self.modified_at,
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "is_symlink": self.is_symlink,
            "symlink_target": self.symlink_target,
            "readable": self.readable,
            "within_workspace": self.within_workspace,
            "sensitive": self.sensitive,
        }

    def audit_summary(self) -> str:
        return f"info for {self.path}: kind={self.kind} size={self.size_bytes}"


@dataclass(frozen=True)
class SearchMatch:
    path: str
    match_type: str  # "name" or "content"
    line_number: int | None
    line: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "match_type": self.match_type,
            "line_number": self.line_number,
            "line": self.line,
        }


@dataclass(frozen=True)
class SearchResults:
    query: str
    root: str
    matches: list[SearchMatch]
    files_scanned: int
    directories_skipped: int
    files_skipped: int
    truncated: bool
    timed_out: bool
    duration_ms: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "root": self.root,
            "matches": [match.as_dict() for match in self.matches],
            "files_scanned": self.files_scanned,
            "directories_skipped": self.directories_skipped,
            "files_skipped": self.files_skipped,
            "truncated": self.truncated,
            "timed_out": self.timed_out,
            "duration_ms": self.duration_ms,
        }

    def audit_summary(self) -> str:
        return (
            f"search in {self.root}: {len(self.matches)} matches across "
            f"{self.files_scanned} files (truncated={self.truncated}, "
            f"timed_out={self.timed_out})"
        )


# --------------------------------------------------------------------------
# binary / text classification
# --------------------------------------------------------------------------


def _classify_bytes(data: bytes) -> tuple[bool, str]:
    """Return ``(is_text, detail)`` for a byte sample."""
    if not data:
        return True, "empty"
    if b"\x00" in data[:BINARY_SNIFF_BYTES]:
        return False, "NUL byte in header"
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        # A truncated multi-byte sequence at the sample boundary is not proof
        # of a binary file; retry ignoring the tail.
        try:
            data[: max(0, len(data) - 4)].decode("utf-8")
        except UnicodeDecodeError:
            return False, "not valid UTF-8"
    control = sum(
        1
        for byte in data[:BINARY_SNIFF_BYTES]
        if byte < 0x09 or (0x0E <= byte < 0x20) or byte == 0x7F
    )
    if control and control / min(len(data), BINARY_SNIFF_BYTES) > 0.10:
        return False, "high proportion of control bytes"
    return True, "utf-8"


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------


@audited("filesystem.list_directory", Capability.FS_LIST)
def list_directory(
    path: str | os.PathLike[str],
    *,
    policy: PathPolicy | None = None,
    limit: int = DEFAULT_LISTING_LIMIT,
    include_hidden: bool = True,
) -> DirectoryListing:
    """List the immediate children of ``path``.

    Blocked directories and sensitive files are reported with a
    ``skip_reason`` rather than silently omitted, so the caller can see that
    something was withheld. Symlinks are never followed out of the workspace.
    """
    active = _resolve_policy(policy)
    target = active.resolve_within_roots(path)
    if not target.is_dir():
        raise FilesystemError(f"not a directory: {target}")

    entries: list[DirectoryEntry] = []
    skipped = 0
    truncated = False

    with os.scandir(target) as scanner:
        raw_entries = sorted(
            scanner,
            key=lambda entry: (not entry.is_dir(follow_symlinks=False), entry.name.lower()),
        )
    total = len(raw_entries)

    for position, entry in enumerate(raw_entries):
        if len(entries) >= limit:
            truncated = position < total
            break
        if not include_hidden and entry.name.startswith("."):
            skipped += 1
            continue

        child = Path(entry.path)
        kind = _entry_kind(child)
        skip_reason: str | None = None

        if kind == "directory" and PathPolicy.is_blocked_directory_name(entry.name):
            skip_reason = "blocked directory"
        elif kind == "symlink":
            if not active.is_within_roots(child):
                skip_reason = "symlink points outside the workspace"
        elif PathPolicy.is_sensitive_file_name(entry.name):
            skip_reason = "may contain secrets"

        size: int | None = None
        modified: str | None = None
        try:
            info = entry.stat(follow_symlinks=False)
            size = info.st_size if kind == "file" else None
            modified = _iso(info.st_mtime)
        except OSError:
            pass

        if skip_reason is not None:
            skipped += 1
            entries.append(
                DirectoryEntry(
                    name=entry.name,
                    path=str(child),
                    kind=kind,
                    size_bytes=None,
                    modified_at=modified,
                    readable=False,
                    skip_reason=skip_reason,
                )
            )
        else:
            entries.append(
                DirectoryEntry(
                    name=entry.name,
                    path=str(child),
                    kind=kind,
                    size_bytes=size,
                    modified_at=modified,
                    readable=kind in {"file", "directory"},
                    skip_reason=None,
                )
            )

    return DirectoryListing(
        path=str(target),
        entries=entries,
        total_entries=total,
        truncated=truncated,
        skipped=skipped,
    )


@audited("filesystem.read_file", Capability.FS_READ)
def read_file(
    path: str | os.PathLike[str],
    *,
    policy: PathPolicy | None = None,
    max_bytes: int | None = None,
) -> FileContent:
    """Read a UTF-8 text file from the permitted workspace.

    Raises
    ------
    PathSecurityError
        Outside the roots, inside a blocked directory, or a sensitive file.
    FileTooLarge
        Larger than the policy's ``max_file_bytes`` (128 KiB by default).
    BinaryFileRejected
        The file is not decodable text.
    """
    active = _resolve_policy(policy)
    target = active.resolve_within_roots(path)
    limit = active.max_file_bytes if max_bytes is None else int(max_bytes)

    info = target.stat()
    if not stat.S_ISREG(info.st_mode):
        raise FilesystemError(
            f"not a regular file (kind={_entry_kind(target, follow=True)}): {target}"
        )
    if info.st_size > limit:
        raise FileTooLarge(target, info.st_size, limit)

    with open(target, "rb") as handle:
        data = handle.read(limit + 1)

    truncated = len(data) > limit
    if truncated:
        data = data[:limit]

    is_text, detail = _classify_bytes(data)
    if not is_text:
        raise BinaryFileRejected(target, detail)

    text = data.decode("utf-8", errors="replace")
    return FileContent(
        path=str(target),
        size_bytes=info.st_size,
        encoding="utf-8",
        line_count=text.count("\n") + (1 if text and not text.endswith("\n") else 0),
        truncated=truncated,
        content=text,
    )


@audited("filesystem.get_file_info", Capability.FS_READ)
def get_file_info(
    path: str | os.PathLike[str],
    *,
    policy: PathPolicy | None = None,
) -> FileInfo:
    """Return metadata for a path. Metadata only -- the file is never opened.

    Unlike :func:`read_file` this tolerates sensitive file names: knowing that
    ``.env`` exists and how big it is leaks nothing, and the result flags it as
    ``sensitive`` so callers do not try to read it.
    """
    active = _resolve_policy(policy)
    resolved = active._resolve(path)
    if not active.is_within_roots(resolved):
        raise PathSecurityError(
            "path is outside the permitted workspace roots "
            f"({', '.join(active.describe_roots())})",
            resolved,
        )
    if not resolved.exists() and not resolved.is_symlink():
        raise PathSecurityError("path does not exist", resolved)

    raw_path = Path(path).expanduser()
    is_symlink = raw_path.is_symlink()
    link_target: str | None = None
    if is_symlink:
        try:
            link_target = str(raw_path.readlink())
        except OSError:
            link_target = None

    info = resolved.lstat()
    sensitive = PathPolicy.is_sensitive_file_name(resolved.name)
    kind = _entry_kind(resolved)
    blocked_dir = kind == "directory" and PathPolicy.is_blocked_directory_name(resolved.name)

    return FileInfo(
        path=str(resolved),
        name=resolved.name,
        kind=kind,
        size_bytes=int(info.st_size),
        permissions=oct(stat.S_IMODE(info.st_mode)),
        owner_uid=int(info.st_uid),
        group_gid=int(info.st_gid),
        modified_at=_iso(info.st_mtime),
        created_at=_iso(info.st_ctime),
        accessed_at=_iso(info.st_atime),
        is_symlink=is_symlink,
        symlink_target=link_target,
        readable=kind == "file" and not sensitive and info.st_size <= active.max_file_bytes,
        within_workspace=True,
        sensitive=sensitive or blocked_dir,
    )


@audited("filesystem.search_files", Capability.FS_SEARCH)
def search_files(
    query: str,
    path: str | os.PathLike[str],
    *,
    policy: PathPolicy | None = None,
    max_results: int = DEFAULT_SEARCH_MAX_RESULTS,
    timeout_seconds: float = DEFAULT_SEARCH_TIMEOUT_SECONDS,
    max_files: int = DEFAULT_SEARCH_MAX_FILES,
    search_content: bool = True,
) -> SearchResults:
    """Search file names and text content under ``path`` for ``query``.

    ``query`` is treated as a literal, case-insensitive substring -- never as a
    regex or a shell pattern, so a hostile query cannot widen the search.
    The walk is bounded three ways: ``max_results``, ``timeout_seconds`` and
    ``max_files``. Blocked directories are pruned and sensitive files are never
    opened.
    """
    active = _resolve_policy(policy)
    root = active.resolve_within_roots(path)
    if not root.is_dir():
        raise FilesystemError(f"not a directory: {root}")

    needle = str(query)
    if not needle.strip():
        raise FilesystemError("search query must not be empty")
    lowered = needle.lower()

    started = time.monotonic()
    deadline = started + max(0.1, float(timeout_seconds))

    matches: list[SearchMatch] = []
    files_scanned = 0
    directories_skipped = 0
    files_skipped = 0
    truncated = False
    timed_out = False

    for current_root, directory_names, file_names in os.walk(root, followlinks=False):
        if time.monotonic() > deadline:
            timed_out = True
            break

        kept: list[str] = []
        for name in directory_names:
            if PathPolicy.is_blocked_directory_name(name):
                directories_skipped += 1
                continue
            candidate = Path(current_root) / name
            if candidate.is_symlink() and not active.is_within_roots(candidate):
                directories_skipped += 1
                continue
            kept.append(name)
        directory_names[:] = sorted(kept)

        for name in sorted(file_names):
            if time.monotonic() > deadline:
                timed_out = True
                break
            if files_scanned >= max_files:
                truncated = True
                break

            candidate = Path(current_root) / name

            if PathPolicy.is_sensitive_file_name(name):
                files_skipped += 1
                continue
            if candidate.is_symlink() and not active.is_within_roots(candidate):
                files_skipped += 1
                continue

            files_scanned += 1

            if lowered in name.lower():
                matches.append(
                    SearchMatch(
                        path=str(candidate), match_type="name", line_number=None, line=None
                    )
                )
                if len(matches) >= max_results:
                    truncated = True
                    break

            if not search_content:
                continue

            try:
                if candidate.stat().st_size > active.max_file_bytes:
                    files_skipped += 1
                    continue
                with open(candidate, "rb") as handle:
                    data = handle.read(active.max_file_bytes)
            except OSError:
                files_skipped += 1
                continue

            is_text, _detail = _classify_bytes(data)
            if not is_text:
                files_skipped += 1
                continue

            text = data.decode("utf-8", errors="replace")
            if lowered not in text.lower():
                continue

            for number, line in enumerate(text.splitlines(), start=1):
                if lowered not in line.lower():
                    continue
                snippet = redact_text(line.strip())[:MAX_SEARCH_LINE_CHARS]
                matches.append(
                    SearchMatch(
                        path=str(candidate),
                        match_type="content",
                        line_number=number,
                        line=snippet,
                    )
                )
                if len(matches) >= max_results:
                    truncated = True
                    break

            if len(matches) >= max_results:
                break

        if truncated or timed_out or len(matches) >= max_results:
            break

    return SearchResults(
        query=needle,
        root=str(root),
        matches=matches,
        files_scanned=files_scanned,
        directories_skipped=directories_skipped,
        files_skipped=files_skipped,
        truncated=truncated,
        timed_out=timed_out,
        duration_ms=int((time.monotonic() - started) * 1000),
    )
