"""Read-only git inspection bridge.

Only a fixed set of subcommands is ever run: ``rev-parse``, ``status``,
``log``, ``diff`` and ``config --get``. Every invocation is an argument array
with a timeout; no part of a command is built from caller-supplied shell text,
and ``shell=False`` is never overridden.

Repository-controlled code is kept out of the way:

* ``core.hooksPath`` is pointed at an empty value and ``core.fsmonitor`` is
  disabled, so no repository hook or monitor process runs;
* ``--no-optional-locks`` keeps ``status`` from writing index extensions;
* ``GIT_CONFIG_NOSYSTEM``, ``GIT_TERMINAL_PROMPT=0`` and an empty
  ``GIT_ASKPASS`` prevent system config and credential prompts;
* ``core.pager`` is disabled so nothing is piped to an external program.

No build, no checkout, no fetch, no write.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..security.audit import audited
from ..security.paths import PathPolicy, default_policy
from ..security.permissions import Capability

__all__ = [
    "BranchInfo",
    "CommitInfo",
    "DiffFileStat",
    "DiffSummary",
    "GitError",
    "GitNotAvailable",
    "NotARepository",
    "RepositoryStatus",
    "StatusEntry",
    "get_current_branch",
    "get_diff_summary",
    "get_recent_commits",
    "get_repository_status",
    "git_available",
]

GIT_TIMEOUT_SECONDS = 10.0
MAX_GIT_OUTPUT_CHARS = 512 * 1024
MAX_STATUS_ENTRIES = 500
MAX_COMMITS = 200

#: Field separator used in ``git log`` format strings (ASCII unit separator).
_FIELD_SEPARATOR = "\x1f"

#: Flags prepended to every git invocation to neutralise repository-controlled
#: hooks, monitors and pagers.
_HARDENING_FLAGS: tuple[str, ...] = (
    "-c",
    "core.hooksPath=",
    "-c",
    "core.fsmonitor=false",
    "-c",
    "core.pager=cat",
    "-c",
    "advice.detachedHead=false",
    "--no-optional-locks",
)


class GitError(RuntimeError):
    """Raised when a git invocation fails in a way the caller should see."""


class GitNotAvailable(GitError):
    """Raised when the ``git`` executable is not on PATH."""


class NotARepository(GitError):
    def __init__(self, path: Path) -> None:
        super().__init__(f"not inside a git work tree: {path}")
        self.path = str(path)


def git_available() -> bool:
    return shutil.which("git") is not None


def _git_environment() -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", ""),
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PAGER": "cat",
        "GIT_FLUSH": "1",
    }


def _run_git(
    directory: Path,
    arguments: list[str],
    *,
    timeout: float = GIT_TIMEOUT_SECONDS,
    allow_failure: bool = False,
) -> str:
    """Run one hardened git command inside ``directory`` and return stdout."""
    executable = shutil.which("git")
    if executable is None:
        raise GitNotAvailable("git is not installed on this system")
    if any(not isinstance(part, str) for part in arguments):
        raise GitError("every git argument must be a string")

    argv = [executable, *_HARDENING_FLAGS, *arguments]
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, shell=False
            argv,
            cwd=str(directory),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=_git_environment(),
        )
    except subprocess.TimeoutExpired as exc:
        raise GitError(
            f"git {' '.join(arguments[:2])} timed out after {timeout}s in {directory}"
        ) from exc
    except OSError as exc:
        raise GitError(f"could not run git: {type(exc).__name__}: {exc}") from exc

    if completed.returncode != 0 and not allow_failure:
        detail = (completed.stderr or "").strip().splitlines()
        message = detail[0] if detail else f"exit code {completed.returncode}"
        raise GitError(f"git {' '.join(arguments[:2])} failed: {message}")

    return (completed.stdout or "")[:MAX_GIT_OUTPUT_CHARS]


def _resolve_repository(
    path: str | os.PathLike[str], policy: PathPolicy | None
) -> tuple[Path, Path]:
    """Return ``(requested_directory, repository_root)``, both inside the roots.

    The repository root is re-checked against the policy: a worktree whose
    ``.git`` points elsewhere must not pull inspection outside the workspace.
    """
    active = policy if policy is not None else default_policy()
    target = active.resolve_within_roots(path)
    directory = target if target.is_dir() else target.parent

    toplevel = _run_git(directory, ["rev-parse", "--show-toplevel"], allow_failure=True).strip()
    if not toplevel:
        raise NotARepository(directory)

    root = active.resolve_within_roots(toplevel)
    return directory, root


# --------------------------------------------------------------------------
# result types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StatusEntry:
    path: str
    index_status: str
    worktree_status: str
    staged: bool
    unstaged: bool
    untracked: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "index_status": self.index_status,
            "worktree_status": self.worktree_status,
            "staged": self.staged,
            "unstaged": self.unstaged,
            "untracked": self.untracked,
        }


@dataclass(frozen=True)
class RepositoryStatus:
    root: str
    branch: str
    detached: bool
    head_commit: str | None
    upstream: str | None
    ahead: int | None
    behind: int | None
    clean: bool
    staged_count: int
    unstaged_count: int
    untracked_count: int
    conflicted_count: int
    entries: list[StatusEntry] = field(default_factory=list)
    entries_truncated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "branch": self.branch,
            "detached": self.detached,
            "head_commit": self.head_commit,
            "upstream": self.upstream,
            "ahead": self.ahead,
            "behind": self.behind,
            "clean": self.clean,
            "staged_count": self.staged_count,
            "unstaged_count": self.unstaged_count,
            "untracked_count": self.untracked_count,
            "conflicted_count": self.conflicted_count,
            "entries": [entry.as_dict() for entry in self.entries],
            "entries_truncated": self.entries_truncated,
        }

    def audit_summary(self) -> str:
        return (
            f"status {self.root}: branch={self.branch} clean={self.clean} "
            f"staged={self.staged_count} unstaged={self.unstaged_count} "
            f"untracked={self.untracked_count}"
        )


@dataclass(frozen=True)
class BranchInfo:
    root: str
    branch: str
    detached: bool
    head_commit: str | None
    upstream: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "branch": self.branch,
            "detached": self.detached,
            "head_commit": self.head_commit,
            "upstream": self.upstream,
        }

    def audit_summary(self) -> str:
        return f"branch {self.branch} (detached={self.detached}) in {self.root}"


@dataclass(frozen=True)
class CommitInfo:
    hash: str
    short_hash: str
    author: str
    date: str
    subject: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "hash": self.hash,
            "short_hash": self.short_hash,
            "author": self.author,
            "date": self.date,
            "subject": self.subject,
        }


@dataclass(frozen=True)
class DiffFileStat:
    path: str
    insertions: int | None
    deletions: int | None
    binary: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "insertions": self.insertions,
            "deletions": self.deletions,
            "binary": self.binary,
        }


@dataclass(frozen=True)
class DiffSummary:
    root: str
    staged: bool
    files_changed: int
    insertions: int
    deletions: int
    files: list[DiffFileStat] = field(default_factory=list)
    truncated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "staged": self.staged,
            "files_changed": self.files_changed,
            "insertions": self.insertions,
            "deletions": self.deletions,
            "files": [stat.as_dict() for stat in self.files],
            "truncated": self.truncated,
        }

    def audit_summary(self) -> str:
        return (
            f"diff {self.root} (staged={self.staged}): {self.files_changed} files "
            f"+{self.insertions}/-{self.deletions}"
        )


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------


def _parse_porcelain_v2(output: str) -> dict[str, Any]:
    branch = ""
    detached = False
    head_commit: str | None = None
    upstream: str | None = None
    ahead: int | None = None
    behind: int | None = None
    entries: list[StatusEntry] = []
    conflicted = 0
    truncated = False

    for line in output.splitlines():
        if not line:
            continue
        if line.startswith("# branch.head "):
            branch = line[len("# branch.head ") :].strip()
            detached = branch == "(detached)"
        elif line.startswith("# branch.oid "):
            oid = line[len("# branch.oid ") :].strip()
            head_commit = None if oid == "(initial)" else oid
        elif line.startswith("# branch.upstream "):
            upstream = line[len("# branch.upstream ") :].strip()
        elif line.startswith("# branch.ab "):
            for token in line[len("# branch.ab ") :].split():
                try:
                    if token.startswith("+"):
                        ahead = int(token[1:])
                    elif token.startswith("-"):
                        behind = int(token[1:])
                except ValueError:
                    continue
        elif line.startswith("# "):
            continue
        elif line[0] in {"1", "2", "u", "?"}:
            if len(entries) >= MAX_STATUS_ENTRIES:
                truncated = True
                continue
            if line[0] == "?":
                entries.append(
                    StatusEntry(
                        path=line[2:],
                        index_status="?",
                        worktree_status="?",
                        staged=False,
                        unstaged=False,
                        untracked=True,
                    )
                )
                continue
            parts = line.split(" ")
            if len(parts) < 2:
                continue
            xy = parts[1]
            if line[0] == "u":
                conflicted += 1
                path = parts[-1]
                entries.append(
                    StatusEntry(
                        path=path,
                        index_status=xy[0] if xy else "U",
                        worktree_status=xy[1] if len(xy) > 1 else "U",
                        staged=False,
                        unstaged=True,
                        untracked=False,
                    )
                )
                continue
            # Ordinary (1) or renamed (2) entry; path is the final field.
            path = parts[-1]
            if line[0] == "2" and "\t" in path:
                path = path.split("\t", 1)[0]
            index_status = xy[0] if xy else "."
            worktree_status = xy[1] if len(xy) > 1 else "."
            entries.append(
                StatusEntry(
                    path=path,
                    index_status=index_status,
                    worktree_status=worktree_status,
                    staged=index_status not in {".", "?"},
                    unstaged=worktree_status not in {".", "?"},
                    untracked=False,
                )
            )

    return {
        "branch": branch,
        "detached": detached,
        "head_commit": head_commit,
        "upstream": upstream,
        "ahead": ahead,
        "behind": behind,
        "entries": entries,
        "conflicted": conflicted,
        "truncated": truncated,
    }


@audited("git.get_repository_status", Capability.GIT_INSPECT)
def get_repository_status(
    path: str | os.PathLike[str],
    *,
    policy: PathPolicy | None = None,
    timeout: float = GIT_TIMEOUT_SECONDS,
) -> RepositoryStatus:
    """Return working-tree status for the repository containing ``path``."""
    _directory, root = _resolve_repository(path, policy)
    output = _run_git(
        root,
        ["status", "--porcelain=v2", "--branch", "--untracked-files=normal", "--no-renames"],
        timeout=timeout,
    )
    parsed = _parse_porcelain_v2(output)
    entries: list[StatusEntry] = parsed["entries"]

    staged = sum(1 for entry in entries if entry.staged)
    unstaged = sum(1 for entry in entries if entry.unstaged and not entry.untracked)
    untracked = sum(1 for entry in entries if entry.untracked)

    return RepositoryStatus(
        root=str(root),
        branch=parsed["branch"] or "(unknown)",
        detached=bool(parsed["detached"]),
        head_commit=parsed["head_commit"],
        upstream=parsed["upstream"],
        ahead=parsed["ahead"],
        behind=parsed["behind"],
        clean=not entries,
        staged_count=staged,
        unstaged_count=unstaged,
        untracked_count=untracked,
        conflicted_count=int(parsed["conflicted"]),
        entries=entries,
        entries_truncated=bool(parsed["truncated"]),
    )


@audited("git.get_current_branch", Capability.GIT_INSPECT)
def get_current_branch(
    path: str | os.PathLike[str],
    *,
    policy: PathPolicy | None = None,
    timeout: float = GIT_TIMEOUT_SECONDS,
) -> BranchInfo:
    """Return the current branch (or detached-HEAD marker) and its upstream."""
    _directory, root = _resolve_repository(path, policy)

    branch = _run_git(
        root, ["rev-parse", "--abbrev-ref", "HEAD"], timeout=timeout, allow_failure=True
    ).strip()
    head = _run_git(root, ["rev-parse", "HEAD"], timeout=timeout, allow_failure=True).strip()
    upstream = _run_git(
        root,
        ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"],
        timeout=timeout,
        allow_failure=True,
    ).strip()

    if not branch:
        # A repository with no commits yet: read the symbolic ref directly.
        branch = _run_git(
            root, ["symbolic-ref", "--short", "HEAD"], timeout=timeout, allow_failure=True
        ).strip() or "(unknown)"

    return BranchInfo(
        root=str(root),
        branch=branch,
        detached=branch == "HEAD",
        head_commit=head or None,
        upstream=upstream or None,
    )


@audited("git.get_recent_commits", Capability.GIT_INSPECT)
def get_recent_commits(
    path: str | os.PathLike[str],
    *,
    limit: int = 10,
    policy: PathPolicy | None = None,
    timeout: float = GIT_TIMEOUT_SECONDS,
) -> list[CommitInfo]:
    """Return the most recent commits. Author e-mail is deliberately omitted."""
    _directory, root = _resolve_repository(path, policy)
    count = max(1, min(int(limit), MAX_COMMITS))

    pretty = _FIELD_SEPARATOR.join(["%H", "%h", "%an", "%ad", "%s"])
    output = _run_git(
        root,
        [
            "log",
            f"--max-count={count}",
            "--date=iso-strict",
            f"--pretty=format:{pretty}",
            "--no-color",
            "--no-decorate",
        ],
        timeout=timeout,
        allow_failure=True,
    )

    commits: list[CommitInfo] = []
    for line in output.splitlines():
        fields = line.split(_FIELD_SEPARATOR)
        if len(fields) < 5:
            continue
        commits.append(
            CommitInfo(
                hash=fields[0],
                short_hash=fields[1],
                author=fields[2],
                date=fields[3],
                subject=fields[4][:500],
            )
        )
    return commits


@audited("git.get_diff_summary", Capability.GIT_INSPECT)
def get_diff_summary(
    path: str | os.PathLike[str],
    *,
    staged: bool = False,
    policy: PathPolicy | None = None,
    timeout: float = GIT_TIMEOUT_SECONDS,
    max_files: int = 200,
) -> DiffSummary:
    """Return per-file insertion/deletion counts. Never the diff text itself."""
    _directory, root = _resolve_repository(path, policy)

    arguments = ["diff", "--numstat", "--no-color", "--no-ext-diff"]
    if staged:
        arguments.append("--cached")
    output = _run_git(root, arguments, timeout=timeout, allow_failure=True)

    files: list[DiffFileStat] = []
    insertions = 0
    deletions = 0
    truncated = False

    for line in output.splitlines():
        fields = line.split("\t")
        if len(fields) < 3:
            continue
        if len(files) >= max_files:
            truncated = True
            break
        added_raw, removed_raw, file_path = fields[0], fields[1], fields[-1]
        binary = added_raw == "-" or removed_raw == "-"
        added = None if binary else int(added_raw) if added_raw.isdigit() else 0
        removed = None if binary else int(removed_raw) if removed_raw.isdigit() else 0
        if added:
            insertions += added
        if removed:
            deletions += removed
        files.append(
            DiffFileStat(path=file_path, insertions=added, deletions=removed, binary=binary)
        )

    return DiffSummary(
        root=str(root),
        staged=staged,
        files_changed=len(files),
        insertions=insertions,
        deletions=deletions,
        files=files,
        truncated=truncated,
    )
