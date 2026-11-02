"""Path security policy for the Omarchy Agent filesystem bridge.

Every filesystem access in this project must pass through :class:`PathPolicy`.
The policy answers three questions:

1. Is the fully resolved path inside one of the permitted workspace roots?
2. Is the path (or any of its parents) a directory we refuse to walk into?
3. Is the file itself something that may hold secrets and must never be read?

The policy is deliberately conservative: anything it cannot positively
classify as safe is rejected.
"""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "MAX_FILE_BYTES",
    "PathPolicy",
    "PathSecurityError",
    "default_policy",
    "default_workspace_roots",
]


class PathSecurityError(PermissionError):
    """Raised when a path violates the workspace policy."""

    def __init__(self, message: str, path: str | os.PathLike[str] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.path = str(path) if path is not None else None

    def __str__(self) -> str:  # pragma: no cover - trivial
        if self.path:
            return f"{self.message} (path: {self.path})"
        return self.message


#: Hard cap on how many bytes the bridge will read from a single file.
MAX_FILE_BYTES = 128 * 1024

#: Directory names that are never listed, walked or searched. These are either
#: noise (build output, caches) or credential stores.
BLOCKED_DIRECTORY_NAMES: frozenset[str] = frozenset(
    {
        # version control internals
        ".git",
        ".hg",
        ".svn",
        ".jj",
        # dependency trees
        "node_modules",
        "bower_components",
        "vendor",
        "site-packages",
        # python environments and caches
        ".venv",
        "venv",
        "env",
        ".env.d",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        ".eggs",
        # build output
        "build",
        "dist",
        "out",
        "target",
        ".next",
        ".nuxt",
        ".svelte-kit",
        ".turbo",
        ".parcel-cache",
        ".gradle",
        ".terraform",
        # generic caches
        ".cache",
        "cache",
        ".idea",
        ".vscode-server",
        # credential stores -- never traversed even if symlinked into a root
        ".ssh",
        ".gnupg",
        ".aws",
        ".azure",
        ".kube",
        ".docker",
        ".password-store",
        ".mozilla",
        ".thunderbird",
        ".pki",
        "keyrings",
    }
)

#: Glob patterns for files that may contain secrets. Matched case-insensitively
#: against the file name only.
SENSITIVE_FILE_PATTERNS: tuple[str, ...] = (
    # dotenv family
    ".env",
    ".env.*",
    "*.env",
    # private keys and certificates
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "*.pem",
    "*.key",
    "*.pkcs12",
    "*.p12",
    "*.pfx",
    "*.jks",
    "*.keystore",
    "*.ovpn",
    "*.asc",
    "*.gpg",
    # credential files
    ".netrc",
    "_netrc",
    ".pgpass",
    ".my.cnf",
    ".npmrc",
    ".pypirc",
    ".git-credentials",
    ".dockercfg",
    "credentials",
    "credentials.json",
    "client_secret*.json",
    "service-account*.json",
    "*token*.json",
    "secrets.*",
    "*.secret",
    "*.secrets",
    "*_secret",
    "*_secrets",
    "htpasswd",
    ".htpasswd",
    "shadow",
    # password / browser credential databases
    "*.kdbx",
    "*.kdb",
    "*.agilekeychain",
    "logins.json",
    "key3.db",
    "key4.db",
    "signons.sqlite",
    "cookies.sqlite",
    "Login Data",
    "Cookies",
)

#: Patterns that override :data:`SENSITIVE_FILE_PATTERNS`. These are the
#: harmless documentation/example variants that projects commit on purpose.
SENSITIVE_ALLOWLIST_PATTERNS: tuple[str, ...] = (
    ".env.example",
    ".env.sample",
    ".env.template",
    ".env.defaults",
    ".env.dist",
    ".env.test.example",
    "*.env.example",
    "*.env.sample",
    "*.env.template",
    "*.example",
    "*.sample",
    "*.template",
    "*.dist",
)


def default_workspace_roots() -> tuple[Path, ...]:
    """Return the default permitted workspace roots.

    The default is ``~/Projects``. It can be overridden with the
    ``OMARCHY_AGENT_WORKSPACE_ROOTS`` environment variable, which takes an
    ``os.pathsep``-separated list of directories. Entries that do not exist are
    dropped so a typo can never silently widen the policy.
    """
    raw = os.environ.get("OMARCHY_AGENT_WORKSPACE_ROOTS", "").strip()
    if raw:
        candidates = [Path(part).expanduser() for part in raw.split(os.pathsep) if part.strip()]
    else:
        candidates = [Path.home() / "Projects"]

    roots: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if resolved.is_dir() and resolved not in roots:
            roots.append(resolved)
    return tuple(roots)


@dataclass(frozen=True)
class PathPolicy:
    """Immutable path-access policy.

    Parameters
    ----------
    roots:
        Fully resolved directories the policy permits. A path is only allowed
        if its real location is one of these roots or below one of them.
    max_file_bytes:
        Maximum size of a file the bridge is willing to read.
    """

    roots: tuple[Path, ...]
    max_file_bytes: int = MAX_FILE_BYTES

    @classmethod
    def from_paths(
        cls,
        paths: "list[str | os.PathLike[str]] | tuple[str | os.PathLike[str], ...]",
        max_file_bytes: int = MAX_FILE_BYTES,
    ) -> "PathPolicy":
        """Build a policy from arbitrary path-likes, resolving each one."""
        roots: list[Path] = []
        for raw in paths:
            resolved = Path(raw).expanduser().resolve()
            if resolved not in roots:
                roots.append(resolved)
        return cls(roots=tuple(roots), max_file_bytes=max_file_bytes)

    # -- root containment ------------------------------------------------

    def describe_roots(self) -> list[str]:
        return [str(root) for root in self.roots]

    def _containing_root(self, resolved: Path) -> Path | None:
        for root in self.roots:
            if resolved == root:
                return root
            if resolved.is_relative_to(root):
                return root
        return None

    def is_within_roots(self, path: str | os.PathLike[str]) -> bool:
        """Return ``True`` if ``path`` resolves inside a permitted root."""
        try:
            resolved = self._resolve(path)
        except PathSecurityError:
            return False
        return self._containing_root(resolved) is not None

    # -- classification --------------------------------------------------

    @staticmethod
    def is_blocked_directory_name(name: str) -> bool:
        """Return ``True`` for directory names the bridge refuses to traverse."""
        if name in BLOCKED_DIRECTORY_NAMES:
            return True
        lowered = name.lower()
        if lowered in BLOCKED_DIRECTORY_NAMES:
            return True
        return lowered.endswith(".egg-info")

    @classmethod
    def is_sensitive_file_name(cls, name: str) -> bool:
        """Return ``True`` if a file name looks like it may contain secrets."""
        lowered = name.lower()
        for pattern in SENSITIVE_ALLOWLIST_PATTERNS:
            if fnmatch.fnmatch(lowered, pattern.lower()):
                return False
        for pattern in SENSITIVE_FILE_PATTERNS:
            if fnmatch.fnmatch(lowered, pattern.lower()):
                return True
        return False

    # -- the single entry point used by the bridge -----------------------

    @staticmethod
    def _resolve(path: str | os.PathLike[str]) -> Path:
        """Fully resolve ``path``, following every symlink component.

        ``strict=False`` is used so that a non-existent path still resolves to
        a canonical location; existence is checked separately by callers that
        need it. Resolution happens *before* any containment check, which is
        what defeats ``..`` traversal and symlink escapes.
        """
        try:
            return Path(path).expanduser().resolve()
        except (OSError, RuntimeError) as exc:  # e.g. symlink loop
            raise PathSecurityError(f"path could not be resolved: {exc}", path) from exc

    def resolve_within_roots(
        self,
        path: str | os.PathLike[str],
        *,
        must_exist: bool = True,
    ) -> Path:
        """Resolve ``path`` and assert it is a permitted target.

        Raises
        ------
        PathSecurityError
            If the path escapes the permitted roots, lives under a blocked
            directory, is a sensitive file, or (when ``must_exist``) is missing.
        """
        if not self.roots:
            raise PathSecurityError("no permitted workspace roots are configured", path)

        raw = str(path)
        if "\x00" in raw:
            raise PathSecurityError("path contains a NUL byte", raw)

        resolved = self._resolve(path)
        root = self._containing_root(resolved)
        if root is None:
            raise PathSecurityError(
                "path is outside the permitted workspace roots "
                f"({', '.join(self.describe_roots())})",
                resolved,
            )

        if must_exist and not resolved.exists():
            raise PathSecurityError("path does not exist", resolved)

        relative_parts = resolved.relative_to(root).parts
        # The final component is only a "directory name" if it is a directory;
        # check parents unconditionally and the leaf only when it is a dir.
        parent_parts = relative_parts[:-1] if relative_parts else ()
        for part in parent_parts:
            if self.is_blocked_directory_name(part):
                raise PathSecurityError(
                    f"path lies inside a blocked directory: {part!r}", resolved
                )
        if relative_parts:
            leaf = relative_parts[-1]
            if resolved.is_dir() and self.is_blocked_directory_name(leaf):
                raise PathSecurityError(
                    f"path is a blocked directory: {leaf!r}", resolved
                )
            if not resolved.is_dir() and self.is_sensitive_file_name(leaf):
                raise PathSecurityError(
                    "refusing to access a file that may contain secrets", resolved
                )
        return resolved


_DEFAULT_POLICY: PathPolicy | None = None


def default_policy() -> PathPolicy:
    """Return the process-wide default policy (``~/Projects``, cached)."""
    global _DEFAULT_POLICY
    if _DEFAULT_POLICY is None:
        _DEFAULT_POLICY = PathPolicy(roots=default_workspace_roots())
    return _DEFAULT_POLICY
