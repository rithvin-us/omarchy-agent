"""Bridges: the agent's read-only view of the host.

Each bridge function is a self-contained tool with plain arguments and a
dataclass result, so a later milestone can expose the same functions over MCP
without touching their implementation.
"""

from . import filesystem, git, system

__all__ = ["filesystem", "git", "system"]
