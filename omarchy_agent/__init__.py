"""Omarchy Agent -- a local, auditable automation foundation for Omarchy Linux.

v0.1 is an **inspection** foundation. It can:

* read files inside permitted workspace roots;
* report OS, CPU, memory, storage, GPU and process information;
* inspect git repositories read-only;
* confirm the local LocalAI API is up and list its installed models;
* keep persistent tasks, memories and tool-execution metadata in SQLite.

It cannot -- by design, enforced centrally in
:mod:`omarchy_agent.security.permissions` -- run shell commands, modify files,
install packages, control the desktop, drive a browser, or touch Docker.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
