"""Open real desktop windows and browser tabs (Omarchy/Hyprland), not just terminal text."""

from __future__ import annotations

import configparser
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .browser import BrowserError, validate_url

__all__ = [
    "DesktopError",
    "find_desktop_entry",
    "focus_window",
    "list_windows",
    "open_app",
    "open_file",
    "open_url",
]

APP_DIRS = (
    Path("/usr/share/applications"),
    Path("~/.local/share/applications").expanduser(),
    Path("/var/lib/flatpak/exports/share/applications"),
)


class DesktopError(RuntimeError):
    """A desktop action was refused or failed."""


def _spawn(argv: list[str]) -> None:
    """Start a detached GUI process; never uses a shell."""
    exe = shutil.which(argv[0])
    if exe is None:
        raise DesktopError(f"{argv[0]} not found")
    try:
        subprocess.Popen(  # noqa: S603
            [exe, *argv[1:]],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        raise DesktopError(f"could not start {argv[0]}: {exc}") from exc


def open_url(url: str) -> dict[str, Any]:
    """Open an http(s) URL as a new tab in the user's real browser."""
    try:
        safe = validate_url(url)
    except BrowserError as exc:
        raise DesktopError(str(exc)) from exc
    launcher = "omarchy-launch-browser" if shutil.which("omarchy-launch-browser") else "xdg-open"
    _spawn([launcher, safe])
    return {"opened": safe, "via": launcher}


def open_file(path: Path) -> dict[str, Any]:
    """Open a file or directory with its default application."""
    if not path.exists():
        raise DesktopError(f"no such path: {path}")
    _spawn(["xdg-open", str(path)])
    return {"opened": str(path)}


def _entries() -> dict[str, tuple[str, str]]:
    """Map desktop-entry id -> (display name, exec-name) for visible applications."""
    found: dict[str, tuple[str, str]] = {}
    for directory in APP_DIRS:
        if not directory.is_dir():
            continue
        for file in directory.glob("*.desktop"):
            parser = configparser.ConfigParser(interpolation=None, strict=False)
            try:
                parser.read(file, encoding="utf-8")
                section = parser["Desktop Entry"]
            except (configparser.Error, OSError, KeyError, UnicodeDecodeError):
                continue
            if section.get("NoDisplay", "false").lower() == "true":
                continue
            if section.get("Type", "Application") != "Application":
                continue
            found.setdefault(file.stem, (section.get("Name", file.stem), section.get("Exec", "")))
    return found


def find_desktop_entry(name: str) -> str | None:
    """Best match for a user-spoken app name: exact id/name, then prefix, then substring."""
    needle = name.strip().lower()
    if not needle:
        return None
    entries = _entries()
    for matcher in (
        lambda i, n: needle in {i.lower(), n.lower()},
        lambda i, n: i.lower().startswith(needle) or n.lower().startswith(needle),
        lambda i, n: needle in i.lower() or needle in n.lower(),
    ):
        hits = sorted(
            (i for i, (n, _e) in entries.items() if matcher(i, n)), key=lambda i: (len(i), i)
        )
        if hits:
            return hits[0]
    return None


def open_app(name: str) -> dict[str, Any]:
    """Launch an installed desktop application by name (e.g. 'firefox', 'spotify')."""
    entry = find_desktop_entry(name)
    if entry is None:
        raise DesktopError(f"no installed application matches {name!r}")
    argv = ["gtk-launch", entry]
    if shutil.which("uwsm-app"):
        argv = ["uwsm-app", "--", *argv]
    _spawn(argv)
    return {"launched": entry}


def list_windows() -> list[dict[str, Any]]:
    """Open windows from Hyprland: address, class, title, workspace."""
    if not os.environ.get("HYPRLAND_INSTANCE_SIGNATURE"):
        raise DesktopError("not running inside a Hyprland session")
    try:
        out = subprocess.run(  # noqa: S603
            [shutil.which("hyprctl") or "hyprctl", "clients", "-j"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
        clients = json.loads(out)
    except (subprocess.SubprocessError, OSError, ValueError) as exc:
        raise DesktopError(f"hyprctl failed: {type(exc).__name__}") from exc
    return [
        {
            "address": c.get("address"),
            "class": c.get("class"),
            "title": str(c.get("title", ""))[:120],
            "workspace": c.get("workspace", {}).get("name"),
        }
        for c in clients
        if c.get("mapped")
    ]


def focus_window(address: str) -> dict[str, Any]:
    """Focus a window by the address from list_windows."""
    if not address.startswith("0x") or not address[2:].isalnum():
        raise DesktopError("address must look like 0x55d0c1a0")
    _spawn(["hyprctl", "dispatch", "focuswindow", f"address:{address}"])
    return {"focused": address}
