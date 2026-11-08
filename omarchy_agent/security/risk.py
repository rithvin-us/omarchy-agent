"""Static tool risk tiers (ported from the old assistant's ADR-0005).

Risk is a property of the tool declaration. It is never read from model output.
"""

from __future__ import annotations

from enum import IntEnum

__all__ = ["TOOL_RISK", "RiskLevel", "needs_approval", "risk_of"]


class RiskLevel(IntEnum):
    GREEN = 0  # read-only
    YELLOW = 1  # reversible organisation of own data
    ORANGE = 2  # consequential but recoverable
    RED = 3  # destructive, irreversible, or visible to third parties


TOOL_RISK: dict[str, RiskLevel] = {
    "browser_open": RiskLevel.GREEN,
    "browser_snapshot": RiskLevel.GREEN,
    "browser_screenshot": RiskLevel.GREEN,
    "browser_click": RiskLevel.ORANGE,
    "browser_type": RiskLevel.ORANGE,
    "browser_tab": RiskLevel.YELLOW,
    "open_url": RiskLevel.YELLOW,
    "open_app": RiskLevel.ORANGE,
    "open_file": RiskLevel.ORANGE,
    "list_windows": RiskLevel.GREEN,
    "focus_window": RiskLevel.YELLOW,
    "gmail_search": RiskLevel.GREEN,
    "gmail_read": RiskLevel.GREEN,
    "calendar_list": RiskLevel.GREEN,
    "free_time": RiskLevel.GREEN,
    "write_file": RiskLevel.ORANGE,
    "edit_file": RiskLevel.ORANGE,
    "calendar_create": RiskLevel.ORANGE,
    "calendar_delete": RiskLevel.RED,
    "recall": RiskLevel.GREEN,
    "remember": RiskLevel.YELLOW,
    "forget": RiskLevel.ORANGE,
}


def risk_of(tool: str) -> RiskLevel:
    """Unlisted tools are treated as RED so a new tool cannot skip review."""
    return TOOL_RISK.get(tool, RiskLevel.RED)


def needs_approval(tool: str) -> bool:
    return risk_of(tool) >= RiskLevel.ORANGE
