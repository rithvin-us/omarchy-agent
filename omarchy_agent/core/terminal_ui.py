"""Rich-based terminal presentation for the interactive LocalAI chat."""

from __future__ import annotations

import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TextIO

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def _safe_terminal_text(value: str) -> str:
    """Remove control characters before rendering untrusted model text."""
    return _CONTROL_CHARS.sub("", value)


class TerminalUI:
    """Render phase progress, live token usage, and a Markdown chat response."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream if stream is not None else sys.stdout
        self.is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.console = Console(
            file=self.stream,
            force_terminal=self.is_tty,
            highlight=False,
            markup=False,
        )
        self._live: Live | None = None
        self._answer = ""
        self._phase = "Ready"
        self._usage: dict[str, int | bool] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated": True,
        }

    def header(self, model: str, root: str | Path) -> None:
        root_text = str(root)
        home = str(Path.home())
        if root_text == home:
            root_text = "~"
        elif root_text.startswith(home + "/"):
            root_text = "~" + root_text[len(home) :]
        title = Text("Omarchy Agent · LocalAI", style="bold cyan")
        details = Text.assemble(
            (_safe_terminal_text(model), "bold"),
            ("  ·  ", "dim"),
            (_safe_terminal_text(root_text), "dim"),
        )
        help_text = Text("/help · /files · /clear · /exit", style="dim")
        self.console.print(Panel.fit(Group(title, details, help_text), border_style="cyan"))

    def begin_turn(self) -> None:
        self.finish_activity()
        self._answer = ""
        self._phase = "Preparing your request"
        self._usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated": True,
        }
        if self.is_tty:
            self._live = Live(
                self._render_live(),
                console=self.console,
                auto_refresh=True,
                refresh_per_second=12,
                screen=False,
                transient=True,
                vertical_overflow="visible",
            )
            self._live.start()

    def status(self, message: str) -> None:
        """Translate orchestration events into visible, factual progress phases."""
        clean = _safe_terminal_text(message)
        lowered = clean.lower()
        if "searching repository" in lowered or "context: searching" in lowered:
            self._set_phase("Finding relevant repository files", animate=True)
        elif lowered.startswith("context:"):
            detail = clean.partition(":")[2].strip()
            self._set_phase(f"Repository context ready · {detail}")
        elif lowered.startswith("model: analyzing"):
            self._set_phase("Analyzing your request", animate=True)
        elif lowered.startswith("model: reviewing"):
            self._set_phase("Reviewing repository results", animate=True)
        elif lowered.startswith("model: preparing"):
            self._set_phase("Preparing the final answer", animate=True)
        elif lowered.startswith("generating response"):
            self._set_phase("Analyzing your request", animate=True)
        elif lowered.startswith("tool result ready:"):
            tool_name = clean.partition(":")[2].strip()
            self._set_phase(f"Read-only tool finished · {tool_name}")
        elif lowered.startswith("tool:"):
            tool_name = clean.partition(":")[2].strip()
            self._set_phase(f"Calling read-only tool · {tool_name}", animate=True)
        elif lowered == "response complete":
            self._set_phase("Formatting the answer")
        else:
            self._set_phase(clean)

    def update_usage(self, usage: Mapping[str, int | bool]) -> None:
        """Update the live counter with estimated or server-reported token totals."""
        self._usage = {
            "prompt_tokens": _as_nonnegative_int(usage.get("prompt_tokens")),
            "completion_tokens": _as_nonnegative_int(usage.get("completion_tokens")),
            "total_tokens": _as_nonnegative_int(usage.get("total_tokens")),
            "estimated": bool(usage.get("estimated", True)),
        }
        self._refresh_live()

    def write_delta(self, fragment: str) -> None:
        """Append streamed answer text to the live Markdown render."""
        self._answer += _safe_terminal_text(fragment)
        self._phase = "Writing the answer"
        self._refresh_live()

    def finish_turn(self, answer: str) -> None:
        self.finish_activity()
        final_answer = _safe_terminal_text(answer)
        if final_answer:
            self._answer = final_answer
        self._stop_live()
        if self._answer:
            self.console.print(Markdown(self._answer))
        else:
            self.console.print(Text("(empty response)", style="dim"))
        self.console.print(self._usage_text())
        self._answer = ""

    def end_turn(self) -> None:
        """Close the live view cleanly after an interrupted or failed turn."""
        self.finish_activity()
        self._stop_live()
        if self._answer:
            self.console.print(Markdown(self._answer))
            self.console.print(self._usage_text())
        self._answer = ""

    def finish_activity(self) -> None:
        """Compatibility hook used by the CLI when a turn ends with an error."""
        self._stop_live()

    def error(self, error: BaseException) -> None:
        detail = _safe_terminal_text(str(error))
        label = _safe_terminal_text(type(error).__name__)
        Console(file=sys.stderr, force_terminal=sys.stderr.isatty(), markup=False).print(
            Text(f"error: {label}: {detail}", style="bold red")
        )

    def _set_phase(self, message: str, *, animate: bool = False) -> None:
        self._phase = message
        if not self.is_tty and animate:
            self.console.print(Text(f"  · {message}", style="dim"))
        self._refresh_live()

    def _refresh_live(self) -> None:
        if self._live is not None:
            self._live.update(self._render_live(), refresh=False)

    def _render_live(self) -> Group:
        activity = Group(
            Spinner("dots", text=Text(self._phase, style="cyan"), style="cyan"),
            self._usage_text(),
        )
        if self._answer:
            return Group(activity, Markdown(self._answer))
        return activity

    def _usage_text(self) -> Text:
        prompt = _as_nonnegative_int(self._usage.get("prompt_tokens"))
        completion = _as_nonnegative_int(self._usage.get("completion_tokens"))
        total = _as_nonnegative_int(self._usage.get("total_tokens"))
        estimated = bool(self._usage.get("estimated", True))
        marker = "≈ " if estimated else ""
        suffix = " estimated" if estimated else ""
        return Text(
            f"{marker}Input {prompt:,} · Output {completion:,} · Total {total:,} tokens{suffix}",
            style="dim",
        )

    def _stop_live(self) -> None:
        if self._live is None:
            return
        self._live.stop()
        self._live = None


def _as_nonnegative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)
