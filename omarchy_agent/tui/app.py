"""Full-screen terminal UI: markdown transcript, streaming replies, voice, approvals."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any, Protocol

from rich.markup import escape
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Input, Label, Markdown, Static

HELP = """\
**Keys**
- `F2` talk (push-to-talk, local Whisper)
- `Ctrl+S` speak replies (local Piper)
- `Ctrl+L` clear · `Ctrl+Q` quit · `Up`/`Down` input history

**Commands**
- `/help` this text · `/clear` · `/speak` toggle spoken replies
- `/talk` record one utterance · `/quit`

Gated tools (file writes, shell, calendar, browser clicks) ask `y` / `n` first."""

SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class Session(Protocol):
    on_status: Callable[[str], None]
    on_confirm_command: Callable[[str, Any], bool] | None

    def run_turn(self, question: str, *, on_delta: Callable[[str], None] | None = ...) -> str: ...
    def close(self) -> None: ...


class ApprovalScreen(ModalScreen[bool]):
    """Y/N prompt for ORANGE/RED tool calls. Escape or anything else means no."""

    BINDINGS = [
        Binding("y", "answer(True)", "Approve"),
        Binding("n,escape", "answer(False)", "Deny"),
    ]
    DEFAULT_CSS = """
    ApprovalScreen { align: center middle; background: $background 60%; }
    #box { width: 72%; max-width: 100; padding: 1 2; border: round $warning;
           background: $surface; height: auto; }
    #box .title { text-style: bold; color: $warning; }
    #box .keys { margin-top: 1; color: $text-muted; }
    """

    def __init__(self, description: str) -> None:
        super().__init__()
        self.description_text = description

    def compose(self) -> ComposeResult:
        with Vertical(id="box"):
            yield Label("Approval needed", classes="title")
            yield Label(self.description_text[:600], markup=False)
            yield Label("[b]y[/b] approve    [b]n[/b] / [b]esc[/b] deny", classes="keys")

    def action_answer(self, value: bool) -> None:
        self.dismiss(value)


class AgentTUI(App[None]):
    TITLE = "omarchy-agent"
    BINDINGS = [
        Binding("f2", "talk", "Talk"),
        Binding("ctrl+s", "toggle_speak", "Speak"),
        Binding("ctrl+l", "clear", "Clear"),
        Binding("ctrl+q", "quit", "Quit"),
    ]
    CSS = """
    #chat { height: 1fr; padding: 0 2; scrollbar-size-vertical: 1; }
    .you { margin: 1 0 0 0; padding: 0 1; border-left: thick $accent; color: $text; }
    .agent { margin: 1 0 0 0; padding: 0 1; border-left: thick $success; }
    .agent > MarkdownBlock { margin: 0; }
    .tool { margin: 0 0 0 2; color: $text-muted; text-style: italic; }
    .sys { margin: 1 0 0 0; color: $text-muted; }
    .err { margin: 1 0 0 0; padding: 0 1; border-left: thick $error; color: $error; }
    #status { height: 1; padding: 0 2; background: $boost; color: $text-muted; }
    #input { margin: 0 1; }
    """

    def __init__(
        self,
        session: Session,
        *,
        transcriber: Any = None,
        speaker: Any = None,
        record: Callable[[], Any] | None = None,
        subtitle: str = "",
    ) -> None:
        super().__init__()
        self.session = session
        self.transcriber = transcriber
        self.speaker = speaker
        self._record = record
        self._subtitle = subtitle
        self.speak_replies = False
        self.transcript: list[str] = []  # plain "who: text" lines, for tests and export
        self._busy = threading.Lock()
        self._activity = ""
        self._frame = 0
        self._history: list[str] = []
        self._hist_pos = 0
        session.on_status = self._status_from_thread
        session.on_confirm_command = self._confirm_from_thread

    # -- layout ----------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield VerticalScroll(id="chat")
        yield Static("ready", id="status")
        yield Input(placeholder="Message… (F2 talk · /help)", id="input")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = self._subtitle
        self.query_one(Input).focus()
        self.set_interval(0.1, self._tick)
        voice = "voice ready" if self.transcriber else "voice off (uv sync --extra voice)"
        self._add("sys", f"Type a message or press F2 to talk. /help for keys. {voice}.")

    def _tick(self) -> None:
        if self._activity:
            self._frame = (self._frame + 1) % len(SPINNER)
            self._render_status(f"{SPINNER[self._frame]} {self._activity}")

    # -- rendering helpers -----------------------------------------------
    def _render_status(self, text: str) -> None:
        voice = " · speak:on" if self.speak_replies else ""
        self.query_one("#status", Static).update(escape(text) + voice)

    def _set_status(self, text: str) -> None:
        self._activity = ""
        self._render_status(text)

    def _chat(self) -> VerticalScroll:
        return self.query_one("#chat", VerticalScroll)

    def _add(self, kind: str, text: str) -> Any:
        """Append one transcript entry. Agent text renders as markdown."""
        if kind == "agent":
            widget: Any = Markdown(text, classes="agent")
        else:
            prefix = {"you": "", "err": "error: "}.get(kind, "")
            widget = Static(escape(prefix + text), classes=kind, markup=True)
        self.transcript.append(f"{kind}: {text}")
        chat = self._chat()
        chat.mount(widget)
        chat.scroll_end(animate=False)
        return widget

    # -- thread-safe status / approval -----------------------------------
    def _status_from_thread(self, text: str) -> None:
        self.call_from_thread(self._on_session_status, text)

    def _on_session_status(self, text: str) -> None:
        if text.startswith("tool:"):
            self._add("tool", f"↳ {text[5:].strip()}")
        if text.startswith(("model:", "tool", "context")):
            self._activity = text
        elif text == "response complete":
            self._activity = ""

    def _confirm_from_thread(self, description: str, _cwd: Any) -> bool:
        done = threading.Event()
        result = {"ok": False}

        def show() -> None:
            def got(value: bool | None) -> None:
                result["ok"] = value is True
                done.set()

            self.push_screen(ApprovalScreen(description), got)

        self.call_from_thread(show)
        done.wait(timeout=300)
        return result["ok"]

    # -- actions ---------------------------------------------------------
    def action_clear(self) -> None:
        self._chat().remove_children()
        self.transcript.clear()

    def action_toggle_speak(self) -> None:
        if self.speaker is None:
            self._set_status("speech unavailable (uv sync --extra voice)")
            return
        self.speak_replies = not self.speak_replies
        self._set_status(f"speak replies: {'on' if self.speak_replies else 'off'}")

    def action_talk(self) -> None:
        if self.transcriber is None or self._record is None:
            self._set_status("voice unavailable (uv sync --extra voice)")
            return
        self._listen()

    def on_key(self, event: Any) -> None:
        if not isinstance(self.focused, Input) or not self._history:
            return
        if event.key == "up":
            self._hist_pos = max(0, self._hist_pos - 1)
        elif event.key == "down":
            self._hist_pos = min(len(self._history), self._hist_pos + 1)
        else:
            return
        event.stop()
        value = self._history[self._hist_pos] if self._hist_pos < len(self._history) else ""
        self.focused.value = value
        self.focused.cursor_position = len(value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        self._history.append(text)
        self._hist_pos = len(self._history)
        if text.startswith("/"):
            self._command(text)
        else:
            self._submit(text)

    def _command(self, text: str) -> None:
        name = text.split()[0].lower()
        if name == "/help":
            self._add("agent", HELP)
        elif name == "/clear":
            self.action_clear()
        elif name == "/speak":
            self.action_toggle_speak()
        elif name == "/talk":
            self.action_talk()
        elif name in {"/quit", "/exit"}:
            self.exit()
        else:
            self._add("err", f"unknown command {name}; try /help")

    def _submit(self, text: str) -> None:
        self._add("you", text)
        self._activity = "thinking"
        self._run_chat(text)

    # -- workers ---------------------------------------------------------
    @work(thread=True, exclusive=True, group="listen")
    def _listen(self) -> None:
        self.call_from_thread(self._start_activity, "listening… (speak now)")
        try:
            audio = self._record() if self._record else None
            if audio is None:
                self.call_from_thread(self._set_status, "heard nothing")
                return
            self.call_from_thread(self._start_activity, "transcribing…")
            heard = self.transcriber.transcribe(audio)
        except Exception as exc:  # noqa: BLE001 - surface any audio failure in the UI
            self.call_from_thread(self._set_status, f"voice error: {type(exc).__name__}")
            return
        if heard:
            self.call_from_thread(self._submit, heard)
        else:
            self.call_from_thread(self._set_status, "heard nothing")

    def _start_activity(self, text: str) -> None:
        self._activity = text

    @work(thread=True, group="chat")
    def _run_chat(self, text: str) -> None:
        if not self._busy.acquire(blocking=False):
            self.call_from_thread(self._add, "sys", "still working on the previous message")
            return
        holder: dict[str, Any] = {}
        buffer: list[str] = []

        def show_stream() -> None:
            if "w" not in holder:
                holder["w"] = self._add("agent", "".join(buffer))
                self.transcript.pop()  # replaced by the final entry below
            else:
                holder["w"].update("".join(buffer))
            self._chat().scroll_end(animate=False)

        def on_delta(fragment: str) -> None:
            buffer.append(fragment)
            self.call_from_thread(show_stream)

        try:
            answer = self.session.run_turn(text, on_delta=on_delta)
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(self._add, "err", f"{type(exc).__name__}: {exc}")
            self.call_from_thread(self._set_status, "ready")
            return
        finally:
            self._busy.release()

        def finish() -> None:
            if "w" in holder:
                holder["w"].update(answer)
                self.transcript.append(f"agent: {answer}")
            else:
                self._add("agent", answer)
            self._set_status("ready")

        self.call_from_thread(finish)
        if self.speak_replies and self.speaker is not None:
            self.call_from_thread(self._start_activity, "speaking…")
            try:
                self.speaker.say(answer)
            except Exception as exc:  # noqa: BLE001
                self.call_from_thread(self._set_status, f"speech error: {type(exc).__name__}")
                return
            self.call_from_thread(self._set_status, "ready")

    def on_unmount(self) -> None:
        self.session.close()
