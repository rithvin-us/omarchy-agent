"""Full-screen terminal UI: transcript, input, push-to-talk, spoken replies, approvals."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any, Protocol

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Input, Label, RichLog, Static


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
    ApprovalScreen { align: center middle; }
    #box { width: 70%; padding: 1 2; border: thick $warning; background: $surface; height: auto; }
    """

    def __init__(self, description: str) -> None:
        super().__init__()
        self.description_text = description

    def compose(self) -> ComposeResult:
        with Vertical(id="box"):
            yield Label("[b]Approval needed[/b]")
            yield Label(self.description_text[:500], markup=False)
            yield Label("\n[b]y[/b] approve    [b]n[/b] / esc deny")

    def action_answer(self, value: bool) -> None:
        self.dismiss(value)


class AgentTUI(App[None]):
    TITLE = "omarchy-agent"
    BINDINGS = [
        Binding("f2", "talk", "Talk"),
        Binding("ctrl+s", "toggle_speak", "Speak replies"),
        Binding("ctrl+l", "clear", "Clear"),
        Binding("ctrl+q", "quit", "Quit"),
    ]
    CSS = """
    #log { height: 1fr; padding: 0 1; }
    #live { height: auto; padding: 0 1; color: $text-muted; }
    #status { height: 1; padding: 0 1; color: $text-muted; }
    """

    def __init__(
        self,
        session: Session,
        *,
        transcriber: Any = None,
        speaker: Any = None,
        record: Callable[[], Any] | None = None,
    ) -> None:
        super().__init__()
        self.session = session
        self.transcriber = transcriber
        self.speaker = speaker
        self._record = record
        self.speak_replies = False
        self._busy = threading.Lock()
        session.on_status = self._status_from_thread
        session.on_confirm_command = self._confirm_from_thread

    # -- layout ----------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(id="log", wrap=True, markup=False, highlight=False)
        yield Static("", id="live")
        yield Static("ready", id="status")
        yield Input(placeholder="message  (F2: talk, ctrl+s: speak replies)", id="input")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(Input).focus()
        if self.transcriber is None:
            self._set_status("voice unavailable (uv sync --extra voice)")

    # -- thread-safe helpers ---------------------------------------------
    def _set_status(self, text: str) -> None:
        self.query_one("#status", Static).update(text)

    def _status_from_thread(self, text: str) -> None:
        self.call_from_thread(self._set_status, text)

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

    def _write(self, who: str, text: str) -> None:
        self.query_one(RichLog).write(f"{who}: {text}")

    # -- actions ---------------------------------------------------------
    def action_clear(self) -> None:
        self.query_one(RichLog).clear()

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

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if text:
            self._submit(text)

    def _submit(self, text: str) -> None:
        self._write("you", text)
        self._chat(text)

    # -- workers ---------------------------------------------------------
    @work(thread=True, exclusive=True, group="listen")
    def _listen(self) -> None:
        self.call_from_thread(self._set_status, "listening...")
        try:
            audio = self._record() if self._record else None
            if audio is None:
                self.call_from_thread(self._set_status, "heard nothing")
                return
            self.call_from_thread(self._set_status, "transcribing...")
            heard = self.transcriber.transcribe(audio)
        except Exception as exc:  # noqa: BLE001 - surface any audio failure in the UI
            self.call_from_thread(self._set_status, f"voice error: {type(exc).__name__}")
            return
        if heard:
            self.call_from_thread(self._submit, heard)
        else:
            self.call_from_thread(self._set_status, "heard nothing")

    @work(thread=True, group="chat")
    def _chat(self, text: str) -> None:
        if not self._busy.acquire(blocking=False):
            self.call_from_thread(self._write, "sys", "still working on the previous message")
            return
        buffer: list[str] = []

        def on_delta(fragment: str) -> None:
            buffer.append(fragment)
            self.call_from_thread(self.query_one("#live", Static).update, "".join(buffer)[-400:])

        try:
            answer = self.session.run_turn(text, on_delta=on_delta)
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(self._write, "error", f"{type(exc).__name__}: {exc}")
            return
        finally:
            self._busy.release()
            self.call_from_thread(self.query_one("#live", Static).update, "")
        self.call_from_thread(self._write, "agent", answer)
        self.call_from_thread(self._set_status, "ready")
        if self.speak_replies and self.speaker is not None:
            try:
                self.speaker.say(answer)
            except Exception as exc:  # noqa: BLE001
                self.call_from_thread(self._set_status, f"speech error: {type(exc).__name__}")

    def on_unmount(self) -> None:
        self.session.close()
