"""Headless Textual tests with a fake session (no model, no audio)."""

import asyncio
import unittest

try:
    from textual.widgets import Input, RichLog

    from omarchy_agent.tui.app import AgentTUI

    HAVE_TEXTUAL = True
except ImportError:  # pragma: no cover
    HAVE_TEXTUAL = False


class FakeSession:
    def __init__(self) -> None:
        self.on_status = lambda _m: None
        self.on_confirm_command = None
        self.closed = False
        self.approved: bool | None = None

    def run_turn(self, question, *, on_delta=None):
        if on_delta:
            on_delta("par")
        if "danger" in question:
            self.approved = self.on_confirm_command("[RED] calendar_delete {}", None)
            return f"approved={self.approved}"
        return f"echo: {question}"

    def close(self) -> None:
        self.closed = True


class FakeTranscriber:
    def transcribe(self, _audio):
        return "spoken words"


async def _settle(pilot, predicate, tries=100):
    for _ in range(tries):
        await pilot.pause(0.05)
        if predicate():
            return True
    return False


def _log_text(app) -> str:
    return "\n".join(line.text for line in app.query_one(RichLog).lines)


@unittest.skipUnless(HAVE_TEXTUAL, "textual not installed")
class TuiTests(unittest.TestCase):
    def test_typed_message_gets_reply(self) -> None:
        async def go() -> None:
            app = AgentTUI(FakeSession())
            async with app.run_test() as pilot:
                app.query_one(Input).value = "hello"
                await pilot.press("enter")
                self.assertTrue(await _settle(pilot, lambda: "echo: hello" in _log_text(app)))

        asyncio.run(go())

    def test_approval_modal_yes_and_default_deny(self) -> None:
        async def go(key: str, expect: str) -> None:
            session = FakeSession()
            app = AgentTUI(session)
            async with app.run_test() as pilot:
                app.query_one(Input).value = "danger"
                await pilot.press("enter")
                await pilot.pause(0.5)
                await pilot.press(key)
                self.assertTrue(await _settle(pilot, lambda: expect in _log_text(app)))

        asyncio.run(go("y", "approved=True"))
        asyncio.run(go("escape", "approved=False"))

    def test_push_to_talk_submits_transcript(self) -> None:
        async def go() -> None:
            app = AgentTUI(FakeSession(), transcriber=FakeTranscriber(), record=lambda: object())
            async with app.run_test() as pilot:
                await pilot.press("f2")
                self.assertTrue(
                    await _settle(pilot, lambda: "echo: spoken words" in _log_text(app))
                )

        asyncio.run(go())
