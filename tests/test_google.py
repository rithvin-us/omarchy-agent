"""Socket-free tests for free-time maths, risk tiers, and Google tool gating."""

import unittest
from datetime import datetime, timedelta
from unittest import mock

from omarchy_agent.core.chat import ChatSession
from omarchy_agent.google.freetime import calculate_free_slots, merge_intervals
from omarchy_agent.security.risk import RiskLevel, needs_approval, risk_of

T = datetime(2026, 1, 1, 9, 0)


def at(h: float) -> datetime:
    return T + timedelta(hours=h - 9)


class FreeTimeTests(unittest.TestCase):
    def test_merge_overlap_and_touching(self) -> None:
        got = merge_intervals([(at(9), at(10)), (at(10), at(11)), (at(13), at(14))])
        self.assertEqual(got, [(at(9), at(11)), (at(13), at(14))])

    def test_slots_respect_duration_and_buffer(self) -> None:
        busy = [(at(10), at(11)), (at(12), at(12.25))]
        slots = calculate_free_slots(busy, at(9), at(14), 60, 0)
        self.assertEqual(
            [(s.start, s.end) for s in slots],
            [(at(9), at(10)), (at(11), at(12)), (at(12.25), at(14))],
        )
        buffered = calculate_free_slots(busy, at(9), at(14), 60, 15)
        self.assertEqual([(s.start, s.end) for s in buffered], [(at(12.5), at(14))])

    def test_empty_window(self) -> None:
        self.assertEqual(calculate_free_slots([], at(10), at(9), 30), [])


class RiskTests(unittest.TestCase):
    def test_tiers(self) -> None:
        self.assertFalse(needs_approval("gmail_search"))
        self.assertTrue(needs_approval("calendar_create"))
        self.assertEqual(risk_of("calendar_delete"), RiskLevel.RED)

    def test_unknown_tool_is_red(self) -> None:
        self.assertEqual(risk_of("brand_new_tool"), RiskLevel.RED)


class GateTests(unittest.TestCase):
    def _session(self, confirm: bool | None) -> ChatSession:
        cb = None if confirm is None else (lambda *_: confirm)
        return ChatSession(mock.Mock(), ".", on_confirm_command=cb)

    def test_orange_tool_rejected_without_approval(self) -> None:
        s = self._session(None)
        s.google = mock.Mock()
        out = s._dispatch_tool(
            "calendar_create", '{"summary":"x","start":"2026-01-01T09:00","end":"2026-01-01T10:00"}'
        )
        self.assertIn("rejected", out["error"])
        s.google.calendar_create.assert_not_called()

    def test_orange_tool_runs_when_approved(self) -> None:
        s = self._session(True)
        s.google = mock.Mock()
        s.google.calendar_create.return_value = {"id": "e1"}
        out = s._dispatch_tool(
            "calendar_create", '{"summary":"x","start":"2026-01-01T09:00","end":"2026-01-01T10:00"}'
        )
        self.assertEqual(out, {"result": {"id": "e1"}})

    def test_green_tool_needs_no_approval(self) -> None:
        s = self._session(None)
        s.google = mock.Mock()
        s.google.gmail_search.return_value = []
        self.assertEqual(s._dispatch_tool("gmail_search", '{"query":"is:unread"}'), {"result": []})


class MemoryToolTests(unittest.TestCase):
    def test_remember_runs_and_forget_needs_approval(self) -> None:
        mem = mock.Mock()
        mem.remember.return_value = {"ok": 1}
        s = ChatSession(mock.Mock(), ".", memory=mem)
        out = s._dispatch_tool("remember", '{"category":"pref","key":"tz","content":"IST"}')
        self.assertEqual(out, {"result": {"ok": 1}})
        denied = s._dispatch_tool("forget", '{"category":"pref","key":"tz"}')
        self.assertIn("rejected", denied["error"])
        mem.forget.assert_not_called()

    def test_memory_unavailable_is_an_error_not_a_crash(self) -> None:
        s = ChatSession(mock.Mock(), ".")
        out = s._dispatch_tool("recall", '{"query":"x"}')
        self.assertIn("not available", out["error"])

    def test_preface_lists_memories(self) -> None:
        rec = mock.Mock(category="pref", key="tz", content="IST")
        mem = mock.Mock()
        mem.list_memories.return_value = [rec]
        s = ChatSession(mock.Mock(), ".", memory=mem)
        self.assertIn("pref/tz: IST", s._memory_preface())
