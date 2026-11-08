"""Hardware-free tests for the voice helpers."""

import unittest
from unittest import mock

from omarchy_agent.voice.engine import speakable
from omarchy_agent.voice.loop import make_voice_confirm


class SpeakableTests(unittest.TestCase):
    def test_strips_code_urls_and_marks(self) -> None:
        text = "## Title\nRun `ls` now.\n```py\nprint(1)\n```\nSee https://x.io/a **bold**"
        out = speakable(text)
        self.assertNotIn("print", out)
        self.assertNotIn("https", out)
        self.assertNotIn("*", out)
        self.assertIn("Run ls now", out)

    def test_length_capped(self) -> None:
        self.assertLessEqual(len(speakable("word " * 1000)), 1200)


class VoiceConfirmTests(unittest.TestCase):
    def _confirm(self, heard: str | None) -> bool:
        speaker, transcriber = mock.Mock(), mock.Mock()
        transcriber.transcribe.return_value = heard or ""
        audio = None if heard is None else object()
        with mock.patch("omarchy_agent.voice.loop.record_utterance", return_value=audio):
            return make_voice_confirm(speaker, transcriber)("calendar_delete {}", None)

    def test_yes_approves(self) -> None:
        self.assertTrue(self._confirm("yes go ahead"))

    def test_silence_denies(self) -> None:
        self.assertFalse(self._confirm(None))

    def test_ambiguous_denies(self) -> None:
        self.assertFalse(self._confirm("no, do not do that"))

    def test_negated_yes_denies(self) -> None:
        self.assertFalse(self._confirm("no do not go ahead"))
