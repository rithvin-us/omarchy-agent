"""Hands-free voice conversation loop around a ChatSession."""

from __future__ import annotations

import re
from collections.abc import Callable

from ..core.chat import ChatSession
from .audio import record_utterance
from .engine import Speaker, Transcriber

EXIT_PHRASES = re.compile(r"^\W*(stop listening|exit|quit|goodbye|good bye)\W*$", re.I)
APPROVE = re.compile(r"\b(yes|yeah|yep|approve|approved|do it|go ahead|confirm)\b", re.I)


DENY = re.compile(r"\b(no|not|don'?t|stop|cancel|wait|never)\b", re.I)


def make_voice_confirm(speaker: Speaker, transcriber: Transcriber) -> Callable[[str, object], bool]:
    """Spoken approval: ask aloud, accept only an explicit 'yes'. Anything else is a no."""

    def confirm(description: str, _cwd: object) -> bool:
        speaker.say(f"Approval needed. {description[:200]}. Say yes to allow.")
        audio = record_utterance(wait_seconds=8.0)
        if audio is None:
            return False
        heard = transcriber.transcribe(audio)
        return bool(APPROVE.search(heard)) and not DENY.search(heard)

    return confirm


def run_voice_loop(
    session: ChatSession,
    transcriber: Transcriber,
    speaker: Speaker,
    *,
    on_text: Callable[[str, str], None] = lambda _role, _text: None,
) -> None:
    speaker.say("Voice mode ready.")
    while True:
        audio = record_utterance(wait_seconds=60.0)
        if audio is None:
            continue
        heard = transcriber.transcribe(audio)
        if not heard:
            continue
        on_text("you", heard)
        if EXIT_PHRASES.match(heard):
            speaker.say("Goodbye.")
            return
        answer = session.run_turn(heard)
        on_text("agent", answer)
        speaker.say(answer)
