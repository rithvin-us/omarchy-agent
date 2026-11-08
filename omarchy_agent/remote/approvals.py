"""Blocking human-approval broker for remote clients (ORANGE/RED tool calls)."""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field

APPROVAL_TIMEOUT_SECONDS = 120.0


@dataclass
class _Pending:
    id: int
    description: str
    event: threading.Event = field(default_factory=threading.Event)
    approved: bool = False


class ApprovalBroker:
    """A tool thread calls :meth:`request` and blocks; a client resolves it via the API.

    Timeout, shutdown, and anything but an explicit approve all mean *denied*.
    """

    def __init__(self, timeout: float = APPROVAL_TIMEOUT_SECONDS) -> None:
        self.timeout = timeout
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._pending: dict[int, _Pending] = {}

    def request(self, description: str, on_pending=None) -> bool:  # noqa: ANN001
        item = _Pending(next(self._ids), description[:500])
        with self._lock:
            self._pending[item.id] = item
        try:
            if on_pending is not None:
                on_pending({"id": item.id, "description": item.description})
            item.event.wait(self.timeout)
            return item.approved
        finally:
            with self._lock:
                self._pending.pop(item.id, None)

    def pending(self) -> list[dict[str, object]]:
        with self._lock:
            return [
                {"id": p.id, "description": p.description}
                for p in self._pending.values()
                if not p.event.is_set()
            ]

    def resolve(self, approval_id: int, approve: bool) -> bool:
        with self._lock:
            item = self._pending.get(approval_id)
        if item is None or item.event.is_set():
            return False
        item.approved = bool(approve)
        item.event.set()
        return True
