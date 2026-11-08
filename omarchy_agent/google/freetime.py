"""Deterministic free-slot calculation (ported from the old Rust free_time.rs)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

__all__ = ["FreeSlot", "calculate_free_slots", "merge_intervals"]


@dataclass(frozen=True)
class FreeSlot:
    start: datetime
    end: datetime

    @property
    def minutes(self) -> int:
        return int((self.end - self.start).total_seconds() // 60)


def merge_intervals(intervals: list[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
    """Merge overlapping and touching intervals into sorted disjoint ones."""
    valid = sorted(i for i in intervals if i[0] < i[1])
    merged: list[tuple[datetime, datetime]] = []
    for start, end in valid:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def calculate_free_slots(
    busy: list[tuple[datetime, datetime]],
    window_start: datetime,
    window_end: datetime,
    duration_minutes: int,
    buffer_minutes: int = 0,
) -> list[FreeSlot]:
    """Free slots in the window at least ``duration_minutes`` long.

    Each busy interval is padded by ``buffer_minutes`` on both sides.
    """
    if window_start >= window_end or duration_minutes <= 0:
        return []
    pad = timedelta(minutes=max(0, buffer_minutes))
    clipped = [
        (max(s - pad, window_start), min(e + pad, window_end))
        for s, e in busy
        if e + pad > window_start and s - pad < window_end
    ]
    slots: list[FreeSlot] = []
    cursor = window_start
    for start, end in merge_intervals(clipped):
        if start > cursor:
            slots.append(FreeSlot(cursor, start))
        cursor = max(cursor, end)
    if cursor < window_end:
        slots.append(FreeSlot(cursor, window_end))
    return [s for s in slots if s.minutes >= duration_minutes]
