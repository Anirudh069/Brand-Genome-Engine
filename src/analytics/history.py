"""
history.py – Live analysis_history counters and score trend.

These are computed directly from SQLite at request time (never cached) so
the Analytics page reflects new Consistency/Benchmark/Rewrite events
immediately without requiring a rebuild of the analytics artifact.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

SUPPORTED_EVENT_TYPES = ["consistency", "benchmark", "rewrite"]


def _extract_score(value: Any) -> float | None:
    if value is None:
        return None
    payload = value
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    if isinstance(payload, dict):
        for key in ("overall_score", "score", "overall"):
            candidate = payload.get(key)
            if isinstance(candidate, (int, float)):
                return float(candidate)
        return None
    if isinstance(payload, (int, float)):
        return float(payload)
    return None


def compute_history_counters(conn: sqlite3.Connection) -> dict[str, int]:
    cur = conn.cursor()
    counts = {event_type: 0 for event_type in SUPPORTED_EVENT_TYPES}
    cur.execute(
        "SELECT event_type, COUNT(*) FROM analysis_history "
        "WHERE event_type IS NOT NULL GROUP BY event_type"
    )
    for event_type, count in cur.fetchall():
        if event_type in counts:
            counts[event_type] = count
    counts["total"] = sum(counts.values())
    return counts


def compute_score_trend(conn: sqlite3.Connection) -> list[dict]:
    cur = conn.cursor()
    cur.execute(
        "SELECT event_type, pre_score, post_score, created_at FROM analysis_history "
        "WHERE event_type IS NOT NULL ORDER BY created_at ASC, id ASC"
    )
    trend: list[dict] = []
    for event_type, pre_score, post_score, created_at in cur.fetchall():
        score = _extract_score(pre_score)
        if score is None:
            score = _extract_score(post_score)
        if score is None:
            continue
        trend.append({"timestamp": created_at, "event_type": event_type, "score": round(score, 2)})
    return trend
