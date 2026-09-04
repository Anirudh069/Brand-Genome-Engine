# src/scoring/analysis_log.py
# Person C — Phase 4A deliverable 4: the analysis_history logging contract
#
# Defines exactly what is written to analysis_history for every scoring or
# rewrite run, and provides the SQL behind the UI counters. Person D calls
# log_analysis() once per request; nothing else writes to this table.
#
# One row = one run. The diagnostics_json column holds the full payload so a
# past run can be re-displayed without recomputation, while the flat pre_score
# and post_score columns stay indexable for the counters.

import json
import sqlite3
from datetime import datetime, timezone

from src.profiles.brand_profile_builder import SQLITE_DB_PATH

EVENT_CONSISTENCY = "consistency"
EVENT_REWRITE = "rewrite"
EVENT_BENCHMARK = "benchmark"
VALID_EVENTS = {EVENT_CONSISTENCY, EVENT_REWRITE, EVENT_BENCHMARK}

LOG_SCHEMA_VERSION = 1

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS analysis_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    brand_id        TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    input_text      TEXT,
    pre_score       REAL,
    post_score      REAL,
    diagnostics_json TEXT NOT NULL,
    extra_json      TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
)
"""

INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_history_brand ON analysis_history(brand_id)",
    "CREATE INDEX IF NOT EXISTS idx_history_event ON analysis_history(event_type)",
]


# Columns this module needs, with the DDL used to add them to a table that
# already exists. Person A's earlier schema had analysis_history with
# (brand_id, analysis_type, result_json, created_at); CREATE TABLE IF NOT
# EXISTS silently does nothing against that table, so every insert failed with
# "no such column: event_type". Adding the missing columns in place keeps her
# existing rows and avoids a second, competing history table.
REQUIRED_COLUMNS = {
    "event_type": "TEXT NOT NULL DEFAULT 'consistency'",
    "input_text": "TEXT",
    "pre_score": "REAL",
    "post_score": "REAL",
    "diagnostics_json": "TEXT NOT NULL DEFAULT '{}'",
    "extra_json": "TEXT",
    "created_at": "TEXT",
}


def init_history_table(db_path=SQLITE_DB_PATH):
    """
    Create analysis_history if absent, or bring an older version of the table
    up to the columns this contract needs. Safe to call repeatedly.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(CREATE_SQL)

    existing = {row[1] for row in cur.execute("PRAGMA table_info(analysis_history)")}
    for column, ddl in REQUIRED_COLUMNS.items():
        if column not in existing:
            cur.execute(f"ALTER TABLE analysis_history ADD COLUMN {column} {ddl}")

    for sql in INDEX_SQL:
        cur.execute(sql)
    conn.commit()
    conn.close()


class _ScoreLike:
    """Minimal stand-in exposing .overall_score and .to_dict()."""

    def __init__(self, data):
        self._data = dict(data or {})
        self.overall_score = self._data.get("overall_score")

    def to_dict(self):
        return dict(self._data)


def score_from_dict(scores):
    """
    Wrap a plain score dict so it can be logged like a ScoreResult.

    The API layer works with dicts — that is what it returns to the frontend —
    while this module reads `.overall_score` and `.to_dict()`. This adapter lets
    both sides stay as they are instead of forcing one to convert.
    """
    if scores is None:
        return None
    if hasattr(scores, "to_dict"):
        return scores
    return _ScoreLike(scores)


def build_log_payload(brand_id, event_type, score_before=None, score_after=None,
                      diagnostics=None, drift_report=None, edit_plan=None,
                      extra=None):
    """
    Build the diagnostics_json payload.

    THIS IS THE LOGGING CONTRACT. Every key below is always present; a value is
    null when it does not apply to the event. Person D can rely on the shape
    without checking which event type produced the row.

    {
      "schema_version": 1,
      "event_type": "consistency" | "rewrite" | "benchmark",
      "brand_id": str,
      "scores": {"before": {...} | null, "after": {...} | null},
      "diagnostics": {...} | null,
      "drift_report": {...} | null,
      "edit_plan": {...} | null,
      "extra": {...},
      "logged_at": ISO-8601 UTC
    }
    """
    if event_type not in VALID_EVENTS:
        raise ValueError(
            f"event_type must be one of {sorted(VALID_EVENTS)}, got {event_type!r}")

    def dump(obj):
        if obj is None:
            return None
        return obj.to_dict() if hasattr(obj, "to_dict") else obj

    return {
        "schema_version": LOG_SCHEMA_VERSION,
        "event_type": event_type,
        "brand_id": brand_id,
        "scores": {
            "before": dump(score_before),
            "after": dump(score_after),
        },
        "diagnostics": dump(diagnostics),
        "drift_report": dump(drift_report),
        "edit_plan": dump(edit_plan),
        "extra": extra or {},
        "logged_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def log_analysis(brand_id, event_type, input_text="", score_before=None,
                 score_after=None, diagnostics=None, drift_report=None,
                 edit_plan=None, extra=None, db_path=SQLITE_DB_PATH):
    """
    Write one analysis_history row. Returns the new row id.

    Logging must never break the request that triggered it: a failure here is
    swallowed and reported as -1 rather than raised, because a user losing
    their rewrite because an analytics insert failed is the worse outcome.
    """
    payload = build_log_payload(
        brand_id, event_type, score_before, score_after,
        diagnostics, drift_report, edit_plan, extra,
    )

    try:
        init_history_table(db_path)
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO analysis_history "
            "(brand_id, event_type, input_text, pre_score, post_score, "
            " diagnostics_json, extra_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                brand_id,
                event_type,
                (input_text or "")[:5000],
                getattr(score_before, "overall_score", None),
                getattr(score_after, "overall_score", None),
                json.dumps(payload),
                json.dumps(extra or {}),
                payload["logged_at"],
            ),
        )
        row_id = cur.lastrowid
        conn.commit()
        conn.close()
        return row_id
    except sqlite3.Error as exc:      # pragma: no cover - defensive
        print(f"[analysis_log] WARNING: failed to log analysis: {exc}")
        return -1


# ── Counter queries (Person D's Analytics page) ───────────────────────────────
#
# Kept here as named constants so the frontend and the tests read the same SQL.

SQL_COPIES_ANALYSED = """
SELECT COUNT(*) FROM analysis_history
WHERE event_type IN ('consistency', 'rewrite')
"""

SQL_AVG_CONSISTENCY = """
SELECT AVG(COALESCE(post_score, pre_score)) FROM analysis_history
WHERE COALESCE(post_score, pre_score) IS NOT NULL
"""

SQL_DEVIATIONS_FIXED = """
SELECT COUNT(*) FROM analysis_history
WHERE event_type = 'rewrite'
  AND post_score IS NOT NULL
  AND pre_score IS NOT NULL
  AND post_score > pre_score
"""


def get_counters(db_path=SQLITE_DB_PATH):
    """
    Return the three UI counters.

    copies_analysed  — consistency checks + rewrites
    avg_consistency  — mean final score (post-rewrite where one exists)
    deviations_fixed — rewrites that actually improved the score
    """
    try:
        init_history_table(db_path)
        conn = sqlite3.connect(db_path)
        copies = conn.execute(SQL_COPIES_ANALYSED).fetchone()[0] or 0
        avg = conn.execute(SQL_AVG_CONSISTENCY).fetchone()[0]
        fixed = conn.execute(SQL_DEVIATIONS_FIXED).fetchone()[0] or 0
        conn.close()
    except sqlite3.Error:             # pragma: no cover - defensive
        return {"copies_analysed": 0, "avg_consistency": 0.0, "deviations_fixed": 0}

    return {
        "copies_analysed": int(copies),
        "avg_consistency": round(float(avg), 1) if avg is not None else 0.0,
        "deviations_fixed": int(fixed),
    }


def get_trend(db_path=SQLITE_DB_PATH, points=5):
    """
    Rolling average of the last runs, oldest first — for the Analytics sparkline.

    Returns [] when there is not enough history, so the UI can hide the chart
    rather than draw a flat line through seeded placeholder values.
    """
    try:
        init_history_table(db_path)
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT COALESCE(post_score, pre_score) AS s FROM analysis_history "
            "WHERE COALESCE(post_score, pre_score) IS NOT NULL "
            "ORDER BY id DESC LIMIT ?", (points + 2,)
        ).fetchall()
        conn.close()
    except sqlite3.Error:              # pragma: no cover - defensive
        return []

    scores = [r[0] for r in rows][::-1]
    if len(scores) < 3:
        return []

    trend = []
    for i in range(len(scores)):
        window = scores[max(0, i - 2):i + 1]
        trend.append(round(sum(window) / len(window), 1))
    return trend[-points:]


def recent_runs(limit=20, brand_id=None, db_path=SQLITE_DB_PATH):
    """Most recent runs, newest first — for a history panel or debugging."""
    init_history_table(db_path)
    conn = sqlite3.connect(db_path)
    if brand_id:
        rows = conn.execute(
            "SELECT id, brand_id, event_type, pre_score, post_score, created_at "
            "FROM analysis_history WHERE brand_id = ? ORDER BY id DESC LIMIT ?",
            (brand_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, brand_id, event_type, pre_score, post_score, created_at "
            "FROM analysis_history ORDER BY id DESC LIMIT ?", (limit,),
        ).fetchall()
    conn.close()
    return [
        {"id": r[0], "brand_id": r[1], "event_type": r[2],
         "pre_score": r[3], "post_score": r[4], "created_at": r[5]}
        for r in rows
    ]
