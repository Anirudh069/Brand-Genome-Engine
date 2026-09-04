from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from typing import Any, Iterable

from src.feature_extraction.readability_extractor import extract_readability
from src.api.genome_service import USER_BRAND_DB_ID, is_user_brand_identifier, load_active_user_genome, write_history_event

SUPPORTED_BENCHMARK_METRICS = ("tone", "sentiment", "readability")


@dataclass(slots=True)
class BenchmarkError(Exception):
    status_code: int
    detail: dict[str, Any]

    def __str__(self) -> str:
        return str(self.detail.get("message") or self.detail.get("error") or "benchmark_error")


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def _profile_mapping(profile: dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(profile, dict):
        return profile
    return {}


def _profile_metric(profile: dict[str, Any], *keys: str, default: float) -> float:
    tone_features = _profile_mapping(profile.get("tone_features"))
    for key in keys:
        if key in profile:
            value = _to_float(profile.get(key), default)
            if math.isfinite(value):
                return value
        if key in tone_features:
            value = _to_float(tone_features.get(key), default)
            if math.isfinite(value):
                return value
    return default


def _normalise_designation(profile: dict[str, Any], fallback: str) -> str:
    for key in ("designation", "brand_name", "name"):
        value = profile.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


def _scale_percent(value: float) -> float:
    return round(max(0.0, min(100.0, value * 100.0)), 2)


def _scale_sentence_length(value: float) -> float:
    return round(max(0.0, min(100.0, value * 4.0)), 2)


def _is_usable_competitor_profile(profile: dict[str, Any]) -> bool:
    if not profile.get("brand_id") or not _normalise_designation(profile, ""):
        return False
    required = ("mean_formality", "mean_sentiment", "mean_flesch", "mean_vocab_richness")
    for key in required:
        value = _profile_metric(profile, key, default=float("nan"))
        if not math.isfinite(value):
            return False
    return True


def _load_competitor_profile_row(conn: sqlite3.Connection, competitor_brand_id: str) -> dict[str, Any] | None:
    cur = conn.cursor()
    row = cur.execute(
        "SELECT brand_id, profile_json FROM brand_profiles WHERE brand_id = ?",
        (competitor_brand_id,),
    ).fetchone()
    if row is None:
        return None

    profile = json.loads(row["profile_json"] or "{}")
    if not isinstance(profile, dict):
        return None
    profile.setdefault("brand_id", row["brand_id"])
    profile.setdefault("brand_name", profile.get("brand_name") or profile.get("designation") or str(row["brand_id"]).replace("_", " ").title())
    profile.setdefault("designation", profile.get("brand_name") or profile.get("brand_id") or "")
    return profile


def list_benchmark_competitors(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    cur = conn.cursor()
    rows = cur.execute("SELECT brand_id, profile_json FROM brand_profiles ORDER BY brand_id").fetchall()
    competitors: list[dict[str, Any]] = []
    for row in rows:
        profile = json.loads(row["profile_json"] or "{}")
        if not isinstance(profile, dict):
            continue
        profile.setdefault("brand_id", row["brand_id"])
        profile.setdefault("brand_name", profile.get("brand_name") or profile.get("designation") or str(row["brand_id"]).replace("_", " ").title())
        if is_user_brand_identifier(profile.get("brand_id")):
            continue
        if not profile.get("brand_name"):
            continue
        if not all(
            key in profile or key in _profile_mapping(profile.get("tone_features"))
            for key in ("mean_formality", "mean_sentiment", "mean_flesch", "mean_vocab_richness")
        ):
            continue
        competitors.append(
            {
                "brand_id": str(profile["brand_id"]),
                "designation": str(profile["brand_name"]),
            }
        )
    return competitors


_SENTENCE_LENGTH_CACHE: dict[str, float] = {}


def _competitor_sentence_length(conn: sqlite3.Connection, competitor_brand_id: str) -> float:
    cached = _SENTENCE_LENGTH_CACHE.get(competitor_brand_id)
    if cached is not None:
        return cached

    cur = conn.cursor()
    rows = cur.execute("SELECT text FROM brand_texts WHERE brand_id = ?", (competitor_brand_id,)).fetchall()
    if not rows:
        _SENTENCE_LENGTH_CACHE[competitor_brand_id] = 0.0
        return 0.0

    sentence_lengths: list[float] = []
    for row in rows:
        _, average_sentence_length = extract_readability(row["text"])
        sentence_lengths.append(float(average_sentence_length))

    value = round(sum(sentence_lengths) / len(sentence_lengths), 2) if sentence_lengths else 0.0
    _SENTENCE_LENGTH_CACHE[competitor_brand_id] = value
    return value


def _user_profile_from_genome(genome: dict[str, Any]) -> dict[str, Any]:
    tone_features = _profile_mapping(genome.get("tone_features"))
    designation = _normalise_designation(genome, "")
    return {
        "brand_id": USER_BRAND_DB_ID,
        "designation": designation,
        "brand_name": genome.get("brand_name") or designation,
        "tone_features": tone_features,
        "mean_formality": _profile_metric(genome, "mean_formality", "avg_formality", default=_to_float(tone_features.get("mean_formality"), 0.0)),
        "mean_sentiment": _profile_metric(genome, "mean_sentiment", "avg_sentiment", default=_to_float(tone_features.get("mean_sentiment"), 0.0)),
        "mean_flesch": _profile_metric(genome, "mean_flesch", "avg_readability_flesch", default=_to_float(tone_features.get("mean_flesch"), 0.0)),
        "avg_sentence_length": _profile_metric(genome, "avg_sentence_length", default=_to_float(tone_features.get("avg_sentence_length"), 0.0)),
        "mean_vocab_richness": _profile_metric(genome, "mean_vocab_richness", "vocabulary_richness", default=_to_float(tone_features.get("mean_vocab_richness"), 0.0)),
        "tone_label": tone_features.get("tone_label") or genome.get("tone_label") or genome.get("tone") or "",
    }


def _validate_profile(profile: dict[str, Any], *, role: str) -> None:
    if not profile:
        raise BenchmarkError(
            404,
            {
                "error": f"{role}_profile_missing",
                "message": f"{role.title()} profile is unavailable.",
            },
        )
    if not _normalise_designation(profile, "").strip():
        raise BenchmarkError(
            422,
            {
                "error": f"{role}_profile_unusable",
                "message": f"{role.title()} profile is missing a designation.",
            },
        )


def _build_metric_series(
    conn: sqlite3.Connection,
    metric: str,
    user_profile: dict[str, Any],
    competitor_profile: dict[str, Any],
) -> tuple[list[str], list[float], list[float], dict[str, Any]]:
    if metric == "tone":
        labels = ["Formality", "Sentence Length", "Vocabulary Richness"]
        competitor_sentence_length = _competitor_sentence_length(conn, str(competitor_profile["brand_id"]))
        user_series = [
            _scale_percent(_profile_metric(user_profile, "mean_formality", "avg_formality", default=0.0)),
            _scale_sentence_length(_profile_metric(user_profile, "avg_sentence_length", default=0.0)),
            _scale_percent(_profile_metric(user_profile, "mean_vocab_richness", "vocabulary_richness", default=0.0)),
        ]
        competitor_series = [
            _scale_percent(_profile_metric(competitor_profile, "mean_formality", "avg_formality", default=0.0)),
            _scale_sentence_length(competitor_sentence_length),
            _scale_percent(_profile_metric(competitor_profile, "mean_vocab_richness", "vocabulary_richness", default=0.0)),
        ]
        return labels, user_series, competitor_series, {
            "raw_labels": ["mean_formality", "avg_sentence_length", "mean_vocab_richness"],
            "raw_user_series": [
                round(_profile_metric(user_profile, "mean_formality", "avg_formality", default=0.0), 4),
                round(_profile_metric(user_profile, "avg_sentence_length", default=0.0), 2),
                round(_profile_metric(user_profile, "mean_vocab_richness", "vocabulary_richness", default=0.0), 4),
            ],
            "raw_competitor_series": [
                round(_profile_metric(competitor_profile, "mean_formality", "avg_formality", default=0.0), 4),
                round(competitor_sentence_length, 2),
                round(_profile_metric(competitor_profile, "mean_vocab_richness", "vocabulary_richness", default=0.0), 4),
            ],
        }
    if metric == "sentiment":
        labels = ["Sentiment"]
        user_series = [_scale_percent(_profile_metric(user_profile, "mean_sentiment", "avg_sentiment", default=0.0))]
        competitor_series = [_scale_percent(_profile_metric(competitor_profile, "mean_sentiment", "avg_sentiment", default=0.0))]
        return labels, user_series, competitor_series, {
            "raw_labels": ["mean_sentiment"],
            "raw_user_series": [round(_profile_metric(user_profile, "mean_sentiment", "avg_sentiment", default=0.0), 4)],
            "raw_competitor_series": [round(_profile_metric(competitor_profile, "mean_sentiment", "avg_sentiment", default=0.0), 4)],
        }
    if metric == "readability":
        labels = ["Flesch Reading Ease", "Average Sentence Length"]
        competitor_sentence_length = _competitor_sentence_length(conn, str(competitor_profile["brand_id"]))
        user_series = [
            round(_profile_metric(user_profile, "mean_flesch", "avg_readability_flesch", default=0.0), 2),
            _scale_sentence_length(_profile_metric(user_profile, "avg_sentence_length", default=0.0)),
        ]
        competitor_series = [
            round(_profile_metric(competitor_profile, "mean_flesch", "avg_readability_flesch", default=0.0), 2),
            _scale_sentence_length(competitor_sentence_length),
        ]
        return labels, user_series, competitor_series, {
            "raw_labels": ["mean_flesch", "avg_sentence_length"],
            "raw_user_series": [
                round(_profile_metric(user_profile, "mean_flesch", "avg_readability_flesch", default=0.0), 2),
                round(_profile_metric(user_profile, "avg_sentence_length", default=0.0), 2),
            ],
            "raw_competitor_series": [
                round(_profile_metric(competitor_profile, "mean_flesch", "avg_readability_flesch", default=0.0), 2),
                round(competitor_sentence_length, 2),
            ],
        }
    raise BenchmarkError(
        422,
        {
            "error": "invalid_metric",
            "message": f"Unsupported benchmark metric: {metric}",
        },
    )


def _tone_summary(competitor_name: str, labels: list[str], user_series: list[float], competitor_series: list[float]) -> str:
    tolerance = {"Formality": 1.0, "Sentence Length": 2.0, "Vocabulary Richness": 1.0}
    phrases: list[str] = []
    for label, user_value, competitor_value in zip(labels, user_series, competitor_series):
        delta = user_value - competitor_value
        if abs(delta) <= tolerance[label]:
            if label == "Formality":
                phrases.append("similar in formality")
            elif label == "Sentence Length":
                phrases.append("uses similar sentence lengths")
            else:
                phrases.append("uses similar vocabulary richness")
        elif label == "Formality":
            phrases.append("is more formal" if delta > 0 else "is less formal")
        elif label == "Sentence Length":
            phrases.append("uses longer sentences" if delta > 0 else "uses shorter sentences")
        else:
            phrases.append("uses richer vocabulary" if delta > 0 else "uses plainer vocabulary")
    return f"Compared with {competitor_name}, your brand {', '.join(phrases[:-1]) + ', and ' if len(phrases) > 1 else ''}{phrases[-1]}."


def _sentiment_summary(competitor_name: str, user_value: float, competitor_value: float) -> str:
    delta = user_value - competitor_value
    if abs(delta) <= 1.0:
        return f"Your brand and {competitor_name} have similar sentiment."
    return f"Your brand has {'higher' if delta > 0 else 'lower'} sentiment than {competitor_name} by {abs(delta):.1f} points."


def _readability_summary(competitor_name: str, user_series: list[float], competitor_series: list[float]) -> str:
    flesch_delta = user_series[0] - competitor_series[0]
    sentence_delta = user_series[1] - competitor_series[1]
    if abs(flesch_delta) <= 1.0 and abs(sentence_delta) <= 2.0:
        return f"Your brand and {competitor_name} have similar readability."
    if flesch_delta >= 1.0 and sentence_delta <= 2.0:
        return f"Your brand is easier to read than {competitor_name}, with a higher Flesch Reading Ease score and shorter sentences."
    if flesch_delta <= -1.0 and sentence_delta >= -2.0:
        return f"Your brand is harder to read than {competitor_name}, with a lower Flesch Reading Ease score and longer sentences."
    direction = "higher" if flesch_delta > 0 else "lower"
    length_phrase = "shorter" if sentence_delta < 0 else "longer"
    return f"Compared with {competitor_name}, your brand has a {direction} Flesch Reading Ease score but {length_phrase} sentences."


def _summary_text(metric: str, competitor_name: str, labels: list[str], user_series: list[float], competitor_series: list[float]) -> str:
    if metric == "tone":
        return _tone_summary(competitor_name, labels, user_series, competitor_series)
    if metric == "sentiment":
        return _sentiment_summary(competitor_name, user_series[0], competitor_series[0])
    if metric == "readability":
        return _readability_summary(competitor_name, user_series, competitor_series)
    raise BenchmarkError(422, {"error": "invalid_metric", "message": f"Unsupported benchmark metric: {metric}"})


def run_market_benchmark(conn: sqlite3.Connection, *, competitor_brand_id: str, metric: str) -> dict[str, Any]:
    metric_key = str(metric).strip().lower()
    if metric_key not in SUPPORTED_BENCHMARK_METRICS:
        raise BenchmarkError(
            422,
            {
                "error": "invalid_metric",
                "message": f"Unsupported benchmark metric: {metric}",
            },
        )

    genome = load_active_user_genome(conn)
    if not genome:
        raise BenchmarkError(
            400,
            {
                "error": "genome_not_initialized",
                "message": "Initialize the genome first via POST /api/genome/init.",
                "action": "setup_genome",
                "endpoint": "/api/genome/init",
            },
        )

    user_profile = _user_profile_from_genome(genome)
    if not user_profile.get("designation"):
        raise BenchmarkError(
            400,
            {
                "error": "genome_not_initialized",
                "message": "Initialize the genome first via POST /api/genome/init.",
                "action": "setup_genome",
                "endpoint": "/api/genome/init",
            },
        )

    competitor_key = str(competitor_brand_id).strip().lower()
    if not competitor_key:
        raise BenchmarkError(422, {"error": "missing_competitor_brand_id", "message": "competitor_brand_id is required."})
    if is_user_brand_identifier(competitor_key):
        raise BenchmarkError(400, {"error": "invalid_competitor", "message": "The user brand cannot be selected as a competitor."})

    competitor_profile = _load_competitor_profile_row(conn, competitor_key)
    if competitor_profile is None:
        raise BenchmarkError(404, {"error": "competitor_not_found", "message": "Competitor brand not found in SQLite."})
    if not _is_usable_competitor_profile(competitor_profile):
        raise BenchmarkError(422, {"error": "competitor_profile_unusable", "message": "Competitor profile is missing required benchmark data."})

    _validate_profile(user_profile, role="user")
    _validate_profile(competitor_profile, role="competitor")

    labels, user_series, competitor_series, metric_meta = _build_metric_series(conn, metric_key, user_profile, competitor_profile)
    if len(user_series) != len(labels) or len(competitor_series) != len(labels):
        raise BenchmarkError(500, {"error": "benchmark_series_mismatch", "message": "Benchmark series lengths do not match labels."})
    if not all(math.isfinite(value) for value in [*user_series, *competitor_series]):
        raise BenchmarkError(500, {"error": "benchmark_nonfinite", "message": "Benchmark produced non-finite values."})

    competitor_name = str(competitor_profile.get("brand_name") or competitor_profile.get("designation") or competitor_key).strip()
    summary_text = _summary_text(metric_key, competitor_name, labels, user_series, competitor_series)
    result = {
        "metric": metric_key,
        "user_series": user_series,
        "competitor_series": competitor_series,
        "labels": labels,
        "summary_text": summary_text,
        "user_brand": {
            "brand_id": USER_BRAND_DB_ID,
            "designation": user_profile.get("designation", ""),
        },
        "competitor_brand": {
            "brand_id": competitor_key,
            "designation": competitor_name,
        },
        "competitor_brand_id": competitor_key,
        "metric_details": metric_meta,
    }

    write_history_event(
        conn,
        brand_id=USER_BRAND_DB_ID,
        event_type="benchmark",
        input_text=None,
        pre_score=None,
        post_score=None,
        diagnostics_json=[],
        extra_json={
            "competitor_brand_id": competitor_key,
            "competitor_designation": competitor_name,
            "metric": metric_key,
            "labels": labels,
            "summary_text": summary_text,
        },
    )

    return result