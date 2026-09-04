from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

import numpy as np

from src.feature_extraction.embedding_extractor import get_embedding
from src.feature_extraction.formality_extractor import extract_formality
from src.feature_extraction.readability_extractor import extract_readability
from src.feature_extraction.sentiment_extractor import extract_sentiment
from src.feature_extraction.topic_extractor import extract_topics
from src.feature_extraction.vocabulary_extractor import extract_vocab_metrics

USER_BRAND_DB_ID = 0
USER_BRAND_ALIAS = "user_brand"
EMBEDDING_DIM = 384
SNIPPET_COUNT = 7


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _to_float(value: Any) -> float:
    return float(value)


def _to_float_list(values: list[Any]) -> list[float]:
    return [_to_float(value) for value in values]


def _tone_label(formality: float, sentiment: float) -> str:
    if formality >= 0.55 and sentiment >= 0.6:
        return "authoritative"
    if formality >= 0.55:
        return "formal"
    if sentiment >= 0.6:
        return "motivational"
    return "neutral"


def ensure_canonical_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS brands (
            id INTEGER PRIMARY KEY,
            designation TEXT,
            mission_core_vision TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS brand_profile (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brand_id INTEGER,
            keywords_json TEXT,
            tone_features_json TEXT,
            aggregate_embedding TEXT,
            metadata_json TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brand_id INTEGER,
            analysis_type TEXT,
            result_json TEXT,
            created_at TEXT
        )
        """
    )
    cur.execute("PRAGMA table_info(analysis_history)")
    analysis_columns = {row[1] for row in cur.fetchall()}
    required_columns = {
        "event_type": "TEXT",
        "input_text": "TEXT",
        "pre_score": "TEXT",
        "post_score": "TEXT",
        "diagnostics_json": "TEXT",
        "extra_json": "TEXT",
    }
    for column_name, column_type in required_columns.items():
        if column_name not in analysis_columns:
            cur.execute(f"ALTER TABLE analysis_history ADD COLUMN {column_name} {column_type}")


def validate_user_genome_input(
    designation: str,
    mission_core_vision: str,
    snippets: list[str],
) -> tuple[str, str, list[str]]:
    cleaned_designation = designation.strip() if isinstance(designation, str) else ""
    cleaned_mission = mission_core_vision.strip() if isinstance(mission_core_vision, str) else ""
    cleaned_snippets = [snippet.strip() if isinstance(snippet, str) else "" for snippet in snippets]

    if not cleaned_designation:
        raise ValueError("designation must be nonblank")
    if not cleaned_mission:
        raise ValueError("mission_core_vision must be nonblank")
    if len(cleaned_snippets) != SNIPPET_COUNT:
        raise ValueError("snippets must contain exactly 7 entries")
    if any(not snippet for snippet in cleaned_snippets):
        raise ValueError("snippets must not contain blank entries")

    return cleaned_designation, cleaned_mission, cleaned_snippets


def _build_feature_bundle(
    designation: str,
    mission_core_vision: str,
    snippets: list[str],
) -> dict[str, Any]:
    sample_texts = [mission_core_vision, *snippets]
    sample_features: list[dict[str, Any]] = []
    sample_embeddings: list[list[float]] = []
    embedding_model_name = "all-MiniLM-L6-v2"

    for index, text in enumerate(sample_texts):
        sentiment = _to_float(extract_sentiment(text))
        formality = _to_float(extract_formality(text))
        readability_flesch, avg_sentence_length = extract_readability(text)
        vocab_metrics = extract_vocab_metrics(text)
        top_topics, topic_weights = extract_topics(text, num_topics=5)
        embedding, embedding_model_name = get_embedding(text)
        embedding_list = _to_float_list(embedding)

        sample_embeddings.append(embedding_list)
        sample_features.append(
            {
                "index": index,
                "kind": "mission_core_vision" if index == 0 else "snippet",
                "sentiment": sentiment,
                "formality": formality,
                "readability_flesch": _to_float(readability_flesch),
                "avg_sentence_length": _to_float(avg_sentence_length),
                "vocab_diversity": _to_float(vocab_metrics["vocab_diversity"]),
                "punctuation_density": _to_float(vocab_metrics["punctuation_density"]),
                "top_topics": top_topics,
                "topic_weights": [float(weight) for weight in topic_weights],
                "embedding_dim": len(embedding_list),
            }
        )

    aggregate_embedding = np.mean(np.asarray(sample_embeddings, dtype=np.float64), axis=0)
    if aggregate_embedding.shape[0] != EMBEDDING_DIM:
        raise ValueError(f"aggregate embedding must be {EMBEDDING_DIM} dimensions")
    if not np.isfinite(aggregate_embedding).all():
        raise ValueError("aggregate embedding contains non-finite values")

    combined_text = " ".join(sample_texts)
    top_keywords, keyword_weights = extract_topics(combined_text, num_topics=10)

    mean_sentiment = float(np.mean([item["sentiment"] for item in sample_features]))
    mean_formality = float(np.mean([item["formality"] for item in sample_features]))
    mean_flesch = float(np.mean([item["readability_flesch"] for item in sample_features]))
    mean_sentence_length = float(np.mean([item["avg_sentence_length"] for item in sample_features]))
    mean_vocab_richness = float(np.mean([item["vocab_diversity"] for item in sample_features]))

    tone_label = _tone_label(mean_formality, mean_sentiment)

    tone_features = {
        "tone_label": tone_label,
        "mean_sentiment": round(mean_sentiment, 4),
        "avg_sentiment": round(mean_sentiment, 4),
        "mean_formality": round(mean_formality, 4),
        "avg_formality": round(mean_formality, 4),
        "mean_flesch": round(mean_flesch, 2),
        "avg_readability_flesch": round(mean_flesch, 2),
        "avg_sentence_length": round(mean_sentence_length, 2),
        "mean_vocab_richness": round(mean_vocab_richness, 4),
        "vocabulary_richness": round(mean_vocab_richness, 4),
        "top_keywords": top_keywords,
        "keyword_weights": [float(weight) for weight in keyword_weights],
        "sample_features": sample_features,
    }

    return {
        "designation": designation,
        "mission_core_vision": mission_core_vision,
        "snippets": snippets,
        "sample_texts": sample_texts,
        "sample_embeddings": sample_embeddings,
        "aggregate_embedding": aggregate_embedding.tolist(),
        "top_keywords": top_keywords,
        "tone_features": tone_features,
        "embedding_model": embedding_model_name,
        "embedding_dim": EMBEDDING_DIM,
    }


def initialize_user_genome(
    conn: sqlite3.Connection,
    designation: str,
    mission_core_vision: str,
    snippets: list[str],
) -> dict[str, Any]:
    ensure_canonical_schema(conn)
    cleaned_designation, cleaned_mission, cleaned_snippets = validate_user_genome_input(
        designation,
        mission_core_vision,
        snippets,
    )

    bundle = _build_feature_bundle(cleaned_designation, cleaned_mission, cleaned_snippets)
    now = _utc_now()

    cur = conn.cursor()
    existing_row = cur.execute(
        "SELECT id, metadata_json, created_at FROM brand_profile WHERE brand_id = ? ORDER BY id DESC LIMIT 1",
        (USER_BRAND_DB_ID,),
    ).fetchone()

    genome_version = 1
    created_at = now
    if existing_row:
        created_at = existing_row["created_at"] or now
        try:
            metadata = json.loads(existing_row["metadata_json"] or "{}")
            genome_version = int(metadata.get("genome_version", 1)) + 1
        except (TypeError, ValueError, json.JSONDecodeError):
            genome_version = 2

    metadata = {
        "brand_id": USER_BRAND_ALIAS,
        "brand_db_id": USER_BRAND_DB_ID,
        "designation": cleaned_designation,
        "mission_core_vision": cleaned_mission,
        "snippets": cleaned_snippets,
        "sample_texts": bundle["sample_texts"],
        "sample_embeddings": bundle["sample_embeddings"],
        "embedding_model": bundle["embedding_model"],
        "embedding_dim": bundle["embedding_dim"],
        "genome_version": genome_version,
        "sample_count": len(bundle["sample_texts"]),
        "snippet_count": len(cleaned_snippets),
        "initialized_at": created_at,
        "updated_at": now,
    }

    tone_features = dict(bundle["tone_features"])
    tone_features["genome_version"] = genome_version

    with conn:
        cur.execute(
            "INSERT OR REPLACE INTO brands (id, designation, mission_core_vision, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (USER_BRAND_DB_ID, cleaned_designation, cleaned_mission, created_at, now),
        )
        cur.execute("DELETE FROM brand_profile WHERE brand_id = ?", (USER_BRAND_DB_ID,))
        cur.execute(
            """
            INSERT INTO brand_profile (
                brand_id,
                keywords_json,
                tone_features_json,
                aggregate_embedding,
                metadata_json,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                USER_BRAND_DB_ID,
                json.dumps(bundle["top_keywords"]),
                json.dumps(tone_features),
                json.dumps(bundle["aggregate_embedding"]),
                json.dumps(metadata),
                created_at,
                now,
            ),
        )

    return get_user_genome_summary(conn)


def _json_field(value: Any | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value)


def write_history_event(
    conn: sqlite3.Connection,
    *,
    brand_id: int,
    event_type: str,
    input_text: str | None = None,
    pre_score: Any | None = None,
    post_score: Any | None = None,
    diagnostics_json: Any | None = None,
    extra_json: Any | None = None,
    created_at: str | None = None,
) -> None:
    ensure_canonical_schema(conn)
    created_at = created_at or _utc_now()
    canonical_pre_score = _json_field(pre_score)
    canonical_post_score = _json_field(post_score)
    canonical_diagnostics = _json_field(diagnostics_json)
    canonical_extra = _json_field(extra_json)

    legacy_result = {
        "event_type": event_type,
        "input_text": input_text,
        "pre_score": pre_score,
        "post_score": post_score,
        "diagnostics_json": diagnostics_json,
        "extra_json": extra_json,
    }

    with conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO analysis_history (
                brand_id,
                event_type,
                input_text,
                pre_score,
                post_score,
                diagnostics_json,
                extra_json,
                created_at,
                analysis_type,
                result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                brand_id,
                event_type,
                input_text,
                canonical_pre_score,
                canonical_post_score,
                canonical_diagnostics,
                canonical_extra,
                created_at,
                event_type,
                json.dumps(legacy_result),
            ),
        )


def _load_user_profile_row(conn: sqlite3.Connection) -> sqlite3.Row | None:
    cur = conn.cursor()
    return cur.execute(
        "SELECT * FROM brand_profile WHERE brand_id = ? ORDER BY id DESC LIMIT 1",
        (USER_BRAND_DB_ID,),
    ).fetchone()


def get_user_genome_summary(conn: sqlite3.Connection) -> dict[str, Any]:
    ensure_canonical_schema(conn)
    row = _load_user_profile_row(conn)
    if row is None:
        return {
            "brand_id": USER_BRAND_ALIAS,
            "brand_db_id": USER_BRAND_DB_ID,
            "designation": "",
            "brand_name": "",
            "name": "",
            "mission_core_vision": "",
            "mission": "",
            "snippets": ["", "", "", "", "", "", ""],
            "snippetsCount": 0,
            "snippet_count": 0,
            "keywords": [],
            "top_keywords": [],
            "tone": "",
            "tone_label": "",
            "tone_features": {},
            "feature_profile": {},
            "feature_ready": False,
            "embedding_ready": False,
            "aggregate_embedding_dim": 0,
            "profile_version": 0,
            "initialized": False,
            "updated_at": None,
        }

    metadata = json.loads(row["metadata_json"] or "{}")
    tone_features = json.loads(row["tone_features_json"] or "{}")
    keywords = json.loads(row["keywords_json"] or "[]")
    snippets = list(metadata.get("snippets", []))
    designation = metadata.get("designation") or metadata.get("brand_name") or ""
    mission = metadata.get("mission_core_vision") or metadata.get("mission") or ""
    profile_version = int(metadata.get("genome_version", 1))

    summary = {
        "brand_id": USER_BRAND_ALIAS,
        "brand_db_id": USER_BRAND_DB_ID,
        "designation": designation,
        "brand_name": designation,
        "name": designation,
        "mission_core_vision": mission,
        "mission": mission,
        "snippets": snippets,
        "snippetsCount": len(snippets),
        "snippet_count": len(snippets),
        "keywords": keywords,
        "top_keywords": tone_features.get("top_keywords", keywords),
        "tone": tone_features.get("tone_label", ""),
        "tone_label": tone_features.get("tone_label", ""),
        "tone_features": tone_features,
        "feature_profile": tone_features,
        "feature_ready": bool(tone_features),
        "embedding_ready": bool(row["aggregate_embedding"]),
        "aggregate_embedding_dim": len(json.loads(row["aggregate_embedding"] or "[]")),
        "profile_version": profile_version,
        "initialized": True,
        "updated_at": row["updated_at"],
    }
    return summary


def load_active_user_genome(conn: sqlite3.Connection) -> dict[str, Any] | None:
    """Return the persisted active user genome, or ``None`` if absent/invalid."""
    ensure_canonical_schema(conn)
    row = _load_user_profile_row(conn)
    if row is None:
        return None

    metadata = json.loads(row["metadata_json"] or "{}")
    tone_features = json.loads(row["tone_features_json"] or "{}")
    keywords = json.loads(row["keywords_json"] or "[]")
    aggregate_embedding = json.loads(row["aggregate_embedding"] or "[]")
    snippets = list(metadata.get("snippets", []))
    designation = metadata.get("designation") or metadata.get("brand_name") or ""
    mission_core_vision = metadata.get("mission_core_vision") or metadata.get("mission") or ""

    return {
        "brand_id": USER_BRAND_ALIAS,
        "brand_db_id": USER_BRAND_DB_ID,
        "designation": designation,
        "brand_name": designation,
        "name": designation,
        "mission_core_vision": mission_core_vision,
        "mission": mission_core_vision,
        "snippets": snippets,
        "snippetsCount": len(snippets),
        "snippet_count": len(snippets),
        "keywords": keywords,
        "top_keywords": tone_features.get("top_keywords", keywords),
        "tone": tone_features.get("tone_label", ""),
        "tone_label": tone_features.get("tone_label", ""),
        "tone_features": tone_features,
        "feature_profile": tone_features,
        "aggregate_embedding": aggregate_embedding,
        "aggregate_embedding_dim": len(aggregate_embedding),
        "embedding_ready": bool(aggregate_embedding),
        "feature_ready": bool(tone_features),
        "profile_version": int(metadata.get("genome_version", 1)),
        "genome_version": int(metadata.get("genome_version", 1)),
        "initialized": True,
        "updated_at": row["updated_at"],
        "metadata": metadata,
    }


def is_user_genome_initialized(conn: sqlite3.Connection) -> bool:
    genome = load_active_user_genome(conn)
    if not genome or not genome.get("initialized"):
        return False

    snippets = genome.get("snippets", [])
    if len(snippets) != SNIPPET_COUNT:
        return False
    if not genome.get("designation", "").strip():
        return False
    if not genome.get("mission_core_vision", "").strip():
        return False

    return genome.get("aggregate_embedding_dim") == EMBEDDING_DIM


def normalise_brand_id(brand_id: Any) -> str:
    if brand_id is None:
        return ""
    return str(brand_id).strip().lower()


def is_user_brand_identifier(brand_id: Any) -> bool:
    normalized = normalise_brand_id(brand_id)
    return normalized in {USER_BRAND_ALIAS, str(USER_BRAND_DB_ID)}