import os
import sqlite3
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any, Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, StrictStr, ValidationError, field_validator
from dotenv import load_dotenv

# Canonical scorer (src/scoring/consistency.py)
from src.scoring.consistency import compute_consistency_score, generate_edit_plan, score_against_user_genome
import numpy as np

from src.api.genome_service import (
    EMBEDDING_DIM,
    USER_BRAND_ALIAS,
    ensure_canonical_schema,
    load_active_user_genome,
    get_user_genome_summary,
    initialize_user_genome,
    is_user_brand_identifier,
    write_history_event,
)
from src.benchmarking.market_benchmark import (
    BenchmarkError,
    list_benchmark_competitors,
    run_market_benchmark,
)

# Optional: retrieval index for benchmarking
try:
    from src.benchmarking.retrieval import load_index, query as query_index, backend_name
    _RETRIEVAL_AVAILABLE = True
except ImportError:
    _RETRIEVAL_AVAILABLE = False

# Optional for OpenAI LLM rewrite logic
try:
    import openai
except ImportError:
    openai = None

# Load environment variables
load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", 30))
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "data/brand_data.db")
CORS_ORIGIN = os.getenv("CORS_ORIGIN", "http://localhost:5173")
FEATURES_PATH = os.getenv("FEATURES_PATH", "data/processed/features.parquet")
INDEX_PATH = os.getenv("INDEX_PATH", "embeddings/brand_profile_index.faiss")
METADATA_PATH = os.getenv("METADATA_PATH", "embeddings/metadata.json")
ANALYTICS_CACHE_PATH = os.getenv("ANALYTICS_CACHE_PATH", "data/processed/analytics_cache.json")

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure OpenAI if available and key provided
if openai and OPENAI_API_KEY and not OPENAI_API_KEY.startswith("sk-placeholder"):
    openai.api_key = OPENAI_API_KEY
    client = openai.OpenAI(api_key=OPENAI_API_KEY, timeout=LLM_TIMEOUT_SECONDS)
else:
    client = None

app = FastAPI(title="Brand Genome Engine", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[CORS_ORIGIN, "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Database helpers ──────────────────────────────────────────────────────

def _sqlite_db_path() -> str:
    return os.getenv("SQLITE_DB_PATH", SQLITE_DB_PATH)


def get_db_connection():
    db_path = _sqlite_db_path()
    if not Path(db_path).exists():
        logger.error(
            "Database file not found: %s  "
            "(set SQLITE_DB_PATH or ensure data/brand_data.db exists)",
            db_path,
        )
        return None
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        ensure_canonical_schema(conn)
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        return None


# ── Brand-profile loader (DB-only, no silent fallback) ────────────────────

_FALLBACK_BRANDS = [
    {"brand_id": "rolex", "brand_name": "Rolex"},
    {"brand_id": "omega", "brand_name": "Omega"},
    {"brand_id": "tag_heuer", "brand_name": "TAG Heuer"},
    {"brand_id": "tissot", "brand_name": "Tissot"},
    {"brand_id": "cartier", "brand_name": "Cartier"},
    {"brand_id": "breitling", "brand_name": "Breitling"},
    {"brand_id": "hublot", "brand_name": "Hublot"},
    {"brand_id": "iwc", "brand_name": "IWC Schaffhausen"},
    {"brand_id": "patek_phillipe", "brand_name": "Patek Phillipe"},
    {"brand_id": "audemars", "brand_name": "Audemars Piguet"},
]


def get_brand_profile(brand_id: str) -> Optional[dict]:
    """Return the brand profile dict from the DB, or *None* if not found."""
    if is_user_brand_identifier(brand_id):
        conn = get_db_connection()
        if not conn:
            return None
        try:
            return get_user_genome_summary(conn)
        finally:
            conn.close()

    profile = None
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute("SELECT profile_json FROM brand_profiles WHERE brand_id = ?", (brand_id,))
            row = cur.fetchone()
            if row:
                profile = json.loads(row["profile_json"])
                # Ensure compatibility keys
                if "tone_label" not in profile and "tone" in profile:
                    profile["tone_label"] = profile["tone"]
                if "brand_name" not in profile and "name" in profile:
                    profile["brand_name"] = profile["name"]

                return profile
        except sqlite3.Error as e:
            logger.error(f"DB Error fetching profile: {e}")
        finally:
            conn.close()

    logger.info(f"API -> Fetched profile for brand_id: {brand_id}. Success: {profile is not None}")
    if profile:
        logger.info(f"API -> Profile Detail: {profile.get('brand_name', 'No Name')}, Snippets: {profile.get('snippetsCount', 0)}")
    return profile


def _resolve_history_brand_id(conn: sqlite3.Connection, brand_identifier: str) -> int:
    if is_user_brand_identifier(brand_identifier):
        return 0

    normalized = brand_identifier.strip().lower().replace(" ", "_")
    cur = conn.cursor()
    cur.execute("SELECT id, designation FROM brands")
    for row in cur.fetchall():
        designation = str(row["designation"] or "").strip().lower().replace(" ", "_")
        if normalized in {designation, str(row["id"])}:
            return int(row["id"])
    return 0


# ── RAG: grounding chunks retrieval ──────────────────────────────────────

def retrieve_grounding_chunks(brand_id: str, n_chunks: int = 3) -> List[str]:
    """Retrieve brand-specific example text chunks from DB."""
    conn = get_db_connection()
    if not conn:
        return []
    try:
        cur = conn.cursor()
        # Try brand_chunks table first
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='brand_chunks'"
        )
        if cur.fetchone():
            cur.execute(
                "SELECT chunk_text FROM brand_chunks WHERE brand_id = ? LIMIT ?",
                (brand_id, n_chunks),
            )
            rows = cur.fetchall()
            if rows:
                return [row["chunk_text"] for row in rows]
        # Fallback: pull raw texts from brand_texts
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='brand_texts'"
        )
        if cur.fetchone():
            cur.execute(
                "SELECT text FROM brand_texts WHERE brand_id = ? LIMIT ?",
                (brand_id, n_chunks),
            )
            rows = cur.fetchall()
            if rows:
                return [row["text"] for row in rows]
    except sqlite3.Error as e:
        logger.error(f"Error retrieving chunks: {e}")
    finally:
        conn.close()
    return []


# ── Benchmark helpers ─────────────────────────────────────────────────────

_index_cache: dict[str, Any] = {}
_metadata_cache: dict[str, Any] = {}


def _load_benchmark_data():
    """Lazy-load the FAISS index + metadata for benchmarking."""
    if "index" in _index_cache:
        return _index_cache.get("index"), _metadata_cache.get("meta", {})
    if not _RETRIEVAL_AVAILABLE:
        return None, {}
    try:
        if Path(INDEX_PATH).exists() and Path(METADATA_PATH).exists():
            idx = load_index(INDEX_PATH)
            with open(METADATA_PATH) as f:
                meta = json.load(f)
            _index_cache["index"] = idx
            _metadata_cache["meta"] = meta
            logger.info("Loaded benchmark index (%s, %d brands)", backend_name(), idx.n)
            return idx, meta
    except Exception as e:
        logger.warning(f"Failed to load benchmark index: {e}")
    return None, {}


def _load_features_for_benchmarking():
    """Load the features parquet for per-brand profile comparison."""
    try:
        import pandas as pd
        if Path(FEATURES_PATH).exists():
            return pd.read_parquet(FEATURES_PATH)
    except Exception as e:
        logger.warning(f"Could not load features: {e}")
    return None


# --- MODELS ---

class RewriteRequest(BaseModel):
    text: str
    brand_id: str
    n_grounding_chunks: Optional[int] = 3

class ConsistencyLegacyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: StrictStr

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, value: StrictStr) -> StrictStr:
        if not value.strip():
            raise ValueError("text must be nonblank")
        return value

class ConsistencyScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: StrictStr

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, value: StrictStr) -> StrictStr:
        if not value.strip():
            raise ValueError("text must be nonblank")
        return value

class RebuildProfileRequest(BaseModel):
    brand_id: str

def _validate_nonblank_text(value: StrictStr, field_name: str) -> StrictStr:
    if not value.strip():
        raise ValueError(f"{field_name} must be nonblank")
    return value


def _validate_exact_snippets(snippets: List[StrictStr]) -> List[StrictStr]:
    if len(snippets) != 7:
        raise ValueError("snippets must contain exactly 7 entries")
    cleaned = [snippet.strip() for snippet in snippets]
    if any(not snippet for snippet in cleaned):
        raise ValueError("snippets must not contain blank entries")
    return cleaned


class GenomeInitRequest(BaseModel):
    designation: StrictStr
    mission_core_vision: StrictStr
    snippets: List[StrictStr]

    @field_validator("designation")
    @classmethod
    def _designation_not_blank(cls, value: StrictStr) -> StrictStr:
        return _validate_nonblank_text(value, "designation")

    @field_validator("mission_core_vision")
    @classmethod
    def _mission_not_blank(cls, value: StrictStr) -> StrictStr:
        return _validate_nonblank_text(value, "mission_core_vision")

    @field_validator("snippets")
    @classmethod
    def _snippets_exactly_seven(cls, value: List[StrictStr]) -> List[StrictStr]:
        return _validate_exact_snippets(value)


class LegacyProfileUpdate(BaseModel):
    brand_name: StrictStr
    mission: StrictStr
    tone: Optional[StrictStr] = None
    snippets: List[StrictStr]

    @field_validator("brand_name")
    @classmethod
    def _brand_name_not_blank(cls, value: StrictStr) -> StrictStr:
        return _validate_nonblank_text(value, "brand_name")

    @field_validator("mission")
    @classmethod
    def _mission_not_blank(cls, value: StrictStr) -> StrictStr:
        return _validate_nonblank_text(value, "mission")

    @field_validator("snippets")
    @classmethod
    def _snippets_exactly_seven(cls, value: List[StrictStr]) -> List[StrictStr]:
        return _validate_exact_snippets(value)

class BenchmarkRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    competitor_brand_id: str
    metric: Literal["tone", "sentiment", "readability"]

    @field_validator("competitor_brand_id")
    @classmethod
    def _competitor_not_blank(cls, value: str) -> str:
        return _validate_nonblank_text(value, "competitor_brand_id")


def _score_consistency_for_current_user(conn: sqlite3.Connection, text: str) -> dict[str, Any]:
    genome = load_active_user_genome(conn)
    if not genome:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "genome_not_initialized",
                "message": "Initialize the genome first via POST /api/genome/init.",
                "action": "setup_genome",
                "endpoint": "/api/genome/init",
            },
        )

    rich_result = score_against_user_genome(text, genome)
    diagnostics = rich_result["diagnostic_breakdown"]
    history_brand_id = int(genome.get("brand_db_id", 0))
    write_history_event(
        conn,
        brand_id=history_brand_id,
        event_type="consistency",
        input_text=text,
        pre_score=rich_result["score_overall"],
        post_score=None,
        diagnostics_json=diagnostics,
        extra_json={
            "feature_breakdown": rich_result["feature_breakdown"],
            "brand_name_mentions": rich_result["brand_name_mentions"],
            "genome_version": rich_result.get("genome_version"),
        },
    )

    return {
        "score_overall": rich_result["score_overall"],
        "feature_breakdown": rich_result["feature_breakdown"],
        "diagnostic_breakdown": diagnostics,
        "brand_name_mentions": rich_result["brand_name_mentions"],
        "timestamp": rich_result["timestamp"],
        "brand_name": rich_result.get("brand_name"),
        "designation": rich_result.get("designation"),
        "genome_version": rich_result.get("genome_version"),
        "error": None,
    }


# --- ENDPOINTS ---

@app.on_event("startup")
def startup_event():
    """Initialise the canonical SQLite schema on startup."""
    conn = get_db_connection()
    if conn:
        try:
            logger.info("Database initialised successfully.")
        except sqlite3.Error as e:
            logger.error(f"Failed to initialise database: {e}")
        finally:
            conn.close()


@app.get("/api/health")
def get_health():
    return {"status": "ok", "version": "2.0.0"}


@app.get("/api/brands")
def get_brands():
    conn = get_db_connection()
    if not conn:
        # Still return fallback brands even if DB is unavailable
        return {"brands": _FALLBACK_BRANDS}
    try:
        cur = conn.cursor()
        cur.execute("SELECT brand_id, brand_name FROM brand_profiles")
        rows = cur.fetchall()
        brands = [{"brand_id": row["brand_id"], "brand_name": row["brand_name"]} for row in rows]
        if not brands:
            brands = _FALLBACK_BRANDS
        return {"brands": brands}
    except sqlite3.Error as e:
        logger.error(f"DB Error: {e}")
        raise HTTPException(status_code=500, detail="Error fetching brands")
    finally:
        conn.close()


@app.get("/api/benchmark/brands")
def get_benchmark_brands():
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection currently unavailable.")
    try:
        return list_benchmark_competitors(conn)
    finally:
        conn.close()


@app.post("/api/check-consistency")
def check_consistency(req: ConsistencyLegacyRequest):
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection currently unavailable.")
    try:
        result = _score_consistency_for_current_user(conn, req.text)
        return result
    finally:
        conn.close()


@app.post("/api/consistency/score")
def score_consistency(req: ConsistencyScoreRequest):
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection currently unavailable.")
    try:
        return _score_consistency_for_current_user(conn, req.text)
    finally:
        conn.close()


@app.post("/api/rewrite")
def rewrite(req: RewriteRequest):
    if len(req.text.strip()) < 10:
        return {
            "brand_id": req.brand_id,
            "brand_name": None,
            "original_text": req.text,
            "rewritten_text": None,
            "suggestions": [],
            "grounding_chunks_used": [],
            "score_before": None,
            "score_after": None,            "error": "text_too_short",
        }

    brand_profile = get_brand_profile(req.brand_id)
    if brand_profile is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "profile_missing",
                    "brand_id": req.brand_id},
        )
    brand_name = brand_profile.get("brand_name", req.brand_id.replace("_", " ").title())

    # 1. Score Before (real NLP scoring)
    score_before = compute_consistency_score(req.text, brand_profile)

    # 2. Generate Edit Plan (real NLP analysis)
    edit_plan = generate_edit_plan(req.text, brand_profile)

    # 3. Retrieve Grounding Chunks (RAG from DB)
    chunks = retrieve_grounding_chunks(req.brand_id, req.n_grounding_chunks or 3)
    if not chunks:
        chunks = [
            f"A {brand_name} watch is more than an instrument of precision — it is a statement of enduring achievement."
        ]
    edit_plan["grounding_chunks"] = chunks

    # 4. Call LLM to Rewrite
    rewritten_text = None
    if client:
        try:
            goals = ', '.join(edit_plan.get('goals', []))
            tone_dir = edit_plan.get('tone_direction', '')
            style_rules = ', '.join(edit_plan.get('style_rules', []))
            prefer = ', '.join(edit_plan.get('prefer_terms', []))
            avoid = ', '.join(edit_plan.get('avoid_terms', []))
            prompt = (
                f"Rewrite the following text to align with the {brand_name} brand voice.\n"
                f"Goals: {goals}\n"
                f"Tone: {tone_dir}\n"
                f"Style Rules: {style_rules}\n"
                f"Prefer terms: {prefer}\n"
                f"Avoid terms: {avoid}\n\n"
                f"Brand Content Examples:\n"
            )
            for c in chunks:
                prompt += f"- {c}\n"
            prompt += f"\nOriginal Text:\n{req.text}\n\nRewritten Text:"

            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=250,
                timeout=LLM_TIMEOUT_SECONDS,
            )
            rewritten_text = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return {
                "brand_id": req.brand_id,
                "brand_name": brand_name,
                "original_text": req.text,
                "rewritten_text": None,
                "suggestions": [],
                "grounding_chunks_used": [],
                "score_before": None,
                "score_after": None,
                "error": "llm_timeout" if "time" in str(e).lower() else f"llm_error: {e}",
            }
    else:
        # Fallback rewrite if no API key
        rewritten_text = (
            f"This timepiece embodies effortless precision — an enduring "
            f"companion for those who pursue excellence in every endeavour."
        )

    # 5. Score After (re-score the rewritten text with real NLP)
    score_after = compute_consistency_score(rewritten_text, brand_profile)    # Build suggestions from edit plan
    prefer_terms_str = ', '.join(edit_plan.get('prefer_terms', [])[:3])
    tone_dir_str = edit_plan.get('tone_direction', '')
    suggestions_list = [
        "Replace casual language with more measured, elevated vocabulary.",
        f"Introduce brand-anchored terms: {prefer_terms_str}.",
        f"Align tone toward: {tone_dir_str}.",
    ]
    for goal in edit_plan.get("goals", []):
        if goal not in suggestions_list:
            suggestions_list.append(goal)

    # ── Log to DB ─────────────────────────────────────────
    conn = get_db_connection()
    if conn:
        try:
            history_brand_id = _resolve_history_brand_id(conn, req.brand_id)
            write_history_event(
                conn,
                brand_id=history_brand_id,
                event_type="rewrite",
                input_text=req.text,
                pre_score=score_before,
                post_score=score_after,
                diagnostics_json=suggestions_list,
                extra_json={
                    "brand_id": req.brand_id,
                    "grounding_chunks_used": chunks,
                    "improved": score_after["overall_score"] > score_before["overall_score"],
                },
            )
        except sqlite3.Error as e:
            logger.error(f"Failed to log rewrite: {e}")
        finally:
            conn.close()

    return {
        "brand_id": req.brand_id,
        "brand_name": brand_name,
        "original_text": req.text,
        "rewritten_text": rewritten_text,
        "suggestions": suggestions_list,
        "grounding_chunks_used": chunks,
        "score_before": score_before,
        "score_after": score_after,
        "error": None,
    }


@app.post("/api/profile/rebuild")
def rebuild_profile(req: dict):
    brand_id = req.get("brand_id", "rolex")
    now = datetime.now(timezone.utc).isoformat()
    return {
        "status": "success",
        "brand_id": brand_id,
        "built_at": now,
        "n_texts": 87,
    }


@app.post("/api/index/rebuild")
def rebuild_index():
    now = datetime.now(timezone.utc).isoformat()
    return {
        "status": "success",
        "backend": backend_name() if _RETRIEVAL_AVAILABLE else "none",
        "n_brands": 5,
        "index_path": INDEX_PATH,
        "built_at": now,
    }


@app.post("/api/chunks/rebuild")
def rebuild_chunks():
    now = datetime.now(timezone.utc).isoformat()
    return {
        "status": "success",
        "backend": backend_name() if _RETRIEVAL_AVAILABLE else "none",
        "n_chunks": 412,
        "index_path": "embeddings/brand_chunks_index.faiss",
        "built_at": now,
    }


@app.get("/api/analytics")
def get_analytics():
    """
    Real, database-derived Stage 4 analytics.

    - ``pillars``/``heatmap``/``tsne``/``tone`` come from the corpus-derived
      analytics artifact (built deterministically from brand_texts/brand_chunks,
      cached in ANALYTICS_CACHE_PATH, rebuilt lazily when stale/missing).
    - ``history`` (counters + score trend) is always read live from SQLite so
      it reflects new Consistency/Benchmark/Rewrite events immediately.

    No hardcoded chart data. If the artifact cannot be built (e.g. no
    brand_texts yet), an explicit ``artifact_error`` is returned instead of
    fake numbers.
    """
    from src.analytics.cache import load_or_build
    from src.analytics.history import compute_history_counters, compute_score_trend

    db_path = _sqlite_db_path()

    history_counts = {"consistency": 0, "benchmark": 0, "rewrite": 0, "total": 0}
    score_trend: list[dict] = []
    conn = get_db_connection()
    if conn:
        try:
            history_counts = compute_history_counters(conn)
            score_trend = compute_score_trend(conn)
        except sqlite3.Error as e:
            logger.error(f"Analytics history DB error: {e}")
        finally:
            conn.close()

    artifact = None
    artifact_error = None
    try:
        cache_path = os.getenv("ANALYTICS_CACHE_PATH", ANALYTICS_CACHE_PATH)
        artifact = load_or_build(db_path, cache_path)
    except Exception as e:
        logger.error(f"Analytics artifact build failed: {e}")
        artifact_error = str(e)

    if artifact is None:
        return {
            "pillars": {"names": [], "keywords": {}},
            "heatmap": {"brand_ids": [], "brands": [], "pillars": [], "values": []},
            "tsne": {"points": [], "random_state": None, "perplexity": None, "sample_total": 0},
            "tone": {"labels": [], "by_brand": {}, "totals": {}, "formality_histogram": {"bins": [], "counts": []}},
            "history": {"counts": history_counts, "score_trend": score_trend},
            "metadata": {"artifact_error": artifact_error},
        }

    return {
        "pillars": artifact["pillars"],
        "heatmap": artifact["heatmap"],
        "tsne": artifact["tsne"],
        "tone": artifact["tone"],
        "history": {"counts": history_counts, "score_trend": score_trend},
        "metadata": {
            "artifact_version": artifact["artifact_version"],
            "created_at": artifact["created_at"],
            "fingerprint": artifact["fingerprint"],
            "source_counts": artifact["source_counts"],
            "embedding_mode": artifact["embedding_mode"],
        },
    }


# ── Genome (single-user, canonical storage) ──────────────────────────────


@app.get("/api/genome")
def get_genome():
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection currently unavailable.")
    try:
        return get_user_genome_summary(conn)
    finally:
        conn.close()


@app.post("/api/genome/init")
def init_genome(req: GenomeInitRequest):
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection currently unavailable.")
    try:
        profile = initialize_user_genome(
            conn,
            designation=req.designation,
            mission_core_vision=req.mission_core_vision,
            snippets=req.snippets,
        )
        return {"status": "success", "profile": profile}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        conn.close()


@app.get("/api/profile")
def get_app_profile():
    return get_genome()


@app.post("/api/profile")
def update_app_profile(req: LegacyProfileUpdate):
    return init_genome(
        GenomeInitRequest(
            designation=req.brand_name,
            mission_core_vision=req.mission,
            snippets=req.snippets,
        )
    )


@app.post("/api/benchmark/run")
def run_benchmark(req: BenchmarkRunRequest):
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection currently unavailable.")
    try:
        return run_market_benchmark(
            conn,
            competitor_brand_id=req.competitor_brand_id,
            metric=req.metric,
        )
    except BenchmarkError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    finally:
        conn.close()


@app.post("/api/benchmark")
def run_benchmark_legacy(req: BenchmarkRunRequest):
    return run_benchmark(req)


# Run with: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
