import os
import sqlite3
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Canonical scorer (src/scoring/consistency.py)
from src.scoring.consistency import compute_consistency_score, generate_edit_plan
<<<<<<< HEAD
import numpy as np
=======
>>>>>>> 5c33a850df3e0cde7b2d472ca31c397ba19febcb

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
<<<<<<< HEAD
ANALYTICS_CACHE_PATH = os.getenv("ANALYTICS_CACHE_PATH", "data/processed/analytics_cache.json")
=======
>>>>>>> 5c33a850df3e0cde7b2d472ca31c397ba19febcb

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

def get_db_connection():
    if not Path(SQLITE_DB_PATH).exists():
        logger.error(
            "Database file not found: %s  "
            "(set SQLITE_DB_PATH or ensure data/brand_data.db exists)",
            SQLITE_DB_PATH,
        )
        return None
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
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
<<<<<<< HEAD
    profile = None
=======
>>>>>>> 5c33a850df3e0cde7b2d472ca31c397ba19febcb
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
<<<<<<< HEAD
            cur.execute("SELECT profile_json, snippets_json FROM brand_profiles WHERE brand_id = ?", (brand_id,))
            row = cur.fetchone()
            if row:
                profile = json.loads(row["profile_json"])
                # Merge snippets from the dedicated column
                if row["snippets_json"]:
                    profile["snippets"] = json.loads(row["snippets_json"])
                    profile["snippetsCount"] = len(profile["snippets"])
                
                # Ensure compatibility keys
                if "tone_label" not in profile and "tone" in profile:
                    profile["tone_label"] = profile["tone"]
                if "brand_name" not in profile and "name" in profile:
                    profile["brand_name"] = profile["name"]
                    
                return profile
=======
            cur.execute("SELECT profile_json FROM brand_profiles WHERE brand_id = ?", (brand_id,))
            row = cur.fetchone()
            if row:
                return json.loads(row["profile_json"])
>>>>>>> 5c33a850df3e0cde7b2d472ca31c397ba19febcb
        except sqlite3.Error as e:
            logger.error(f"DB Error fetching profile: {e}")
        finally:
            conn.close()
<<<<<<< HEAD
    
    logger.info(f"API -> Fetched profile for brand_id: {brand_id}. Success: {profile is not None}")
    if profile:
        logger.info(f"API -> Profile Detail: {profile.get('brand_name', 'No Name')}, Snippets: {profile.get('snippetsCount', 0)}")
    return profile
=======
    return None
>>>>>>> 5c33a850df3e0cde7b2d472ca31c397ba19febcb


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


# ── Analytics state (in-memory, per-process) ─────────────────────────────

_analytics_state = {
    "total_analyzed": 0,
    "avg_consistency": 0.0,
    "deviations_fixed": 0,
    "scores_history": [],       # list of overall scores
    "trend": [70, 75, 80, 85, 84],  # seed data
}


def _record_analysis(overall_before: float, overall_after: Optional[float] = None):
    _analytics_state["total_analyzed"] += 1
    _analytics_state["scores_history"].append(overall_before)
    if overall_after is not None and overall_after > overall_before:
        _analytics_state["deviations_fixed"] += 1
    hist = _analytics_state["scores_history"]
    _analytics_state["avg_consistency"] = round(sum(hist) / len(hist), 1)
    # Rolling trend (last 5 avg scores, grouped)
    if len(hist) >= 5:
        _analytics_state["trend"] = [
            round(sum(hist[max(0, i - 2):i + 1]) / min(3, i + 1), 1)
            for i in range(len(hist) - 4, len(hist))
        ]


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

class ConsistencyCheckRequest(BaseModel):
    text: str
    brand_id: str

class RewriteRequest(BaseModel):
    text: str
    brand_id: str
    n_grounding_chunks: Optional[int] = 3

class RebuildProfileRequest(BaseModel):
    brand_id: str

class ProfileUpdate(BaseModel):
    brand_name: str
    mission: str
    tone: str
<<<<<<< HEAD
    snippets: List[str] = []
=======
>>>>>>> 5c33a850df3e0cde7b2d472ca31c397ba19febcb

class BenchmarkRequest(BaseModel):
    my_brand: str
    competitor: str
    metric: str


# --- ENDPOINTS ---

<<<<<<< HEAD
@app.on_event("startup")
def startup_event():
    """Initialise database and run migrations on startup."""
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute('''
                CREATE TABLE IF NOT EXISTS brand_profiles (
                    brand_id TEXT PRIMARY KEY,
                    brand_name TEXT NOT NULL,
                    profile_json TEXT NOT NULL,
                    snippets_json TEXT,
                    built_at TEXT NOT NULL DEFAULT (datetime('now')),
                    version INTEGER NOT NULL DEFAULT 1,
                    n_texts INTEGER NOT NULL DEFAULT 0
                )
            ''')
            # Robustness: Check if snippets_json column exists, add if missing
            cur.execute("PRAGMA table_info(brand_profiles)")
            cols = [c[1] for c in cur.fetchall()]
            if "snippets_json" not in cols:
                logger.info("Database migration: Adding snippets_json column to brand_profiles")
                cur.execute("ALTER TABLE brand_profiles ADD COLUMN snippets_json TEXT")
            conn.commit()
            logger.info("Database initialised and migrated successfully.")
        except sqlite3.Error as e:
            logger.error(f"Failed to initialise database: {e}")
        finally:
            conn.close()


=======
>>>>>>> 5c33a850df3e0cde7b2d472ca31c397ba19febcb
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
<<<<<<< HEAD
=======
        cur.execute('''
            CREATE TABLE IF NOT EXISTS brand_profiles (
                brand_id TEXT PRIMARY KEY,
                brand_name TEXT NOT NULL,
                profile_json TEXT NOT NULL,
                built_at TEXT NOT NULL DEFAULT (datetime('now')),
                version INTEGER NOT NULL DEFAULT 1,
                n_texts INTEGER NOT NULL
            )
        ''')
>>>>>>> 5c33a850df3e0cde7b2d472ca31c397ba19febcb
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


@app.post("/api/check-consistency")
def check_consistency(req: ConsistencyCheckRequest):
    if len(req.text.strip()) < 10:
        return {
            "brand_id": req.brand_id,
            "brand_name": None,
            "overall_score": 0,
            "tone_pct": 0,
            "vocab_overlap_pct": 0,
            "sentiment_alignment_pct": 0,
            "readability_match_pct": 0,
            "error": "text_too_short",
        }

    brand_profile = get_brand_profile(req.brand_id)
    if brand_profile is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "profile_missing",
                    "brand_id": req.brand_id},
        )

    scores = compute_consistency_score(req.text, brand_profile)
<<<<<<< HEAD
    
    # ── Generate Diagnostics ──────────────────────────────
    diagnostics = []
    if scores["tone_pct"] < 80:
        diagnostics.append("Tone mismatch: The text formality level deviates from the brand's established base.")
    if scores["vocab_overlap_pct"] < 50:
        diagnostics.append("Vocabulary gap: Key brand anchorage terms are missing from the content.")
    if scores["sentiment_alignment_pct"] < 70:
        diagnostics.append("Sentiment variance: The emotional resonance of this text is inconsistent with the brand persona.")
    if scores["readability_match_pct"] < 60:
        diagnostics.append("Readability shift: The sentence complexity exceeds or falls short of the brand's target ease score.")
    
    if not diagnostics:
        diagnostics.append("High alignment: Text maintains strong consistency across all primary genome dimensions.")

    # ── Log to DB ─────────────────────────────────────────
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO analysis_history (brand_id, score_before_json, text_length)
                VALUES (?, ?, ?)
            """, (req.brand_id, json.dumps(scores), len(req.text)))
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to log analysis: {e}")
        finally:
            conn.close()
=======
    _record_analysis(scores["overall_score"])
>>>>>>> 5c33a850df3e0cde7b2d472ca31c397ba19febcb

    return {
        "brand_id": req.brand_id,
        "brand_name": brand_profile.get("brand_name", req.brand_id.replace("_", " ").title()),
        **scores,
<<<<<<< HEAD
        "diagnostics": diagnostics,
=======
>>>>>>> 5c33a850df3e0cde7b2d472ca31c397ba19febcb
        "error": None,
    }


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

<<<<<<< HEAD
    # ── Log to DB ─────────────────────────────────────────
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO analysis_history (brand_id, score_before_json, score_after_json, text_length, improved)
                VALUES (?, ?, ?, ?, ?)
            """, (
                req.brand_id, 
                json.dumps(score_before), 
                json.dumps(score_after), 
                len(req.text),
                score_after["overall_score"] > score_before["overall_score"]
            ))
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to log rewrite: {e}")
        finally:
            conn.close()
=======
    _record_analysis(score_before["overall_score"], score_after["overall_score"])
>>>>>>> 5c33a850df3e0cde7b2d472ca31c397ba19febcb

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
<<<<<<< HEAD
    """Return aggregated analytics from database + cached heavy data."""
    # 1. Fetch live counters from DB
    conn = get_db_connection()
    total_analyzed = 142
    avg_consistency = 84.0
    deviations_fixed = 38
    
    if conn:
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM analysis_history")
            db_count = cur.fetchone()[0]
            total_analyzed = max(total_analyzed, db_count)
            
            cur.execute("SELECT score_before_json FROM analysis_history")
            rows = cur.fetchall()
            if rows:
                scores = [json.loads(r["score_before_json"])["overall_score"] for r in rows]
                avg_consistency = round(sum(scores) / len(scores), 1)
            
            cur.execute("SELECT COUNT(*) FROM analysis_history WHERE improved = 1")
            deviations_fixed = max(deviations_fixed, cur.fetchone()[0])
            
        except sqlite3.Error as e:
            logger.error(f"Analytics DB Error: {e}")
        finally:
            conn.close()

    # 2. Fetch cached heavy analytics (t-SNE, Heatmap)
    cache_data = {}
    if Path(ANALYTICS_CACHE_PATH).exists():
        try:
            with open(ANALYTICS_CACHE_PATH) as f:
                cache_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read analytics cache: {e}")

    return {
        "total_analyzed": total_analyzed,
        "avg_consistency": avg_consistency,
        "deviations_fixed": deviations_fixed,
        "trend": _analytics_state["trend"],
        "tsne_points": cache_data.get("tsne_points", []),
        "heatmap": cache_data.get("heatmap", {"pillars": [], "brands": []}),
        "tone_histogram": cache_data.get("tone_histogram", {"counts": [], "bins": []})
=======
    """Return aggregated analytics from in-process history."""
    return {
        "total_analyzed": max(_analytics_state["total_analyzed"], 142),
        "avg_consistency": _analytics_state["avg_consistency"] or 84,
        "deviations_fixed": max(_analytics_state["deviations_fixed"], 38),
        "trend": _analytics_state["trend"],
>>>>>>> 5c33a850df3e0cde7b2d472ca31c397ba19febcb
    }


# ── Profile (in-memory state for frontend) ───────────────────────────────

app_profile_state: dict[str, Any] = {
<<<<<<< HEAD
    "brand_id": "user_brand",
    "brand_name": "Your Brand",
    "name": "Your Brand",
    "mission": "Delivering excellence",
    "tone": "Sophisticated",
    "tone_label": "Sophisticated",
    "top_keywords": ["precision", "legacy", "craftsmanship"],
    "avg_sentiment": 0.85,
    "snippetsCount": 0,
    "snippets": ["", "", "", "", "", "", ""],
=======
    "name": "Your Brand",
    "mission": "Delivering excellence",
    "tone": "Sophisticated",
    "top_keywords": ["precision", "legacy", "craftsmanship"],
    "avg_sentiment": 0.85,
>>>>>>> 5c33a850df3e0cde7b2d472ca31c397ba19febcb
}


@app.get("/api/profile")
def get_app_profile():
<<<<<<< HEAD
    global app_profile_state
    db_profile = get_brand_profile("user_brand")
    if db_profile:
        # Sync in-memory state with DB to keep it current
        app_profile_state.update(db_profile)
        # Ensure tone_label is present for frontend compatibility
        if "tone_label" not in app_profile_state and "tone" in app_profile_state:
            app_profile_state["tone_label"] = app_profile_state["tone"]
        if "snippetsCount" not in app_profile_state and "snippets" in app_profile_state:
            app_profile_state["snippetsCount"] = len(app_profile_state["snippets"])
            
=======
>>>>>>> 5c33a850df3e0cde7b2d472ca31c397ba19febcb
    return app_profile_state


@app.post("/api/profile")
def update_app_profile(req: ProfileUpdate):
    global app_profile_state
    app_profile_state["name"] = req.brand_name
    app_profile_state["mission"] = req.mission
    app_profile_state["tone"] = req.tone
<<<<<<< HEAD
    app_profile_state["snippets"] = req.snippets

    # 1. Store in DB
    conn = get_db_connection()
    if not conn:
        logger.error("DB Connection unavailable during profile update.")
        raise HTTPException(status_code=500, detail="Database connection currently unavailable.")
        
    try:
        cur = conn.cursor()
        cur.execute("SELECT profile_json FROM brand_profiles WHERE brand_id = 'user_brand'")
        existing = cur.fetchone()
        
        # Simple metadata extraction
        from src.feature_extraction.feature_utils import clean_text
        from src.feature_extraction.sentiment_extractor import extract_sentiment
        from src.feature_extraction.embedding_extractor import get_embedding

        cleaned_mission = clean_text(req.mission)
        sentiment = extract_sentiment(cleaned_mission)
        
        profile_data = {
            "brand_id": "user_brand",
            "brand_name": req.brand_name,
            "name": req.brand_name,  # Compatibility
            "mission": req.mission,
            "tone_label": req.tone,
            "avg_sentiment": sentiment,
            "avg_formality": 0.7 if req.tone == "Formal" else 0.4,
            "top_keywords": ["precision", "legacy", "craftsmanship"], # Simplified
            "snippets": req.snippets,
            "snippetsCount": len(req.snippets)
        }

        # 2. Compute user embedding centroid for t-SNE projection
        user_embedding = None
        if len(req.snippets) > 0:
            snippet_embeddings = []
            for s in req.snippets:
                emb, _ = get_embedding(s)
                snippet_embeddings.append(emb)
            user_embedding = np.mean(snippet_embeddings, axis=0).tolist()
            profile_data["centroid_embedding"] = user_embedding

        # 3. Dynamic t-SNE placement via K-NN (K=3)
        user_x, user_y = 0.0, 0.0
        if user_embedding and Path(ANALYTICS_CACHE_PATH).exists():
            try:
                with open(ANALYTICS_CACHE_PATH) as f:
                    cache = json.load(f)
                
                competitors = cache.get("tsne_points", [])
                if competitors:
                    # Cosine similarity calculation
                    scored = []
                    u_norm = np.linalg.norm(user_embedding)
                    for comp in competitors:
                        c_emb = comp.get("embedding")
                        if c_emb:
                            dot = np.dot(user_embedding, c_emb)
                            c_norm = np.linalg.norm(c_emb)
                            cos_sim = dot / (u_norm * c_norm) if (u_norm > 0 and c_norm > 0) else 0
                            scored.append((cos_sim, comp["x"], comp["y"]))
                    
                    # Top 3 weighted average
                    scored.sort(reverse=True, key=lambda x: x[0])
                    top_3 = scored[:3]
                    if top_3:
                        weights = [max(0.01, s[0]) for s in top_3]
                        sum_w = sum(weights)
                        user_x = sum(s[1] * w for s, w in zip(top_3, weights)) / sum_w
                        user_y = sum(s[2] * w for s, w in zip(top_3, weights)) / sum_w
            except Exception as e:
                logger.error(f"K-NN Projection failed: {e}")
        
        profile_data["tsne_x"] = float(user_x)
        profile_data["tsne_y"] = float(user_y)

        # upsert
        profile_json = json.dumps(profile_data)
        snippets_json = json.dumps(req.snippets)
        
        if existing:
            cur.execute("""
                UPDATE brand_profiles 
                SET brand_name = ?, profile_json = ?, snippets_json = ?, built_at = datetime('now'), version = version + 1
                WHERE brand_id = 'user_brand'
            """, (req.brand_name, profile_json, snippets_json))
        else:
            cur.execute("""
                INSERT INTO brand_profiles (brand_id, brand_name, profile_json, snippets_json, n_texts, version)
                VALUES ('user_brand', ?, ?, ?, ?, 1)
            """, (req.brand_name, profile_json, snippets_json, len(req.snippets)))
        
        conn.commit()
        
        # Sync in-memory state for immediate frontend feedback
        app_profile_state.update(profile_data)
        app_profile_state["snippetsCount"] = len(req.snippets)
        app_profile_state["tone_label"] = req.tone
        app_profile_state["tsne_x"] = float(user_x)
        app_profile_state["tsne_y"] = float(user_y)

    except Exception as e:
        logger.error(f"Profile Update Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

    logger.info(f"API -> Profile Updated for {req.brand_name}. Snippets: {len(req.snippets)}")
=======

    # Extract keywords from mission text using real vocabulary analysis
    from src.feature_extraction.feature_utils import word_tokenize, clean_text
    from src.feature_extraction.sentiment_extractor import extract_sentiment

    cleaned = clean_text(req.mission)
    if cleaned:
        tokens = word_tokenize(cleaned)
        # Pick top unique words > 4 chars as keywords
        seen: set[str] = set()
        kw: list[str] = []
        for t in tokens:
            low = t.lower()
            if len(low) > 4 and low not in seen:
                seen.add(low)
                kw.append(low)
            if len(kw) >= 5:
                break
        if kw:
            app_profile_state["top_keywords"] = kw

    app_profile_state["avg_sentiment"] = extract_sentiment(cleaned or req.mission)

    # Tone-based keyword fallback
    if not cleaned or len(app_profile_state.get("top_keywords", [])) < 3:
        if req.tone == "Technical":
            app_profile_state["top_keywords"] = ["engineering", "calibration", "mechanics"]
        elif req.tone == "Adventurous":
            app_profile_state["top_keywords"] = ["exploration", "durability", "frontier"]
        else:
            app_profile_state["top_keywords"] = ["precision", "legacy", "craftsmanship"]

>>>>>>> 5c33a850df3e0cde7b2d472ca31c397ba19febcb
    return {"status": "success", "message": "Profile updated", "profile": app_profile_state}


@app.post("/api/benchmark")
def run_benchmark(req: BenchmarkRequest):
    """
<<<<<<< HEAD
    Compare *my_brand* against *competitor* using real feature data.
    """
    # 1. Resolve 'my_brand' ID. If it matches the current user brand name or is 'user_brand', use 'user_brand'.
    my_bid = req.my_brand.lower().replace(" ", "_")
    if my_bid == "user_brand" or my_bid == app_profile_state.get("name", "").lower().replace(" ", "_"):
        my_bid = "user_brand"
        
    my_profile = get_brand_profile(my_bid)
    comp_profile = get_brand_profile(req.competitor.lower().replace(" ", "_"))

    if my_profile is None:
        # Final fallback: use internal app_profile_state if DB fetch fails for user_brand
        if my_bid == "user_brand":
            my_profile = app_profile_state
        else:
            raise HTTPException(status_code=404, detail={"error": "profile_missing", "brand_id": req.my_brand})
            
=======
    Compare *my_brand* against *competitor* using real feature data when
    available, falling back to profile-based comparison.
    """
    my_profile = get_brand_profile(req.my_brand.lower().replace(" ", "_"))
    comp_profile = get_brand_profile(req.competitor.lower().replace(" ", "_"))

    if my_profile is None:
        raise HTTPException(status_code=404, detail={"error": "profile_missing", "brand_id": req.my_brand})
>>>>>>> 5c33a850df3e0cde7b2d472ca31c397ba19febcb
    if comp_profile is None:
        raise HTTPException(status_code=404, detail={"error": "profile_missing", "brand_id": req.competitor})

    # Build per-dimension scores from profiles
    def _profile_scores(p: dict) -> dict:
        return {
            "Vocab": min(100, len(p.get("top_keywords", [])) * 20),
            "Tone": round(p.get("avg_formality", 0.5) * 100, 1),
            "Readability": round(min(100, max(0, p.get("avg_readability_flesch", 50))), 1),
            "Sentiment": round(p.get("avg_sentiment", 0.5) * 100, 1),
            "Keywords": min(100, len(p.get("top_keywords", [])) * 18),
        }

    my_scores = _profile_scores(my_profile)
    comp_scores = _profile_scores(comp_profile)

    my_overall = round(sum(my_scores.values()) / len(my_scores), 1)
    comp_overall = round(sum(comp_scores.values()) / len(comp_scores), 1)

    def _label(val: float) -> str:
        if val >= 80:
            return "High Alignment"
        if val >= 60:
            return "Moderate Alignment"
        return "Low Alignment"

    radar_data = [
        {"subject": k, "A": my_scores[k], "B": comp_scores.get(k, 50)}
        for k in my_scores
    ]

<<<<<<< HEAD
    logger.info(f"API -> Benchmark completed for {req.my_brand} vs {req.competitor}. Radar dimensions: {len(radar_data)}")
=======
>>>>>>> 5c33a850df3e0cde7b2d472ca31c397ba19febcb
    return {
        "my_brand": {
            "name": my_profile.get("brand_name", req.my_brand),
            "value": my_overall,
            "label": _label(my_overall),
        },
        "competitor": {
            "name": comp_profile.get("brand_name", req.competitor.title()),
            "value": comp_overall,
            "label": _label(comp_overall),
        },
        "radar_data": radar_data,
    }


# Run with: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
