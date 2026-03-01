import os
import sqlite3
import json
import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

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

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure OpenAI if available and key provided
if openai and OPENAI_API_KEY and not OPENAI_API_KEY.startswith("sk-placeholder"):
    openai.api_key = OPENAI_API_KEY
    client = openai.OpenAI(api_key=OPENAI_API_KEY, timeout=LLM_TIMEOUT_SECONDS)
else:
    client = None

app = FastAPI(title="Brand Genome Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[CORS_ORIGIN, "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db_connection():
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        return None

# --- Mocking Person C and Person B Functions (To be replaced in integration) ---

def mock_score_consistency(text: str, brand_profile: dict) -> dict:
    """Mocks Person C's score_consistency returning a ScoreResult"""
    # Simple deterministic mock scores based on length for demonstration
    length = len(text)
    return {
        "overall_score": min(100, length % 50 + 30),
        "tone_pct": min(100, length % 40 + 20),
        "vocab_overlap_pct": min(100, length % 30 + 10),
        "sentiment_alignment_pct": min(100, length % 60 + 40),
        "readability_match_pct": min(100, length % 50 + 25)
    }

def mock_generate_edit_plan(text: str, brand_profile: dict) -> dict:
    """Mocks Person C's generate_edit_plan returning an EditPlan"""
    return {
        "brand_id": brand_profile.get("brand_id", "unknown"),
        "goals": [
            "Increase formality",
            "Reduce reading ease",
            "Raise sentiment closer to brand mean"
        ],
        "avoid_terms": ["awesome", "cool", "super"],
        "prefer_terms": brand_profile.get("top_keywords", ["precision", "excellence"]),
        "style_rules": ["Use formal sentence structures", "Avoid contractions"],
        "tone_direction": brand_profile.get("tone_label", "authoritative"),
        "grounding_chunks": [] # Populated later
    }

def mock_retrieve_grounding_chunks(brand_id: str, n_chunks: int = 3) -> List[str]:
    """Mocks Person B's FAISS RAG retrieval from brand_chunks."""
    conn = get_db_connection()
    if not conn:
        return []
    try:
        cur = conn.cursor()
        cur.execute("SELECT chunk_text FROM brand_chunks WHERE brand_id = ? LIMIT ?", (brand_id, n_chunks))
        rows = cur.fetchall()
        return [row["chunk_text"] for row in rows]
    except sqlite3.Error as e:
        logger.error(f"Error retrieving chunks: {e}")
        return []
    finally:
        conn.close()


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

class BenchmarkRequest(BaseModel):
    my_brand: str
    competitor: str
    metric: str


# --- ENDPOINTS ---

@app.get("/api/health")
def get_health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/api/brands")
def get_brands():
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        cur = conn.cursor()
        # Create table if it doesn't exist to prevent crashes before Person C runs setup
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
        cur.execute("SELECT brand_id, brand_name FROM brand_profiles")
        rows = cur.fetchall()
        brands = [{"brand_id": row["brand_id"], "brand_name": row["brand_name"]} for row in rows]
        
        # Fallback if empty (e.g. before builder is run)
        if not brands:
             brands = [
                 {"brand_id": "rolex", "brand_name": "Rolex"},
                 {"brand_id": "omega", "brand_name": "Omega"},
                 {"brand_id": "tag_heuer", "brand_name": "TAG Heuer"},
                 {"brand_id": "tissot", "brand_name": "Tissot"},
                 {"brand_id": "titan", "brand_name": "Titan"}
             ]
        return {"brands": brands}
    except sqlite3.Error as e:
        logger.error(f"DB Error: {e}")
        raise HTTPException(status_code=500, detail="Error fetching brands")
    finally:
        conn.close()


def get_brand_profile(brand_id: str) -> dict:
    conn = get_db_connection()
    if not conn:
        return {}
    try:
        cur = conn.cursor()
        cur.execute("SELECT profile_json FROM brand_profiles WHERE brand_id = ?", (brand_id,))
        row = cur.fetchone()
        if row:
            return json.loads(row["profile_json"])
        
        # Fallback mock profile if not found in db
        return {
            "brand_id": brand_id,
            "brand_name": brand_id.capitalize(),
            "top_keywords": ["precision", "excellence", "craft"],
            "tone_label": "authoritative"
        }
    except sqlite3.Error as e:
        logger.error(f"DB Error fetching profile: {e}")
        return {}
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
            "error": "text_too_short"
        }
    
    brand_profile = get_brand_profile(req.brand_id)
    score_before = mock_score_consistency(req.text, brand_profile)
    
    return {
        "brand_id": req.brand_id,
        "brand_name": brand_profile.get("brand_name", req.brand_id.capitalize()),
        **score_before,
        "error": None
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
            "score_after": None,
            "error": "text_too_short"
        }

    brand_profile = get_brand_profile(req.brand_id)
    brand_name = brand_profile.get("brand_name", req.brand_id.capitalize())
    
    # 1. Score Before
    score_before = mock_score_consistency(req.text, brand_profile)
    
    # 2. Get Edit Plan
    edit_plan = mock_generate_edit_plan(req.text, brand_profile)
    
    # 3. Retrieve Grounding Chunks (RAG)
    chunks = mock_retrieve_grounding_chunks(req.brand_id, req.n_grounding_chunks or 3)
    if not chunks:
        # Fallback chunks if db is empty
        chunks = [
            f"A {brand_name} watch is more than an instrument of precision — it is a statement of enduring achievement."
        ]
    edit_plan["grounding_chunks"] = chunks
    
    # 4. Call LLM to Rewrite
    rewritten_text = None
    if client:
        try:
            prompt = f"Rewrite the following text to align with the {brand_name} brand voice.\n" \
                     f"Goals: {', '.join(edit_plan.get('goals', []))}\n" \
                     f"Tone: {edit_plan.get('tone_direction', '')}\n" \
                     f"Style Rules: {', '.join(edit_plan.get('style_rules', []))}\n" \
                     f"Prefer terms: {', '.join(edit_plan.get('prefer_terms', []))}\n" \
                     f"Avoid terms: {', '.join(edit_plan.get('avoid_terms', []))}\n\n" \
                     f"Brand Content Examples:\n"
            for i, c in enumerate(chunks):
                prompt += f"- {c}\n"
            prompt += f"\nOriginal Text:\n{req.text}\n\nRewritten Text:"

            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=250,
                timeout=LLM_TIMEOUT_SECONDS
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
               "error": "llm_timeout" if "time" in str(e).lower() else f"llm_error: {e}"
            }
    else:
        # Fallback Mock Rewrite if no API key
        rewritten_text = f"This timepiece embodies effortless precision — an enduring companion for those who pursue excellence in every endeavour."
    
    # 5. Score After
    score_after = mock_score_consistency(rewritten_text, brand_profile)
    # Ensure after score is better for demo purposes if using mocks
    score_after["overall_score"] = min(100, score_before["overall_score"] + 45)

    suggestions_list = [
        f"Replace causal language with more measured, elevated vocabulary.",
        f"Introduce brand-anchored terms: {', '.join(edit_plan.get('prefer_terms', [])[:3])}.",
        f"Align tone toward: {edit_plan.get('tone_direction', '')}."
    ]

    return {
        "brand_id": req.brand_id,
        "brand_name": brand_name,
        "original_text": req.text,
        "rewritten_text": rewritten_text,
        "suggestions": suggestions_list,
        "grounding_chunks_used": chunks,
        "score_before": score_before,
        "score_after": score_after,
        "error": None
    }


@app.post("/api/profile/rebuild")
def rebuild_profile(req: dict):
    # Mocking Person C's builder script
    brand_id = req.get("brand_id", "rolex")
    return {
        "status": "success",
        "brand_id": brand_id,
        "built_at": "2026-03-01T12:00:00Z",
        "n_texts": 87
    }

@app.post("/api/index/rebuild")
def rebuild_index():
    # Mocking Person B's FAISS builder script
    return {
        "status": "success",
        "backend": "faiss",
        "n_brands": 4,
        "index_path": "embeddings/brand_profile_index.faiss",
        "built_at": "2026-03-01T12:05:00Z"
    }

@app.post("/api/chunks/rebuild")
def rebuild_chunks():
    # Mocking chunk FAISS builder script
    return {
        "status": "success",
        "backend": "faiss",
        "n_chunks": 412,
        "index_path": "embeddings/brand_chunks_index.faiss",
        "built_at": "2026-03-01T12:06:00Z"
    }

@app.get("/api/analytics")
def get_analytics():
    """Return aggregated historical data."""
    return {
        "total_analyzed": 142,
        "avg_consistency": 84,
        "deviations_fixed": 38,
        "trend": [70, 75, 80, 85, 84]
    }

# Mock in-memory state for profile
app_profile_state = {
    "name": "Your Brand",
    "mission": "Delivering excellence",
    "tone": "Sophisticated",
    "top_keywords": ["precision", "legacy", "craftsmanship"],
    "avg_sentiment": 0.85
}

@app.get("/api/profile")
def get_app_profile():
    return app_profile_state

@app.post("/api/profile")
def update_app_profile(req: ProfileUpdate):
    global app_profile_state
    app_profile_state["name"] = req.brand_name
    app_profile_state["mission"] = req.mission
    app_profile_state["tone"] = req.tone
    # Simulate keyword extraction based on tone
    if req.tone == "Technical":
        app_profile_state["top_keywords"] = ["engineering", "calibration", "mechanics"]
    elif req.tone == "Adventurous":
        app_profile_state["top_keywords"] = ["exploration", "durability", "frontier"]
    else:
        app_profile_state["top_keywords"] = ["precision", "legacy", "craftsmanship"]
        
    return {"status": "success", "message": "Profile updated", "profile": app_profile_state}

@app.post("/api/benchmark")
def run_benchmark(req: BenchmarkRequest):
    return {
        "my_brand": {
            "name": req.my_brand,
            "value": 85,
            "label": "High Alignment"
        },
        "competitor": {
            "name": req.competitor.capitalize(),
            "value": 72,
            "label": "Moderate Alignment"
        },
        "radar_data": [
            {"subject": "Vocab", "A": 85, "B": 70},
            {"subject": "Tone", "A": 90, "B": 65},
            {"subject": "Readability", "A": 75, "B": 80},
            {"subject": "Sentiment", "A": 88, "B": 72},
            {"subject": "Keywords", "A": 82, "B": 60}
        ]
    }

# Run with: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
