<div align="center">

# 🧬 Brand Genome Engine

**Semantic Brand-Consistency Scoring, Benchmarking & Rewrite Platform**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()
[![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688.svg)]()
[![React 18](https://img.shields.io/badge/frontend-React%2018-61dafb.svg)]()
[![Tests](https://img.shields.io/badge/tests-347%20total-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

</div>

---

## 1. What This Project Does

Brand Genome Engine is a full-stack application that **measures how consistently a piece of text aligns with a target brand's voice**, then optionally **rewrites the text to improve alignment** using an LLM.

It solves a concrete problem: marketing teams produce copy across dozens of channels, and voice drift is hard to detect manually. This engine:

1. **Profiles brands** by ingesting their existing copy (150 luxury-watch brand texts ship with the repo) and computing statistical fingerprints — mean sentiment, formality, readability, keyword frequency, and dense embeddings.
2. **Scores arbitrary text** against a brand profile on four dimensions (tone, vocabulary overlap, sentiment alignment, readability match) and returns a weighted composite score.
3. **Benchmarks** one brand against another using radar-chart comparisons built from real profile data.
4. **Rewrites off-brand text** via a RAG pipeline that retrieves grounding chunks from the database, builds a structured prompt, and calls an OpenAI-compatible LLM.
5. **Visualises analytics** — consistency trajectory, tone distribution, t-SNE clustering, and messaging-pillars heatmap.

The domain dataset that ships with the repo covers **10 luxury-watch brands** (Rolex, Omega, TAG Heuer, Tissot, Cartier, Breitling, Hublot, IWC Schaffhausen, Patek Philippe, Audemars Piguet) with 15 texts each.

---

## 2. Current Status — What Works vs What's Placeholder

| Area | Status | Notes |
|---|---|---|
| **Brand profile ingestion & storage** | ✅ Fully working | 10 brands × 15 texts, profiles stored in SQLite |
| **Consistency scoring** (`/api/check-consistency`) | ✅ Fully working | Real NLP — sentiment, formality, readability, vocabulary overlap |
| **Edit-plan generation** (`generate_edit_plan`) | ✅ Fully working | Deterministic goal/style-rule generation from feature delta |
| **LLM rewrite** (`/api/rewrite`) | ✅ Working (requires API key) | Falls back to a hardcoded sentence when no key is set |
| **Benchmarking** (`/api/benchmark`) | ✅ Working | Compares two brand profiles from the DB; returns radar data |
| **FAISS competitor retrieval** | ✅ Working | 5-brand centroid index ships in `embeddings/` |
| **Frontend — Genome Setup** | ✅ Working | In-memory profile update (ephemeral, resets on server restart) |
| **Frontend — Consistency Check** | ✅ Working | Calls real `/api/check-consistency`; scores are live |
| **Frontend — Rewrite UI** | ⏸️ Gated off | `ENABLE_REWRITE_UI = false` in `ConsistencyCheck.jsx` |
| **Frontend — Benchmarking** | ✅ Working | Calls real `/api/benchmark`; radar + bar charts render |
| **Frontend — Analytics** | ⚠️ Partially hardcoded | Top-line metrics (analyzed count, avg consistency, deviations fixed) are live from in-memory counters; trajectory chart uses live trend data; **tone histogram, t-SNE scatter, and pillars heatmap use hardcoded mock data** |
| **Profile rebuild** (`/api/profile/rebuild`) | ⚠️ Stub | Returns canned JSON, does not re-compute |
| **Index rebuild** (`/api/index/rebuild`) | ⚠️ Stub | Returns canned JSON |
| **Chunks rebuild** (`/api/chunks/rebuild`) | ⚠️ Stub | Returns canned JSON |
| **Docker deployment** | ✅ Working | `docker compose up --build` runs both services |
| **Test suite** | ✅ 347 tests | All pass; embedding tests skipped by default to avoid segfault |

---

## 3. How It Works — High-Level Pipeline

```mermaid
flowchart LR
    subgraph Offline Pipeline
        A[Raw CSV / Scraped Text] -->|data_ingestion_pipeline.py| B[(brand_data.db)]
        B -->|build_brand_profiles.py| C[brand_profiles table]
        B -->|run_feature_extraction.py| D[features.parquet]
        D -->|build_embeddings_index.py| E[FAISS Index]
    end

    subgraph Runtime — FastAPI
        F[User Text] --> G[Consistency Scorer]
        G -->|sentiment, formality,\nreadability, vocabulary| H{Score 0–100}
        H --> I[API Response]

        F --> J[Edit Plan Generator]
        J --> K[RAG: Retrieve Grounding Chunks]
        K --> L[LLM Rewrite Prompt]
        L --> M[OpenAI API]
        M --> N[Rewritten Text]
        N --> G
    end

    subgraph Frontend — React / Vite
        O[Genome Setup] --> P[Consistency Check]
        P --> Q[Market Benchmarking]
        Q --> R[Analytics Dashboard]
    end

    I --> P
    C --> G
```

**Scoring flow in detail:**

1. User submits text + `brand_id` → `POST /api/check-consistency`
2. Backend loads `brand_profiles.profile_json` for that brand from SQLite
3. Four feature extractors run on the input text (all deterministic, no GPU):
   - `sentiment_extractor` → lexicon-based score `[0, 1]`
   - `formality_extractor` → heuristic multi-signal score `[0, 1]`
   - `readability_extractor` → Flesch Reading Ease `[0, ~121]`
   - Vocabulary overlap → Jaccard similarity of content-words vs `top_keywords`
4. Each score is compared to the brand profile's mean/std via Gaussian similarity
5. Weighted combination: `0.30 × tone + 0.25 × sentiment + 0.25 × vocab + 0.20 × readability`
6. Result returned as `overall_score` + four sub-scores, all clamped `[0, 100]`

---

## 4. Repository Structure

```
Brand-Genome-Engine/
├── src/                            # Core Python source
│   ├── api/
│   │   └── main.py                 # FastAPI app — all REST endpoints
│   ├── scoring/
│   │   └── consistency.py          # ★ Canonical scorer + edit-plan generator
│   ├── feature_extraction/         # Deterministic NLP feature extractors
│   │   ├── feature_utils.py        #   Shared text cleaning, tokenisation, helpers
│   │   ├── sentiment_extractor.py  #   Lexicon-based sentiment [0,1]
│   │   ├── formality_extractor.py  #   Multi-signal formality heuristic [0,1]
│   │   ├── readability_extractor.py#   Flesch Reading Ease + avg sentence length
│   │   ├── vocabulary_extractor.py #   Vocab diversity, punctuation density
│   │   ├── topic_extractor.py      #   Domain-lexicon + TF keyword topics
│   │   ├── embedding_extractor.py  #   sentence-transformers (lazy) or hash fallback
│   │   └── text_features.py        #   ExtractedFeatures dataclass (ML contract)
│   └── benchmarking/
│       └── retrieval.py            # FAISS / sklearn nearest-neighbour index
│
├── scripts/                        # Offline data-processing scripts
│   ├── build_brand_profiles.py     #   Reads brand_texts → computes profiles → upserts brand_profiles
│   ├── run_feature_extraction.py   #   DB/CSV → features.parquet (sentiment, formality, embeddings…)
│   ├── build_embeddings_index.py   #   features.parquet → FAISS index + metadata.json
│   ├── query_competitors.py        #   CLI: find k-nearest competitor brands
│   └── validate_db.py              #   5-check DB integrity validator (tables, counts, JSON keys)
│
├── data/
│   ├── brand_data.db               # SQLite database (gitignored; built by scripts)
│   ├── raw/
│   │   └── luxury_watch_dataset.csv#   Source CSV (150 texts, 10 brands)
│   └── processed/
│       └── features.parquet        # Pre-computed features (gitignored)
│
├── embeddings/
│   ├── brand_profile_index.faiss   # FAISS index (5 brand centroids, 384-d)
│   └── metadata.json               # Maps index position → {brand_id, brand_name, n_texts}
│
├── frontend/                       # React 18 + Vite + Tailwind CSS
│   ├── src/
│   │   ├── App.jsx                 #   Root component — tab routing
│   │   ├── pages/
│   │   │   ├── BrandSetup.jsx      #   Configure brand name, mission, tone
│   │   │   ├── ConsistencyCheck.jsx#   Paste text → get scored → (rewrite gated)
│   │   │   ├── Benchmarking.jsx    #   Select competitor → radar + bar chart
│   │   │   └── Analytics.jsx       #   Trajectory + tone histogram + t-SNE + heatmap
│   │   ├── components/ui/          #   Reusable UI widgets (Card, Button, BrandSelector…)
│   │   ├── layouts/MainLayout.jsx  #   Sidebar navigation shell
│   │   └── lib/constants.js        #   API_BASE = "http://localhost:8000/api"
│   ├── package.json                #   React, Recharts, Framer Motion, Tailwind, Lucide
│   ├── vite.config.js              #   Vite dev server config (port 5173)
│   └── Dockerfile                  #   Node 18 Alpine container
│
├── tests/                          # 347 tests (pytest)
│   ├── conftest.py                 #   --include-model-tests flag; TOKENIZERS_PARALLELISM=false
│   ├── test_api.py                 #   44 API integration tests (uses TestClient)
│   ├── test_scoring.py             #   17 scorer contract tests
│   ├── test_feature_utils.py       #   58 text-cleaning / tokenisation tests
│   ├── test_sentiment_extractor.py #   23 tests
│   ├── test_formality_extractor.py #   28 tests
│   ├── test_readability_extractor.py#  28 tests
│   ├── test_vocabulary_extractor.py#   15 tests
│   ├── test_topic_extractor.py     #   33 tests
│   ├── test_embedding_extractor.py #   32 tests (marked requires_model — skipped by default)
│   ├── test_retrieval.py           #   26 tests
│   ├── test_query_competitors_script.py # 27 tests
│   └── test_run_feature_extraction_script.py # 16 tests
│
├── docs/
│   ├── scoring_spec.md             # Canonical scoring algorithm specification
│   ├── ml_contract.md              # ExtractedFeatures schema contract
│   ├── feature_validation_report.md# Feature extractor validation results
│   ├── integration_plan.md         # Integration notes
│   ├── phase4_plan.md              # Phase 4 roadmap (plan only, no code changes)
│   └── Brand-Genome-Complete-Guide.pdf  # Complete project guide (PDF)
│
├── notebooks/
│   └── 02_feature_extraction_analysis.ipynb  # Exploratory feature analysis
│
├── data_ingestion_pipeline.py      # CSV → SQLite ingestion (standalone, for watches.db)
├── Dockerfile                      # Backend container (Python 3.10-slim)
├── docker-compose.yml              # Orchestrates backend + frontend
├── start.sh                        # Dev startup script (both services, with --stop)
├── run.sh / run.bat                # Docker-based startup (Mac/Linux / Windows)
├── requirements.txt                # Python dependencies
├── pytest.ini                      # Pytest config + requires_model marker
├── LICENSE                         # Apache 2.0
└── app/                            # Empty (reserved for future use)
```

---

## 5. Data Model — SQLite Tables

The canonical database is `data/brand_data.db`. It contains four tables:

| Table | Rows | Purpose |
|---|---|---|
| `brand_texts_raw` | 150 | Raw ingested texts with full metadata (segment, country, source_type, page_name, category, year_range, URL) |
| `brand_texts` | 150 | Cleaned texts used by the scoring pipeline. Schema: `text_id, brand_id, brand_name, source_type, text, created_at` |
| `brand_chunks` | 150 | Chunked text segments for RAG retrieval. Schema: `chunk_id, text_id, brand_id, brand_name, source_type, chunk_text, char_count, created_at` |
| `brand_profiles` | 10 | One row per brand. Schema: `brand_id (PK), brand_name, n_texts, version, built_at, profile_json` |

### `profile_json` Structure

Each `brand_profiles.profile_json` blob contains:

```json
{
  "brand_id": "rolex",
  "brand_name": "Rolex",
  "n_texts": 15,
  "version": "v1",
  "built_at": "2026-...",
  "mean_sentiment": 0.5593,
  "std_sentiment": 0.0926,
  "avg_sentiment": 0.5593,
  "mean_formality": 0.8045,
  "std_formality": 0.0231,
  "avg_formality": 0.8045,
  "mean_flesch": 44.41,
  "std_flesch": 9.44,
  "avg_readability_flesch": 44.41,
  "mean_vocab_richness": 0.72,
  "std_vocab_richness": 0.05,
  "vocabulary_richness": 0.72,
  "top_keywords": ["rolex", "watch", "case", "oyster", "time", "first", "crown", "submariner", "hans", "wilsdorf"],
  "tone_label": "formal",
  "mean_embedding": [0.012, "...384 floats..."],
  "embedding_status": "ok"
}
```

Both `mean_*` and `avg_*` keys are written for backward compatibility. The scorer reads `mean_*` first, falling back to `avg_*`.

### Brands in the Dataset

| brand_id | brand_name | n_texts |
|---|---|---|
| `rolex` | Rolex | 15 |
| `omega` | Omega | 15 |
| `tag_heuer` | TAG Heuer | 15 |
| `tissot` | Tissot | 15 |
| `cartier` | Cartier | 15 |
| `breitling` | Breitling | 15 |
| `hublot` | Hublot | 15 |
| `iwc` | IWC Schaffhausen | 15 |
| `patek_phillipe` | Patek Philippe | 15 |
| `audemars` | Audemars Piguet | 15 |

---

## 6. Scoring & Metrics

Canonical implementation: [`src/scoring/consistency.py`](src/scoring/consistency.py)  
Specification: [`docs/scoring_spec.md`](docs/scoring_spec.md)

### `compute_consistency_score(text, brand_profile) → dict`

Returns exactly five keys, all floats clamped to `[0, 100]`:

| Key | Algorithm | Weight |
|---|---|---|
| `tone_pct` | Gaussian similarity of text formality vs brand mean formality | 30% |
| `sentiment_alignment_pct` | Gaussian similarity of text sentiment vs brand mean sentiment | 25% |
| `vocab_overlap_pct` | Jaccard similarity of text content-words vs brand `top_keywords` | 25% |
| `readability_match_pct` | Gaussian similarity of Flesch score vs brand mean Flesch | 20% |
| `overall_score` | Weighted average of the four above | — |

**Constraints:**
- Deterministic — same input always produces same output
- No heavy deps at scoring time — no `sentence-transformers`, no GPU
- Texts with < 10 words return all zeros (short-text guard)

### `generate_edit_plan(text, brand_profile) → dict`

Returns a structured plan with `goals`, `avoid_terms`, `prefer_terms`, `style_rules`, `tone_direction`, and `grounding_chunks` (populated by the rewrite endpoint).

### Feature Extractors

| Module | Output | Range | Method |
|---|---|---|---|
| `sentiment_extractor.py` | Sentiment score | `[0, 1]` | Curated positive/negative lexicon with negation handling and intensifier boosting |
| `formality_extractor.py` | Formality score | `[0, 1]` | 9-signal weighted heuristic (sentence length, word length, contractions, formal/informal markers, emoji, pronouns, exclamation density) |
| `readability_extractor.py` | Flesch Reading Ease | `[0, ~121]` | Classic Flesch formula with syllable counting heuristic |
| `vocabulary_extractor.py` | Vocab diversity, punct density | `[0, 1]` | Type-Token Ratio; punctuation char ratio |
| `topic_extractor.py` | Top topics + weights | — | Domain-lexicon matching + TF keyword scoring |
| `embedding_extractor.py` | 384-d dense vector | — | `all-MiniLM-L6-v2` via sentence-transformers (lazy-loaded); deterministic hash fallback |

---

## 7. Benchmarking & Analytics

### Benchmarking (`POST /api/benchmark`)

Compares two brands from the database by computing per-dimension scores from their profile data:

- **Vocab** — derived from `top_keywords` count
- **Tone** — derived from `avg_formality`
- **Readability** — derived from `avg_readability_flesch`
- **Sentiment** — derived from `avg_sentiment`
- **Keywords** — derived from `top_keywords` count

Returns `my_brand` and `competitor` objects with overall scores and labels, plus `radar_data` for the frontend radar chart.

### FAISS Index

The file `embeddings/brand_profile_index.faiss` stores mean-pooled 384-d embeddings for 5 brands (Omega, Patek Philippe, Rolex, TAG Heuer, Tissot). Built by:

```bash
python -m scripts.build_embeddings_index \
    --features data/processed/features.parquet \
    --out_dir embeddings/
```

Used by `scripts/query_competitors.py` to find k-nearest competitor brands by cosine distance.

### Analytics (`GET /api/analytics`)

Returns in-memory aggregated state:
- `total_analyzed` — count of texts scored (seeded at 142)
- `avg_consistency` — running mean of overall scores (seeded at 84)
- `deviations_fixed` — count of rewrites that improved score (seeded at 38)
- `trend` — rolling 5-point score trajectory

> **Note:** The frontend's tone histogram, t-SNE scatter, and messaging-pillars heatmap currently use **hardcoded mock data** in `Analytics.jsx`. Replacing these with live server-computed data is planned for Phase 4.

---

## 8. Running Locally

### Prerequisites

| Tool | Version | Required for |
|---|---|---|
| Python | 3.10+ | Backend, scripts |
| Node.js | 18+ | Frontend |
| npm | 9+ | Frontend dependencies |
| Docker | 20+ (optional) | One-command deployment |

### Option A: Single-command startup (`start.sh`)

```bash
# Make executable (once)
chmod +x start.sh

# Start both backend (:8000) and frontend (:5173)
./start.sh

# Stop both
./start.sh --stop
```

The script:
- Checks for Python and Node.js
- Installs pip dependencies if `uvicorn` is missing
- Installs npm dependencies if `frontend/node_modules/` is missing
- Kills any previous processes on ports 8000/5173
- Launches both in background, saves PIDs
- Traps Ctrl+C to clean up

### Option B: Manual (two terminals)

#### Backend

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (First time) Build brand profiles from the DB texts
python -m scripts.build_brand_profiles --db-path data/brand_data.db

# 4. (Optional) Validate the database
python -m scripts.validate_db --db-path data/brand_data.db

# 5. Start the API server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Frontend (new terminal)

```bash
cd frontend
npm install
npm run dev
```

### Option C: Docker

```bash
# Mac/Linux
./run.sh

# Windows
run.bat

# Or directly
docker compose up --build
```

### Accessing the Application

| Service | URL |
|---|---|
| Frontend UI | http://localhost:5173 |
| Backend API docs (Swagger) | http://localhost:8000/docs |
| Health check | http://localhost:8000/api/health |

### Environment Variables

Create a `.env` file in the project root (a template ships with the repo but is gitignored):

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | LLM provider |
| `OPENAI_API_KEY` | `sk-placeholder-replace-with-real-key` | OpenAI API key. Rewrite endpoint returns a fallback sentence if this is a placeholder. |
| `LLM_MODEL` | `gpt-4o-mini` | Model for rewrite generation |
| `LLM_TIMEOUT_SECONDS` | `30` | LLM request timeout |
| `SQLITE_DB_PATH` | `data/brand_data.db` | Path to SQLite database |
| `EMBEDDINGS_DIR` | `embeddings/` | Directory for FAISS index files |
| `CORS_ORIGIN` | `http://localhost:5173` | Allowed CORS origin |
| `API_PORT` | `8000` | Backend port |
| `FEATURES_PATH` | `data/processed/features.parquet` | Pre-computed features file |
| `INDEX_PATH` | `embeddings/brand_profile_index.faiss` | FAISS index path |
| `METADATA_PATH` | `embeddings/metadata.json` | Index metadata path |

---

## 9. API Reference

### `GET /api/health`

```bash
curl http://localhost:8000/api/health
# {"status":"ok","version":"2.0.0"}
```

### `GET /api/brands`

```bash
curl http://localhost:8000/api/brands
# {"brands":[{"brand_id":"rolex","brand_name":"Rolex"}, ...]}
```

### `POST /api/check-consistency`

```bash
curl -X POST http://localhost:8000/api/check-consistency \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The Oyster Perpetual embodies precision craftsmanship and perpetual excellence, a testament to enduring horological mastery.",
    "brand_id": "rolex"
  }'
```

Response:
```json
{
  "brand_id": "rolex",
  "brand_name": "Rolex",
  "overall_score": 78.2,
  "tone_pct": 91.3,
  "vocab_overlap_pct": 14.3,
  "sentiment_alignment_pct": 85.0,
  "readability_match_pct": 70.1,
  "error": null
}
```

### `POST /api/rewrite`

```bash
curl -X POST http://localhost:8000/api/rewrite \
  -H "Content-Type: application/json" \
  -d '{"text":"This watch is awesome and super easy to wear every day.","brand_id":"rolex"}'
```

Returns `score_before`, `score_after`, `rewritten_text`, `suggestions`, and `grounding_chunks_used`. If no `OPENAI_API_KEY` is configured, `rewritten_text` is a static fallback.

### `POST /api/benchmark`

```bash
curl -X POST http://localhost:8000/api/benchmark \
  -H "Content-Type: application/json" \
  -d '{"my_brand":"Rolex","competitor":"omega","metric":"Sentiment Distribution"}'
```

### `GET /api/analytics`

```bash
curl http://localhost:8000/api/analytics
# {"total_analyzed":142,"avg_consistency":84,"deviations_fixed":38,"trend":[70,75,80,85,84]}
```

### `GET /api/profile` / `POST /api/profile`

Get or update the in-memory brand profile (ephemeral — resets on server restart).

---

## 10. Tests

### Running Tests

```bash
# Run the full suite (skips embedding/model tests by default)
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ -v --cov=src --cov-report=term-missing

# Include embedding model tests (may segfault on CPython 3.9 + macOS)
python -m pytest tests/ -v --include-model-tests

# Run embedding tests in isolation (recommended)
python -m pytest tests/test_embedding_extractor.py -v
```

### Test Coverage Summary

| Test File | Count | Tests |
|---|---|---|
| `test_api.py` | 44 | Full API integration (health, brands, consistency, rewrite, benchmark, analytics, profile) |
| `test_feature_utils.py` | 58 | Text cleaning, tokenisation, sentence splitting, punctuation density, vocab diversity |
| `test_topic_extractor.py` | 33 | Domain-lexicon and TF-keyword topic extraction |
| `test_embedding_extractor.py` | 32 | Hash fallback + real model embedding (requires\_model) |
| `test_formality_extractor.py` | 28 | Multi-signal formality scoring edge cases |
| `test_readability_extractor.py` | 28 | Flesch RE, syllable counting, sentence length |
| `test_query_competitors_script.py` | 27 | CLI competitor query script |
| `test_retrieval.py` | 26 | FAISS + sklearn index build/save/load/query |
| `test_sentiment_extractor.py` | 23 | Lexicon sentiment with negation/intensifiers |
| `test_scoring.py` | 17 | Scorer contract: output keys, ranges, determinism, on-brand > off-brand |
| `test_run_feature_extraction_script.py` | 16 | Feature extraction pipeline script |
| `test_vocabulary_extractor.py` | 15 | Vocab diversity, punctuation density |

### Model Test Skip Mechanism

Loading both `faiss-cpu` and `sentence-transformers` in the same CPython 3.9 process on macOS can cause a segfault due to competing C-level thread initialisation. To handle this:

- Tests in `test_embedding_extractor.py` are decorated with `@pytest.mark.requires_model`
- `tests/conftest.py` registers a `--include-model-tests` CLI flag
- By default, `pytest_collection_modifyitems` adds a skip marker to all `requires_model` tests
- Pass `--include-model-tests` to run them, or run the file in isolation

The `conftest.py` also sets `TOKENIZERS_PARALLELISM=false` and `HF_HUB_DISABLE_TELEMETRY=1` to prevent fork-related deadlocks and telemetry noise.

---

## 11. Troubleshooting

| Problem | Solution |
|---|---|
| `Database file not found: data/brand_data.db` | The `.db` file is gitignored. Run `python -m scripts.build_brand_profiles` to build it, or ensure the raw CSV is at `data/raw/luxury_watch_dataset.csv`. |
| `ModuleNotFoundError: No module named 'src'` | Run commands from the project root, not from inside `src/`. Use `python -m scripts.build_brand_profiles`, not `python scripts/build_brand_profiles.py`. |
| Port 8000 or 5173 already in use | Run `./start.sh --stop` or manually kill: `lsof -ti:8000 \| xargs kill -9` |
| Rewrite returns a generic fallback sentence | Set a real `OPENAI_API_KEY` in `.env`. The placeholder key `sk-placeholder-replace-with-real-key` disables the LLM client. |
| Segfault when running all tests | Embedding model tests conflict with faiss on CPython 3.9 + macOS. Run `python -m pytest tests/ -v` (skips them) or run `test_embedding_extractor.py` in isolation. |
| `textblob` or NLTK errors | Run `python -m textblob.download_corpora` to install required NLTK data. |
| Frontend shows "Network error connecting to the engine" | Ensure the backend is running on port 8000. Check CORS: the backend allows `http://localhost:5173` by default. |
| `validate_db.py` reports failures | Re-run the build pipeline: `python -m scripts.build_brand_profiles --db-path data/brand_data.db` |
| Docker build fails on ARM Mac | `faiss-cpu` and `torch` wheels may not be available for ARM64. Use the manual setup instead. |

---

## 12. Planned Work

Based on [`docs/phase4_plan.md`](docs/phase4_plan.md) (plan document only — no code changes have been made):

**Phase 4 — Full-Stack Integration & Rewrite Agent**

- [ ] **User brand profile from scratch** — Let users create ephemeral brand profiles by submitting name + mission + sample snippets (new `POST /api/user-brand/init`)
- [ ] **Live analytics** — Replace hardcoded tone histogram, t-SNE, and pillars heatmap with server-computed data from `brand_chunks`
- [ ] **RAG retriever** — Dedicated module (`src/rag/retriever.py`) for FAISS-based chunk retrieval
- [ ] **Rewrite agent UI** — Enable `ENABLE_REWRITE_UI` in frontend; add iterative rewrite loop
- [ ] **Real profile/index/chunks rebuild endpoints** — Replace current stubs with actual re-computation

---

## 13. Contributing / Development Notes

- **Scorer is the contract boundary** — `compute_consistency_score` output schema is frozen (5 keys, all `[0, 100]`). Do not add/rename keys without updating tests.
- **Feature extractors must never raise** — They return sane defaults on bad input and log warnings.
- **No heavy deps at scoring time** — The real-time path (`consistency.py`) must not import `sentence-transformers` or `torch`. Embedding-based tone is reserved for the offline pipeline.
- **Embedding dimension is 384** — Changing the model requires rebuilding all FAISS indices and updating `EMBEDDING_DIM` in `text_features.py`.
- **Profile fields use dual naming** — Both `mean_*` and `avg_*` keys are written by `build_brand_profiles.py` for backward compatibility.
- **Run `validate_db.py` after any data changes** — It checks table existence, row counts, profile JSON integrity, and stray `.db` files.

---

## 14. License

This project is licensed under the [Apache License 2.0](LICENSE).

---

## Appendix: Useful Commands

```bash
# ─── Data Pipeline ─────────────────────────────────────────────
# Build brand profiles from ingested texts
python -m scripts.build_brand_profiles --db-path data/brand_data.db

# Build profiles without embeddings (faster, profiles work but embedding_status="missing")
python -m scripts.build_brand_profiles --db-path data/brand_data.db --no-embeddings

# Extract features to parquet
python -m scripts.run_feature_extraction --out data/processed/features.parquet

# Build FAISS competitor index
python -m scripts.build_embeddings_index --features data/processed/features.parquet --out_dir embeddings/

# Query nearest competitors
python -m scripts.query_competitors --brand_name Rolex --k 3

# Validate database integrity
python -m scripts.validate_db --db-path data/brand_data.db

# ─── Testing ───────────────────────────────────────────────────
# Full suite (default — skips model tests)
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ -v --cov=src --cov-report=term-missing

# Include model tests
python -m pytest tests/ -v --include-model-tests

# Single test file
python -m pytest tests/test_scoring.py -v

# ─── Dev Server ────────────────────────────────────────────────
# Start both (script)
./start.sh

# Stop both (script)
./start.sh --stop

# Backend only
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Frontend only
cd frontend && npm run dev

# Docker
docker compose up --build
docker compose down
```
