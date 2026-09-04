<div align="center">

# Brand Genome Engine

**Semantic Brand-Consistency Scoring, Benchmarking, Analytics & RAG-Grounded Rewrite**

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)]()
[![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688.svg)]()
[![React 18](https://img.shields.io/badge/frontend-React%2018-61dafb.svg)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

</div>

---

## 1. Overview

Brand Genome Engine is a local-first academic proof-of-concept that:

1. Builds a **persistent user "brand genome"** from a mission statement and
   exactly 7 brand snippets, using real NLP feature extraction and a
   MiniLM embedding.
2. **Scores arbitrary copy** for consistency against that genome (tone,
   sentiment, readability, vocabulary overlap).
3. **Benchmarks** the user's brand against any of 10 real competitor
   (luxury-watch) brand profiles.
4. Computes **visual analytics**: five corpus-derived messaging pillars,
   chunk-level t-SNE clustering, tone distribution, and live event history.
5. **Retrieves grounding chunks** via a chunk-level FAISS semantic index
   (RAG), strictly scoped per brand.
6. **Rewrites off-brand copy** with an OpenAI-backed provider, grounded in
   the user's own RAG corpus, scored before/after with the same scorer.

## 2. Final capabilities

| Area | Status |
|---|---|
| Genome Setup | Real — persistent, exact-7-snippet contract |
| Consistency Check | Real — weighted Gaussian/Jaccard scorer, writes history |
| Benchmarking | Real — DB-only, 3 metrics (tone/sentiment/readability) |
| Analytics | Real — five pillars, TF-IDF, chunk t-SNE, tone, live history |
| RAG | Real — MiniLM + per-brand FAISS `IndexFlatIP`, staleness detection |
| Rewrite | Real — OpenAI (or local fallback), same scorer pre/post |
| Rebuild (profile/chunks/index) | Real — idempotent, safe to re-run |
| Frontend | Integrated — Genome Setup / Consistency / Benchmarking / Analytics / Rewrite / Dev Tools |

See [docs/completion_matrix.md](docs/completion_matrix.md) for full,
evidence-based status per item.

## 3. Architecture

```
Frontend (React 18 + Vite)
        |
        v
FastAPI (src/api/main.py)  --lifespan-->  SQLite schema bootstrap
        |
        +--> src/scoring        (consistency scorer, no side effects)
        +--> src/benchmarking   (DB-only competitor comparison)
        +--> src/analytics      (pillars, t-SNE, tone, history, cache)
        +--> src/retrieval      (chunk-level MiniLM + FAISS RAG)
        +--> src/rewrite        (OpenAI / fallback provider)
        +--> src/rebuild        (idempotent profile/chunks/index rebuild)
        |
        v
data/brand_data.db (canonical SQLite, single source of truth)
```

Full data-flow diagram: [docs/architecture.md](docs/architecture.md).
No cloud database, no vector database service — FAISS indexes are local
files rebuilt deterministically from SQLite.

## 4. Data

- 10 competitor luxury-watch brands, 450 processed texts (45/brand),
  657 chunks (51-87/brand), all checked into `data/brand_data.db`.
- User genome: designation + mission + exactly 7 snippets, persisted once
  initialised via `POST /api/genome/init`.

Full schema and table-by-table semantics: [docs/database.md](docs/database.md).

## 5. NLP / ML

Sentiment/formality/readability/vocabulary extractors, MiniLM (384-d)
embeddings, cosine similarity via FAISS `IndexFlatIP`, TF-IDF for pillar
keyword derivation, t-SNE for chunk visualisation, and a weighted
Gaussian/Jaccard consistency scorer. Full methodology:
[docs/methodology.md](docs/methodology.md).

## 6. Setup

Verified environment: **Python 3.10**, **Node.js v24 / npm 11** (current
dev environment; no strict upper bound enforced by tooling).

```bash
# Backend
python -m venv .venv
.venv\Scripts\activate        # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt

# Frontend
cd frontend
npm install
```

## 7. Environment

Copy [.env.example](.env.example) to `.env` and fill in real values.
`OPENAI_API_KEY` is optional — with it unset, Rewrite automatically uses a
local deterministic fallback provider; every other feature (Genome,
Consistency, Benchmark, Analytics, RAG retrieval, Rebuilds) works with no
key at all. Never commit `.env` (it is gitignored).

## 8. Running

```bash
# Backend (from repo root)
uvicorn src.api.main:app --reload
# -> http://localhost:8000  (docs at /docs, health at /api/health)

# Frontend (from frontend/)
npm run dev
# -> http://localhost:5173
```

## 8.1 One-click Windows startup

On Windows, double-click [start-brand-genome.bat](start-brand-genome.bat) from the repository root to start the backend and frontend, wait for readiness, and open the UI in your default browser.

First-run prerequisites:

- Python installed
- Node.js and npm installed
- Python dependencies installed with `pip install -r requirements.txt`

If [frontend/node_modules](frontend/node_modules) is missing, the launcher will run `npm ci` automatically.

Optional shutdown helper: double-click [stop-brand-genome.bat](stop-brand-genome.bat) to stop launcher-owned backend and frontend processes safely.

## 9. Derived artifacts

Two generated, gitignored artifacts are rebuilt deterministically from
`data/brand_data.db` and are fingerprint-validated against it:

```bash
# Analytics cache (pillars/t-SNE/tone), consumed by GET /api/analytics
python -m scripts.build_analytics_cache

# Chunk-level RAG index (per-brand FAISS), consumed by /api/rag/retrieve and /api/rewrite
python -m scripts.build_rag_index
```

Both can also be rebuilt at runtime via `POST /api/rebuild/index` /
`POST /api/rebuild/profile` / `POST /api/rebuild/chunks` (Dev Tools page).

## 10. Testing

```bash
python -m pytest tests/                       # full suite: fast, no real model
python -m pytest tests/ --include-model-tests # + real MiniLM model tests (opt-in)
```

Real-model tests are opt-in (`--include-model-tests` or `RUN_EMBEDDING_TESTS=1`)
to avoid loading `sentence-transformers` in every CI run; they have been
verified to pass locally (see [docs/completion_matrix.md](docs/completion_matrix.md)).

Frontend:

```bash
cd frontend
npm run lint
npm run build
```

One-command Definition-of-Done check (non-destructive, no server required):

```bash
python -m scripts.verify_phase4
```

## 11. Demo flow

Genome -> Consistency -> Benchmark -> Analytics -> Rewrite -> Analytics
(refreshed) -> Dev Tools. Full script with talking points:
[docs/demo_script.md](docs/demo_script.md). Reproducible sample brand:
[docs/demo_fixture.md](docs/demo_fixture.md).

## 12. Limitations

This is an academic proof-of-concept, not a production system:

- Corpus covers only 10 competitor watch brands with a modest text count.
- The user's RAG corpus is only the mission statement + 7 snippets.
- t-SNE is a visualisation aid, not an evaluation metric.
- Tone categories are heuristic/explainable, not a trained classifier.
- Rewrite quality depends on the LLM; the post-score is not guaranteed to
  improve over the pre-score.
- FAISS uses exact (flat) search, appropriate only because each brand's
  corpus is small.
- Single-user, local-only — no authentication, no multi-tenant
  concurrency, no distributed infrastructure.

## 13. Documentation index

- [docs/architecture.md](docs/architecture.md) — data flow, layers, legacy components
- [docs/database.md](docs/database.md) — table-by-table schema and identity model
- [docs/api.md](docs/api.md) — canonical endpoints + deprecated aliases
- [docs/methodology.md](docs/methodology.md) — scoring/analytics/RAG/rewrite methodology
- [docs/scoring_spec.md](docs/scoring_spec.md) — frozen consistency-scorer contract
- [docs/completion_matrix.md](docs/completion_matrix.md) — evidence-based DoD status
- [docs/demo_fixture.md](docs/demo_fixture.md) / [docs/demo_script.md](docs/demo_script.md) — presentation aids

## License

Apache 2.0 — see [LICENSE](LICENSE).
