<div align="center">
  <h1>🧬 Brand Genome Engine</h1>
  <p><strong>Enterprise-Grade Semantic Brand Consistency & Execution Platform</strong></p>
</div>

---

## 📖 Overview

**Brand Genome Engine** is a full-stack, AI-powered application designed to enforce, measure, and automate brand voice consistency across all corporate communications. By combining **Retrieval-Augmented Generation (RAG)** with deterministic linguistic scoring, the engine evaluates incoming text and generates highly-aligned, on-brand rewrites.

This project delivers a state-of-the-art UI with a high-performance Python backend, enabling marketing and editorial teams to execute flawlessly at scale.

## ✨ Core Features

- 🎯 **Brand Genome Setup:** Configure target sentiment, tone, and core keywords to establish a baseline DNA profile for any brand.
- 📊 **Consistency Scoring:** Multi-dimensional semantic evaluation (Tone, Vocabulary, Readability, Sentiment) of any text snippet against the target brand.
- ✍️ **AI Rewrite Pipeline:** A sophisticated RAG-driven workflow that retrieves on-brand grounding examples (via FAISS) and prompts an LLM to elevate off-brand text.
- 📈 **Market Benchmarking:** Direct side-by-side radar and distribution comparisons against top industry competitors.
- 📉 **Historical Analytics:** Visual trajectory mapping of brand alignment improvements over time.

---

## 🏗️ Architecture Stack

### Backend (Python / FastAPI)
- **Framework:** FastAPI for high-performance, asynchronous REST endpoints.
- **Database:** SQLite3 for lightweight, robust storage of profiles and text chunks.
- **AI Integration:** OpenAI API wrapper for advanced semantic generation, configurable via environment variables.
- **RAG & Search:** FAISS (Facebook AI Similarity Search) integration for ultra-fast vector retrieval of brand grounding chunks.

### Frontend (React.js / Vite)
- **Framework:** React 18 powered by Vite for instant Hot Module Replacement (HMR).
- **Styling:** Tailwind CSS for a highly customized, premium "dark mode" enterprise aesthetic.
- **Visualization:** Recharts for dynamic, responsive radar charts, histograms, and line graphs.
- **Icons & Animation:** Lucide-React and Framer Motion for sleek micro-interactions.

---

## 🚀 Getting Started

### Prerequisites

- **Docker** (Recommended for 1-click execution)
- **Python 3.10+** (For manual backend execution)
- **Node.js 18+ & npm** (For manual frontend execution)

### 1. Environment Configuration

Copy the provided `.env.template` (or create a new `.env` file in the root directory) and configure your API keys:

```env
# LLM Provider
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-openai-api-key
LLM_MODEL=gpt-4o-mini
LLM_TIMEOUT_SECONDS=30

# Storage
SQLITE_DB_PATH=data/brand_data.db
EMBEDDINGS_DIR=embeddings/

# Server
CORS_ORIGIN=http://localhost:5173
API_PORT=8000
```

---

### 2. Execution (Docker) - Recommended

Start the entire application stack using the provided execution scripts. This will automatically build the React frontend and spin up the FastAPI backend in parallel containers.

- **Windows:** `.\run.bat`
- **Mac/Linux:** `./run.sh`

*(Alternatively, run `docker compose up --build` directly)*

- **App UI:** `http://localhost:5173`
- **API Docs:** `http://localhost:8000/docs`

---

### 3. Execution (Manual)

If you prefer to run the services locally without Docker:

#### Start the Backend
```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build brand profiles (populates data/brand_data.db → brand_profiles table)
python -m scripts.build_brand_profiles --db-path data/brand_data.db

# 4. Validate the database (all checks must pass)
python -m scripts.validate_db --db-path data/brand_data.db

# 5. Start the API server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

> **Tip:** To skip slow embedding computation, add `--no-embeddings` to the
> profile builder command. Profiles will work but `embedding_status` will be
> `"missing"` instead of `"ok"`.

#### Start the Frontend
Open a **new terminal window**:
```bash
# 1. Navigate to the frontend directory
cd frontend

# 2. Install Node dependencies
npm install

# 3. Start the Vite development server
npm run dev
```

---

## 📂 Project Structure

```text
brand-genome-engine/
├── data/                  # SQLite databases and raw training data
├── embeddings/            # FAISS vector indexes for RAG
├── frontend/              # Vite + React Application
│   ├── src/
│   │   ├── components/    # Reusable UI widgets and layout shells
│   │   ├── lib/           # Utility functions and API constants
│   │   └── pages/         # Top-level route modules (Analytics, Setup, etc.)
│   └── package.json       # Node dependencies
├── src/
│   ├── api/               # FastAPI route definitions and models
│   │   └── main.py        # Core application entrypoint
│   └── pipeline/          # (Team components) Feature extraction & FAISS scripts
├── .env                   # Environment secrets config
├── docker-compose.yml     # Orchestration
└── requirements.txt       # Python backend dependencies
```

---

<div align="center">
  <p>Built for the modern brand execution team.</p>
</div>
