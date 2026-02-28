# Brand Genome Engine

Brand Genome Engine that learns a brand's **"voice DNA"** from its existing text corpus
(website copy, product descriptions, emails, social posts) and then evaluates new copy
for on-brand consistency, using the concepts of **NLP, AI & RAG**.

---

## Repository layout

```
Brand-Genome-Engine/
├── data/
│   ├── raw/                  # Original CSVs / scraped text
│   └── processed/            # Cleaned & chunked artefacts
├── embeddings/               # FAISS indices & serialised embeddings
├── src/
│   ├── feature_extraction/   # Embedding & feature-extraction logic
│   └── benchmarking/         # Evaluation & scoring utilities
├── scripts/                  # One-off or CLI helper scripts
├── tests/                    # Pytest test suite
├── notebooks/                # Exploratory Jupyter notebooks
├── docs/                     # Documentation & design notes
├── data_ingestion_pipeline.py
├── requirements.txt
└── .gitignore
```

## Quick-start

```bash
# 1. Create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux

# 2. Install dependencies (CPU-only)
pip install -r requirements.txt
#    This installs sentence-transformers, torch (CPU), and other deps.
#    The first embedding call will download all-MiniLM-L6-v2 (~80 MB).

# 3. Ingest raw data into SQLite
#    Place your CSV at raw_texts_watches.csv (repo root), then:
python data_ingestion_pipeline.py

# 4. Run feature extraction (once modules are implemented)
python -m src.feature_extraction  # or a dedicated script in scripts/

# 5. Build the FAISS index
python -m src.feature_extraction  # (placeholder – see module README)

# 6. Run the test suite
pytest tests/ -v
```

## Running tests

```bash
# Run all tests (embedding tests use hash fallback – no model download)
pytest tests/ -v --tb=short

# Run REAL embedding model tests (downloads all-MiniLM-L6-v2 on first run)
RUN_EMBEDDING_TESTS=1 pytest tests/test_embedding_extractor.py -k TestRealModel -v

# Run all tests including real embedding model
RUN_EMBEDDING_TESTS=1 pytest tests/ -v --tb=short
```

## ML contract

The canonical feature schema is defined in
[`docs/ml_contract.md`](docs/ml_contract.md) and enforced by
`src.feature_extraction.text_features.ExtractedFeatures`.

Every text document processed by the pipeline produces an `ExtractedFeatures`
record with these key fields:

| Field                 | Type          | Range / Shape    |
|-----------------------|---------------|------------------|
| `sentiment`           | `float`       | [0.0, 1.0]      |
| `formality`           | `float`       | [0.0, 1.0]      |
| `readability_flesch`  | `float`       | [0.0, ~121.0]   |
| `avg_sentence_length` | `float`       | [0.0, ∞)        |
| `punctuation_density` | `float`       | [0.0, 1.0]      |
| `vocab_diversity`     | `float`       | [0.0, 1.0]      |
| `top_topics`          | `list[str]`   | len = num_topics |
| `topic_weights`       | `list[float]` | same length      |
| `embedding`           | `list[float]` | **len = 384**    |

### Consuming features downstream

```python
from src.feature_extraction import TextFeatureExtractor, ExtractedFeatures

extractor = TextFeatureExtractor()

features: ExtractedFeatures = extractor.extract_all_features(
    text="Rolex epitomises timeless luxury...",
    text_id="txt_001",
    brand_id="brand_rolex",
    brand_name="Rolex",
)

# Use individual fields
print(features.sentiment)        # float in [0, 1]
print(len(features.embedding))   # 384
features.validate()              # raises ValueError on contract violation
```

## License

See [LICENSE](LICENSE) for details.
