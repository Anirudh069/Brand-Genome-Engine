"""
text_features.py – Canonical ML contract for the Brand Genome Engine.

Every text processed by the pipeline passes through `TextFeatureExtractor`
and comes out as an `ExtractedFeatures` dataclass.  Down-stream consumers
(index builders, benchmarking, API layers) depend **only** on this schema.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Placeholder imports – each will be a real module later.
# For now we import thin stubs that live next to this file.
# ---------------------------------------------------------------------------
from src.feature_extraction.sentiment_extractor import extract_sentiment
from src.feature_extraction.formality_extractor import extract_formality
from src.feature_extraction.readability_extractor import extract_readability, flesch_reading_ease
from src.feature_extraction.vocabulary_extractor import extract_vocab_metrics
from src.feature_extraction.topic_extractor import extract_topics
from src.feature_extraction.embedding_extractor import extract_embedding, get_embedding
from src.feature_extraction.feature_utils import clean_text

logger = logging.getLogger(__name__)

# ── Embedding dimension contract ──────────────────────────────────────────
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 produces 384-d vectors
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_NUM_TOPICS = 5


# ── Data contract ─────────────────────────────────────────────────────────
@dataclass
class ExtractedFeatures:
    """
    Canonical feature record for a single text document.

    Every field is documented with its expected range / shape so that
    downstream consumers can validate without inspecting extractor internals.

    Ranges
    ------
    sentiment           : [0.0, 1.0]   0 = negative, 0.5 = neutral, 1.0 = positive
    formality           : [0.0, 1.0]    informal → formal
    readability_flesch  : [0.0, 121.0]  higher = easier to read (Flesch RE)
    avg_sentence_length : [0.0, ∞)      words per sentence
    punctuation_density : [0.0, 1.0]    punctuation chars / total chars
    vocab_diversity     : [0.0, 1.0]    unique tokens / total tokens (TTR)
    top_topics          : list[str]     length = num_topics
    topic_weights       : list[float]   same length, sums ≤ 1.0
    embedding           : list[float]   length = 384 (MiniLM)
    """

    # ── Identity (nullable – may not always be present) ───────────────────
    text_id: Optional[str] = None
    brand_id: Optional[str] = None
    brand_name: Optional[str] = None

    # ── Raw text ──────────────────────────────────────────────────────────
    text: str = ""

    # ── Scalar features ───────────────────────────────────────────────────
    sentiment: float = 0.5
    formality: float = 0.5
    readability_flesch: float = 0.0
    avg_sentence_length: float = 0.0
    punctuation_density: float = 0.0
    vocab_diversity: float = 0.0

    # ── Topic features ────────────────────────────────────────────────────
    top_topics: list[str] = field(default_factory=list)
    topic_weights: list[float] = field(default_factory=list)

    # ── Embedding ─────────────────────────────────────────────────────────
    embedding: list[float] = field(default_factory=lambda: [0.0] * EMBEDDING_DIM)
    embedding_model: str = DEFAULT_EMBEDDING_MODEL

    # ── Validation helpers ────────────────────────────────────────────────
    def validate(self) -> None:
        """Raise ``ValueError`` if the record violates the ML contract."""
        if len(self.embedding) != EMBEDDING_DIM:
            raise ValueError(
                f"Embedding length {len(self.embedding)} ≠ {EMBEDDING_DIM}"
            )
        if not (0.0 <= self.sentiment <= 1.0):
            raise ValueError(f"Sentiment {self.sentiment} out of [0, 1]")
        if not (0.0 <= self.formality <= 1.0):
            raise ValueError(f"Formality {self.formality} out of [0, 1]")
        if not (0.0 <= self.punctuation_density <= 1.0):
            raise ValueError(
                f"Punctuation density {self.punctuation_density} out of [0, 1]"
            )
        if not (0.0 <= self.vocab_diversity <= 1.0):
            raise ValueError(
                f"Vocab diversity {self.vocab_diversity} out of [0, 1]"
            )
        if len(self.top_topics) != len(self.topic_weights):
            raise ValueError(
                "top_topics and topic_weights must have the same length"
            )


# ── Extractor orchestrator ────────────────────────────────────────────────
class TextFeatureExtractor:
    """
    Orchestrator that delegates to individual feature extractors
    and assembles an ``ExtractedFeatures`` record.
    """

    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        num_topics: int = DEFAULT_NUM_TOPICS,
    ) -> None:
        self.embedding_model = embedding_model
        self.num_topics = num_topics
        logger.info(
            "TextFeatureExtractor initialised  "
            "(model=%s, topics=%d)",
            self.embedding_model,
            self.num_topics,
        )

    # ------------------------------------------------------------------
    def extract_all_features(
        self,
        text: str,
        text_id: Optional[str] = None,
        brand_id: Optional[str] = None,
        brand_name: Optional[str] = None,
    ) -> ExtractedFeatures:
        """
        Run *all* feature extractors on ``text`` and return a single
        ``ExtractedFeatures`` record that satisfies the ML contract.
        """
        cleaned = clean_text(text)

        # ── Scalar features ───────────────────────────────────────────
        sentiment = extract_sentiment(cleaned)
        formality = extract_formality(cleaned)
        readability_flesch = flesch_reading_ease(cleaned)
        _, avg_sentence_length = extract_readability(cleaned)
        vocab_metrics = extract_vocab_metrics(cleaned)
        punctuation_density = vocab_metrics["punctuation_density"]
        vocab_diversity = vocab_metrics["vocab_diversity"]
        avg_sentence_length = vocab_metrics["avg_sentence_length"]

        # ── Topic features ────────────────────────────────────────────
        top_topics, topic_weights = extract_topics(
            cleaned, num_topics=self.num_topics
        )

        # ── Embedding ─────────────────────────────────────────────────
        embedding, embedding_model_used = get_embedding(
            cleaned, model_name=self.embedding_model
        )

        features = ExtractedFeatures(
            text_id=text_id,
            brand_id=brand_id,
            brand_name=brand_name,
            text=cleaned,
            sentiment=sentiment,
            formality=formality,
            readability_flesch=readability_flesch,
            avg_sentence_length=avg_sentence_length,
            punctuation_density=punctuation_density,
            vocab_diversity=vocab_diversity,
            top_topics=top_topics,
            topic_weights=topic_weights,
            embedding=embedding,
            embedding_model=embedding_model_used,
        )

        features.validate()
        return features
