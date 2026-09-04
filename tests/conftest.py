"""
Shared test fixtures and process-level configuration.

On CPython 3.9 + macOS (LibreSSL 2.8), loading both ``faiss-cpu`` and
``sentence-transformers`` in the same process can cause a segfault due to
competing C-level thread initialisation.  To work around this:

* Tests marked ``@pytest.mark.requires_model`` are **skipped by default**
  when the full suite runs (``python -m pytest tests/``).
* Pass ``--include-model-tests`` to include them, or run them in isolation:
  ``python -m pytest tests/test_embedding_extractor.py``.
"""

import os
import logging

import pytest

# Disable tokenizers parallelism to prevent fork-related deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

logger = logging.getLogger(__name__)


def pytest_addoption(parser):
    parser.addoption(
        "--include-model-tests",
        action="store_true",
        default=False,
        help="Include tests that load the real sentence-transformers model "
             "(may segfault on CPython 3.9 + macOS when combined with faiss).",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--include-model-tests"):
        return  # run everything
    skip_marker = pytest.mark.skip(
        reason="Skipped to avoid CPython 3.9 segfault. "
               "Run with --include-model-tests or in isolation."
    )
    for item in items:
        if "requires_model" in item.keywords:
            item.add_marker(skip_marker)


# ── Person C: scoring fixtures ────────────────────────────────────────────────
#
# The behavioural tests run against a real copy of the brand database with
# profiles built by the real builder, rather than a hand-written profile dict.
# A hand-written profile is what allowed a scorer that ranked copy backwards to
# pass a full unit-test suite.

import shutil
import sqlite3
import tempfile

from src.profiles.brand_profile_builder import build_brand_profiles, load_brand_profile

_SOURCE_DB = os.path.join(os.path.dirname(__file__), "..", "data", "brand_data.db")


@pytest.fixture(scope="session")
def live_db():
    """A throwaway copy of the real database with v2 profiles built."""
    if not os.path.exists(_SOURCE_DB):
        pytest.skip(f"No database at {_SOURCE_DB}")

    tmpdir = tempfile.mkdtemp(prefix="bge_test_")
    path = os.path.join(tmpdir, "brand_data.db")
    shutil.copy(_SOURCE_DB, path)

    conn = sqlite3.connect(path)
    try:
        conn.execute("DROP TABLE IF EXISTS analysis_history")
        conn.commit()
    finally:
        conn.close()

    build_brand_profiles(path, verbose=False)
    yield path
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture(scope="session")
def rolex(live_db):
    return load_brand_profile("rolex", live_db)
