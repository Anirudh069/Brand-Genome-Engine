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
