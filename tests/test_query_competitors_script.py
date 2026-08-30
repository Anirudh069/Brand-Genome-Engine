# filepath: tests/test_query_competitors_script.py
"""
Tests for scripts/query_competitors.py.

Each test builds a tiny synthetic parquet + index in a temp directory,
then exercises the query logic end-to-end.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.query_competitors import (
    CompetitorMatch,
    find_competitors,
    main,
    parse_args,
)
from src.benchmarking.retrieval import backend_name, build_index, save_index


# ── Helpers ───────────────────────────────────────────────────────────────

DIM = 8  # small dimension for speed


def _unit_vec(base: list[float]) -> list[float]:
    """Return L2-normalised copy."""
    arr = np.asarray(base, dtype=np.float32)
    return (arr / np.linalg.norm(arr)).tolist()


def _build_test_env(
    tmp_path: Path,
    n_brands: int = 5,
    texts_per_brand: int = 2,
) -> tuple[str, str, str, list[dict]]:
    """
    Create a synthetic features.parquet, build an index, write metadata.

    Returns (features_path, index_path, metadata_path, brand_metas).
    """
    rng = np.random.RandomState(42)
    rows = []
    brand_metas: list[dict] = []
    brand_vecs: list[list[float]] = []

    for b in range(n_brands):
        brand_id = f"brand_{b}"
        brand_name = f"Brand {b}"
        vecs: list[list[float]] = []
        for t in range(texts_per_brand):
            vec = rng.randn(DIM).astype(np.float32)
            vec = (vec / np.linalg.norm(vec)).tolist()
            vecs.append(vec)
            rows.append(
                {
                    "text_id": f"t_{b}_{t}",
                    "brand_id": brand_id,
                    "brand_name": brand_name,
                    "text": f"text for {brand_name} #{t}",
                    "embedding": vec,
                }
            )
        mean_vec = np.mean(vecs, axis=0).astype(np.float32).tolist()
        brand_vecs.append(mean_vec)
        brand_metas.append(
            {"brand_id": brand_id, "brand_name": brand_name, "n_texts": texts_per_brand}
        )

    # Write parquet
    df = pd.DataFrame(rows)
    features_path = str(tmp_path / "features.parquet")
    df.to_parquet(features_path, index=False)

    # Build & save index
    index = build_index(brand_vecs, metric="cosine")
    ext = ".faiss" if index.backend == "faiss" else ".pkl"
    index_path = str(tmp_path / f"brand_profile_index{ext}")
    save_index(index, index_path)

    # Write metadata
    meta_map = {str(i): m for i, m in enumerate(brand_metas)}
    metadata_path = str(tmp_path / "metadata.json")
    Path(metadata_path).write_text(json.dumps(meta_map, indent=2))

    return features_path, index_path, metadata_path, brand_metas


# ═══════════════════════════════════════════════════════════════════════════
#  Argument parsing
# ═══════════════════════════════════════════════════════════════════════════


class TestParseArgs:
    """CLI argument parsing basics."""

    def test_brand_id_arg(self):
        args = parse_args(["--brand_id", "rolex"])
        assert args.brand_id == "rolex"
        assert args.brand_name is None

    def test_brand_name_arg(self):
        args = parse_args(["--brand_name", "Rolex"])
        assert args.brand_name == "Rolex"
        assert args.brand_id is None

    def test_mutual_exclusion(self):
        """Cannot supply both --brand_id and --brand_name."""
        with pytest.raises(SystemExit):
            parse_args(["--brand_id", "x", "--brand_name", "y"])

    def test_neither_supplied(self):
        """Must supply at least one of --brand_id or --brand_name."""
        with pytest.raises(SystemExit):
            parse_args([])

    def test_default_k(self):
        args = parse_args(["--brand_id", "test"])
        assert args.k == 5

    def test_custom_k(self):
        args = parse_args(["--brand_id", "test", "--k", "3"])
        assert args.k == 3


# ═══════════════════════════════════════════════════════════════════════════
#  find_competitors
# ═══════════════════════════════════════════════════════════════════════════


class TestFindCompetitors:
    """Core find_competitors function on synthetic data."""

    def test_returns_list_of_matches(self, tmp_path: Path):
        fp, ip, mp, metas = _build_test_env(tmp_path, n_brands=5)
        results = find_competitors(fp, ip, mp, brand_id="brand_0", k=3)
        assert isinstance(results, list)
        assert all(isinstance(m, CompetitorMatch) for m in results)

    def test_correct_k(self, tmp_path: Path):
        fp, ip, mp, _ = _build_test_env(tmp_path, n_brands=5)
        results = find_competitors(fp, ip, mp, brand_id="brand_0", k=3)
        assert len(results) == 3

    def test_self_match_excluded(self, tmp_path: Path):
        """The query brand must NOT appear in its own results."""
        fp, ip, mp, _ = _build_test_env(tmp_path, n_brands=5)
        results = find_competitors(fp, ip, mp, brand_id="brand_0", k=4)
        result_ids = [m.brand_id for m in results]
        assert "brand_0" not in result_ids

    def test_ranks_are_sequential(self, tmp_path: Path):
        fp, ip, mp, _ = _build_test_env(tmp_path, n_brands=5)
        results = find_competitors(fp, ip, mp, brand_id="brand_0", k=4)
        assert [m.rank for m in results] == [1, 2, 3, 4]

    def test_distances_are_non_negative(self, tmp_path: Path):
        fp, ip, mp, _ = _build_test_env(tmp_path, n_brands=5)
        results = find_competitors(fp, ip, mp, brand_id="brand_0", k=4)
        assert all(m.distance >= -1e-6 for m in results)

    def test_distances_sorted_ascending(self, tmp_path: Path):
        fp, ip, mp, _ = _build_test_env(tmp_path, n_brands=5)
        results = find_competitors(fp, ip, mp, brand_id="brand_0", k=4)
        dists = [m.distance for m in results]
        for a, b in zip(dists, dists[1:]):
            assert a <= b + 1e-6

    def test_query_by_name(self, tmp_path: Path):
        fp, ip, mp, _ = _build_test_env(tmp_path, n_brands=5)
        results = find_competitors(fp, ip, mp, brand_name="Brand 2", k=2)
        assert len(results) == 2
        assert all(m.brand_id != "brand_2" for m in results)

    def test_query_by_name_case_insensitive(self, tmp_path: Path):
        fp, ip, mp, _ = _build_test_env(tmp_path, n_brands=5)
        results = find_competitors(fp, ip, mp, brand_name="brand 2", k=2)
        assert len(results) == 2

    def test_k_clamped_to_available(self, tmp_path: Path):
        """If k >= n_brands-1, return all except self."""
        fp, ip, mp, _ = _build_test_env(tmp_path, n_brands=3)
        results = find_competitors(fp, ip, mp, brand_id="brand_0", k=100)
        assert len(results) == 2  # 3 brands minus self

    def test_all_brands_appear_in_results(self, tmp_path: Path):
        fp, ip, mp, metas = _build_test_env(tmp_path, n_brands=4)
        results = find_competitors(fp, ip, mp, brand_id="brand_0", k=10)
        result_ids = {m.brand_id for m in results}
        expected = {f"brand_{i}" for i in range(1, 4)}
        assert result_ids == expected


# ═══════════════════════════════════════════════════════════════════════════
#  Error handling
# ═══════════════════════════════════════════════════════════════════════════


class TestErrorHandling:
    """Graceful error messages for bad inputs."""

    def test_missing_features_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="features"):
            find_competitors(
                str(tmp_path / "nope.parquet"),
                str(tmp_path / "idx.faiss"),
                str(tmp_path / "meta.json"),
                brand_id="x",
            )

    def test_missing_index_file(self, tmp_path: Path):
        # create a dummy features file so it passes that check
        fp = tmp_path / "f.parquet"
        pd.DataFrame({"brand_id": [], "brand_name": [], "embedding": []}).to_parquet(
            str(fp), index=False
        )
        with pytest.raises(FileNotFoundError, match="index"):
            find_competitors(str(fp), str(tmp_path / "nope.faiss"), str(tmp_path / "m.json"), brand_id="x")

    def test_missing_metadata_file(self, tmp_path: Path):
        fp = tmp_path / "f.parquet"
        pd.DataFrame({"brand_id": [], "brand_name": [], "embedding": []}).to_parquet(
            str(fp), index=False
        )
        idx_path = tmp_path / "idx.faiss"
        idx_path.write_bytes(b"")  # dummy
        with pytest.raises(FileNotFoundError, match="metadata"):
            find_competitors(str(fp), str(idx_path), str(tmp_path / "nope.json"), brand_id="x")

    def test_brand_not_found(self, tmp_path: Path):
        fp, ip, mp, _ = _build_test_env(tmp_path, n_brands=3)
        with pytest.raises(ValueError, match="Brand not found"):
            find_competitors(fp, ip, mp, brand_id="nonexistent")

    def test_both_brand_args_raises(self, tmp_path: Path):
        fp, ip, mp, _ = _build_test_env(tmp_path, n_brands=3)
        with pytest.raises(ValueError, match="exactly one"):
            find_competitors(fp, ip, mp, brand_id="brand_0", brand_name="Brand 0")

    def test_neither_brand_arg_raises(self, tmp_path: Path):
        fp, ip, mp, _ = _build_test_env(tmp_path, n_brands=3)
        with pytest.raises(ValueError, match="exactly one"):
            find_competitors(fp, ip, mp)

    def test_metadata_mismatch(self, tmp_path: Path):
        """metadata.json has different count than index vectors."""
        fp, ip, mp, _ = _build_test_env(tmp_path, n_brands=3)
        # Overwrite metadata with only 1 entry
        Path(mp).write_text(json.dumps({"0": {"brand_id": "x", "brand_name": "X", "n_texts": 1}}))
        with pytest.raises(ValueError, match="entries but index has"):
            find_competitors(fp, ip, mp, brand_id="brand_0")


# ═══════════════════════════════════════════════════════════════════════════
#  main() end-to-end via CLI args
# ═══════════════════════════════════════════════════════════════════════════


class TestMainCLI:
    """End-to-end test using main() with argv."""

    def test_main_by_brand_id(self, tmp_path: Path, capsys):
        fp, ip, mp, _ = _build_test_env(tmp_path, n_brands=5)
        matches = main([
            "--features", fp,
            "--index", ip,
            "--metadata", mp,
            "--brand_id", "brand_1",
            "--k", "3",
        ])
        assert len(matches) == 3
        assert all(m.brand_id != "brand_1" for m in matches)

        # Verify something was printed
        captured = capsys.readouterr()
        assert "brand_1" in captured.out or "Brand 1" in captured.out

    def test_main_by_brand_name(self, tmp_path: Path):
        fp, ip, mp, _ = _build_test_env(tmp_path, n_brands=5)
        matches = main([
            "--features", fp,
            "--index", ip,
            "--metadata", mp,
            "--brand_name", "Brand 3",
            "--k", "2",
        ])
        assert len(matches) == 2
        assert all(m.brand_id != "brand_3" for m in matches)

    def test_main_missing_file_exits(self, tmp_path: Path):
        with pytest.raises(SystemExit):
            main([
                "--features", str(tmp_path / "nope.parquet"),
                "--index", str(tmp_path / "nope.faiss"),
                "--metadata", str(tmp_path / "nope.json"),
                "--brand_id", "x",
            ])


# ═══════════════════════════════════════════════════════════════════════════
#  Determinism
# ═══════════════════════════════════════════════════════════════════════════


class TestDeterminism:
    """Repeated queries return identical results."""

    def test_deterministic_results(self, tmp_path: Path):
        fp, ip, mp, _ = _build_test_env(tmp_path, n_brands=5)
        r1 = find_competitors(fp, ip, mp, brand_id="brand_0", k=3)
        r2 = find_competitors(fp, ip, mp, brand_id="brand_0", k=3)
        assert [m.brand_id for m in r1] == [m.brand_id for m in r2]
        assert [m.distance for m in r1] == [m.distance for m in r2]
