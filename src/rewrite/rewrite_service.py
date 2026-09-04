"""
rewrite_service.py – Stage 6 canonical Rewrite orchestration.

    INPUT COPY
        -> canonical user-genome scorer (pre-score, no history)
        -> deterministic diagnostics/drift + edit plan
        -> Stage 5 semantic RAG retrieval (brand_id = user_brand only)
        -> ONE OpenAI (or explicit fallback) provider call
        -> canonical user-genome scorer again (post-score, no history)
        -> exactly ONE analysis_history "rewrite" row

This module never uses the legacy lexical ``retrieve_grounding_chunks``
helper and never invents grounding text — every snippet passed to the
provider (and returned to the caller) maps to a real ``chunk_id`` in
``brand_chunks``.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from src.api.genome_service import USER_BRAND_ALIAS, USER_BRAND_DB_ID, load_active_user_genome, write_history_event
from src.retrieval.rag_service import RagError, retrieve_chunks
from src.rewrite.openai_provider import RewriteProviderError, build_provider
from src.scoring.consistency import generate_edit_plan, score_against_user_genome

MAX_OUTPUT_TOKENS = 400
GENOME_CONTEXT_KEYWORD_LIMIT = 8


@dataclass(slots=True)
class RewriteError(Exception):
    """Structured Rewrite failure, mirroring RagError/BenchmarkError's shape."""

    status_code: int
    detail: dict[str, Any]

    def __str__(self) -> str:
        return str(self.detail.get("message") or self.detail.get("error") or "rewrite_error")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _flatten_genome_for_edit_plan(genome: dict[str, Any]) -> dict[str, Any]:
    """generate_edit_plan expects flat mean_formality/mean_sentiment/... keys;
    the persisted genome nests them under tone_features. Flatten only for the
    edit-plan call — does not change any Stage 2 scoring formula/input."""
    flat = dict(genome)
    flat.update(genome.get("tone_features") or {})
    return flat


def _require_user_genome(conn: sqlite3.Connection) -> dict[str, Any]:
    genome = load_active_user_genome(conn)
    if not genome or not genome.get("initialized"):
        raise RewriteError(
            400,
            {
                "error": "genome_not_initialized",
                "message": "Initialize the genome first via POST /api/genome/init.",
                "action": "setup_genome",
                "endpoint": "/api/genome/init",
            },
        )
    return genome


def _require_user_chunks_exist(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM brand_chunks WHERE brand_id = ?", (USER_BRAND_ALIAS,))
    count = cur.fetchone()[0]
    if not count:
        raise RewriteError(
            400,
            {
                "error": "user_genome_chunks_missing",
                "message": "The active user genome has no chunked source text yet. Re-run genome initialization.",
            },
        )


def _retrieve_user_chunks(text: str, top_k: int | None) -> dict[str, Any]:
    try:
        return retrieve_chunks(text, USER_BRAND_ALIAS, top_k=top_k)
    except RagError as exc:
        if exc.detail.get("error") == "unknown_brand":
            raise RewriteError(
                503,
                {
                    "error": "user_grounding_not_indexed",
                    "message": "The active user genome is not present in the RAG index; rebuild required.",
                },
            ) from exc
        # index_missing / index_stale / invalid_query / invalid_top_k propagate with the same shape.
        raise RewriteError(exc.status_code, exc.detail) from exc


def _compact_genome_context(genome: dict[str, Any]) -> dict[str, Any]:
    keywords = genome.get("top_keywords") or genome.get("keywords") or []
    return {
        "designation": genome.get("designation") or "",
        "mission_core_vision": genome.get("mission_core_vision") or "",
        "tone_label": genome.get("tone_label") or "",
        "top_keywords": list(keywords)[:GENOME_CONTEXT_KEYWORD_LIMIT],
    }


def _format_edit_plan_for_prompt(edit_plan: dict[str, Any]) -> str:
    lines = [
        f"Goals: {', '.join(edit_plan.get('goals', [])) or 'none'}",
        f"Tone direction: {edit_plan.get('tone_direction', '')}",
        f"Style rules: {', '.join(edit_plan.get('style_rules', [])) or 'none'}",
        f"Prefer terms: {', '.join(edit_plan.get('prefer_terms', [])[:8]) or 'none'}",
        f"Avoid terms: {', '.join(edit_plan.get('avoid_terms', [])[:8]) or 'none'}",
    ]
    return "\n".join(lines)


def _format_grounding_for_prompt(chunks: list[dict[str, Any]]) -> str:
    lines = [
        f"[chunk_id={chunk['chunk_id']}, similarity={round(chunk['score'], 4)}] \"{chunk['chunk_text']}\""
        for chunk in chunks
    ]
    return "\n".join(lines) if lines else "(no grounding chunks retrieved)"


def _build_prompt(
    genome_context: dict[str, Any],
    edit_plan: dict[str, Any],
    chunks: list[dict[str, Any]],
    text: str,
) -> tuple[str, str]:
    instructions = (
        "You are a brand copy editor. Rewrite the input so it better matches the "
        "supplied brand genome while preserving its intended meaning and factual "
        "claims. Use the retrieved brand snippets only as STYLE/TONE/VOICE evidence "
        "-- never copy facts, numbers, dates, certifications, or guarantees from "
        "them into the rewrite unless those same facts already appear in the input "
        "copy. Follow the edit plan. Do not invent new claims. Do not output "
        "analysis, explanations, or formatting -- return ONLY the rewritten copy."
    )
    prompt_input = (
        "USER BRAND CONTEXT\n"
        f"Designation: {genome_context['designation']}\n"
        f"Mission/core vision: {genome_context['mission_core_vision']}\n"
        f"Tone: {genome_context['tone_label']}\n"
        f"Top keywords: {', '.join(genome_context['top_keywords']) or 'none'}\n\n"
        f"EDIT PLAN\n{_format_edit_plan_for_prompt(edit_plan)}\n\n"
        "GROUNDING EXAMPLES (user-brand voice only, not necessarily applicable facts)\n"
        f"{_format_grounding_for_prompt(chunks)}\n\n"
        f"INPUT COPY\n{text}"
    )
    return instructions, prompt_input


def rewrite_copy(
    conn: sqlite3.Connection,
    text: str,
    top_k: int | None = None,
    provider: Any | None = None,
) -> dict[str, Any]:
    """Canonical Stage 6 Rewrite orchestration. Raises RewriteError on failure."""
    if not text or not text.strip():
        raise RewriteError(400, {"error": "invalid_text", "message": "text must be nonblank"})

    # 1-9: local preconditions, cheapest/DB-only checks BEFORE spending any provider credits.
    genome = _require_user_genome(conn)
    _require_user_chunks_exist(conn)

    retrieval = _retrieve_user_chunks(text, top_k)
    resolved_top_k = retrieval["top_k"]
    chunks = retrieval["results"]

    # 10: pre-score — the SAME pure, no-history scorer the Consistency API uses.
    pre_result = score_against_user_genome(text, genome)
    score_before = pre_result["score_overall"]

    # 11-12: deterministic drift/diagnostics (from the same pre_result) + edit plan.
    edit_plan = generate_edit_plan(text, _flatten_genome_for_edit_plan(genome))
    genome_context = _compact_genome_context(genome)
    instructions, prompt_input = _build_prompt(genome_context, edit_plan, chunks, text)

    # 13-19: exactly ONE provider generation request.
    try:
        active_provider = provider if provider is not None else build_provider()
        rewritten_text = active_provider.rewrite(
            instructions=instructions,
            input_text=prompt_input,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
    except RewriteProviderError as exc:
        raise RewriteError(exc.status_code, exc.detail) from exc

    if not rewritten_text or not rewritten_text.strip():
        raise RewriteError(
            502, {"error": "rewrite_provider_invalid_response", "message": "provider returned empty output"}
        )

    # 20: post-score — SAME canonical scorer, called exactly once more.
    post_result = score_against_user_genome(rewritten_text, genome)
    score_after = post_result["score_overall"]
    score_delta = round(score_after - score_before, 2)

    grounding_chunks = [
        {
            "rank": chunk["rank"],
            "chunk_id": chunk["chunk_id"],
            "text_id": chunk["text_id"],
            "source_type": chunk["source_type"],
            "chunk_text": chunk["chunk_text"],
            "score": chunk["score"],
        }
        for chunk in chunks
    ]
    provider_meta = {"name": active_provider.name, "model": getattr(active_provider, "model", "")}
    timestamp = _utc_now()

    # 22: exactly ONE analysis_history row for this whole request.
    write_history_event(
        conn,
        brand_id=USER_BRAND_DB_ID,
        event_type="rewrite",
        input_text=text,
        pre_score=score_before,
        post_score=score_after,
        diagnostics_json=pre_result["diagnostic_breakdown"],
        extra_json={
            "rewritten_text": rewritten_text,
            "score_delta": score_delta,
            "retrieved_chunk_ids": [chunk["chunk_id"] for chunk in grounding_chunks],
            "retrieval_scores": [chunk["score"] for chunk in grounding_chunks],
            "top_k": resolved_top_k,
            "provider": provider_meta["name"],
            "model": provider_meta["model"],
            "genome_version": genome.get("genome_version"),
            "edit_plan": edit_plan,
        },
    )

    return {
        "original_text": text,
        "rewritten_text": rewritten_text,
        "score_before": score_before,
        "score_after": score_after,
        "score_delta": score_delta,
        "feature_breakdown_before": pre_result["feature_breakdown"],
        "feature_breakdown_after": post_result["feature_breakdown"],
        "diagnostic_breakdown_before": pre_result["diagnostic_breakdown"],
        "diagnostic_breakdown_after": post_result["diagnostic_breakdown"],
        "drift_report": pre_result["diagnostic_breakdown"],
        "edit_plan": edit_plan,
        "grounding_chunks": grounding_chunks,
        "provider": provider_meta,
        "timestamp": timestamp,
    }
