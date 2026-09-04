"""
openai_provider.py – Stage 6 OpenAI rewrite provider abstraction.

Centralizes:
    * provider selection (REWRITE_PROVIDER env, defaults to "openai" when
      OPENAI_API_KEY is present, otherwise an explicit "fallback")
    * model/reasoning/output-token configuration (OPENAI_MODEL env,
      default "gpt-5.6-luna", reasoning effort "low")
    * a single Responses-API call per rewrite request
    * clean, secret-safe error mapping (never leaks the API key)

The provider is intentionally a thin, easily-mockable object — the SDK is
never called directly from route handlers or from rewrite_service.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

try:
    import openai
except ImportError:  # pragma: no cover - openai is a declared dependency
    openai = None

DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_MAX_OUTPUT_TOKENS = 400


@dataclass(slots=True)
class RewriteProviderError(Exception):
    """Structured provider failure, mirroring RagError/BenchmarkError's shape."""

    status_code: int
    detail: dict[str, Any]

    def __str__(self) -> str:
        return str(self.detail.get("message") or self.detail.get("error") or "rewrite_provider_error")


def get_openai_model() -> str:
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL)


def get_rewrite_provider_name() -> str:
    """Explicit provider selector — never silently masquerades as OpenAI."""
    configured = os.getenv("REWRITE_PROVIDER", "").strip().lower()
    if configured:
        return configured
    return "openai" if os.getenv("OPENAI_API_KEY") else "fallback"


class OpenAIRewriteProvider:
    """Single-call OpenAI Responses API rewrite provider."""

    name = "openai"

    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: float | None = None):
        self.model = model or get_openai_model()
        resolved_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if openai is None:
            raise RewriteProviderError(
                503, {"error": "rewrite_provider_unavailable", "message": "openai package is not installed"}
            )
        if not resolved_key:
            raise RewriteProviderError(
                503, {"error": "rewrite_provider_unavailable", "message": "OPENAI_API_KEY is not configured"}
            )
        self._client = openai.OpenAI(
            api_key=resolved_key,
            timeout=timeout if timeout is not None else float(os.getenv("LLM_TIMEOUT_SECONDS", 30)),
        )

    def rewrite(self, *, instructions: str, input_text: str, max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS) -> str:
        """Make exactly ONE OpenAI generation request and return the rewritten text."""
        try:
            response = self._client.responses.create(
                model=self.model,
                instructions=instructions,
                input=input_text,
                reasoning={"effort": DEFAULT_REASONING_EFFORT},
                max_output_tokens=max_output_tokens,
            )
        except openai.AuthenticationError as exc:
            raise RewriteProviderError(
                503, {"error": "rewrite_provider_unavailable", "message": "OpenAI authentication failed"}
            ) from exc
        except openai.RateLimitError as exc:
            raise RewriteProviderError(
                429, {"error": "rewrite_provider_rate_limited", "message": "OpenAI rate limit exceeded"}
            ) from exc
        except openai.APIConnectionError as exc:
            raise RewriteProviderError(
                503, {"error": "rewrite_provider_unavailable", "message": "could not reach OpenAI"}
            ) from exc
        except openai.APIError as exc:
            raise RewriteProviderError(
                502, {"error": "rewrite_provider_error", "message": "OpenAI request failed"}
            ) from exc

        text = getattr(response, "output_text", None)
        if not text or not str(text).strip():
            raise RewriteProviderError(
                502, {"error": "rewrite_provider_invalid_response", "message": "provider returned empty output"}
            )
        return str(text).strip()


class FallbackRewriteProvider:
    """Explicit, honest non-OpenAI fallback. Never labeled as OpenAI-grounded."""

    name = "fallback"

    def __init__(self, model: str | None = None):
        self.model = model or "local-fallback"

    def rewrite(self, *, instructions: str, input_text: str, max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS) -> str:
        return input_text.strip()


def build_provider() -> "OpenAIRewriteProvider | FallbackRewriteProvider":
    """Resolve the configured provider. Raises RewriteProviderError, never fakes success."""
    provider_name = get_rewrite_provider_name()
    if provider_name == "openai":
        return OpenAIRewriteProvider()
    if provider_name == "fallback":
        return FallbackRewriteProvider()
    raise RewriteProviderError(
        503,
        {"error": "rewrite_provider_unavailable", "message": f"unknown REWRITE_PROVIDER '{provider_name}'"},
    )
