"""LLM call monitoring: usage tracking, cost estimation, and JSONL logging.

Every LLM call in the app should go through `track_llm_call` so that token
usage and cost are recorded consistently, whether the call succeeds or fails.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

logger = logging.getLogger("dbi_app.llm_monitor")

LOGS_DIR = Path(__file__).resolve().parent / "logs"
USAGE_LOG_FILE = LOGS_DIR / "llm_usage.jsonl"

# $ per token, derived from published $ per MTok rates.
# Cache writes/reads priced at their standard multipliers of the input rate.
PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-5-5": {
        "input": 2.00 / 1_000_000,
        "output": 10.00 / 1_000_000,
        "cache_write": 2.00 / 1_000_000 * 1.25,
        "cache_read": 2.00 / 1_000_000 * 0.05,
    },
    "claude-haiku-5-5": {
        "input": 0.10 / 1_000_000,
        "output": 0.50 / 1_000_000,
        "cache_write": 0.125 / 1_000_000,
        "cache_read": 0.01 / 1_000_000,
    },
    "claude-sonnet-5": {
        "input": 2.00 / 1_000_000,
        "output": 10.00 / 1_000_000,
        "cache_write": 2.00 / 1_000_000 * 1.25,
        "cache_read": 2.00 / 1_000_000 * 0.1,
    },
    "claude-haiku-4-5": {
        "input": 1.00 / 1_000_000,
        "output": 5.00 / 1_000_000,
        "cache_write": 1.00 / 1_000_000 * 1.25,
        "cache_read": 1.00 / 1_000_000 * 0.1,
    },
}

# Claude Haiku 5.5 charges higher rates for prompts over 100,000 tokens
# (input + cache writes + cache reads for the call).
LONG_PROMPT_THRESHOLD = 100_000
PRICING_LONG_PROMPT: dict[str, dict[str, float]] = {
    "claude-haiku-5-5": {
        "input": 0.50 / 1_000_000,
        "output": 2.50 / 1_000_000,
        "cache_write": 0.625 / 1_000_000,
        "cache_read": 0.05 / 1_000_000,
    },
}

# $ per page, for providers priced by page rather than by token (e.g. OCR).
PAGE_PRICING: dict[str, float] = {
    "mistral-ocr-latest": 4.00 / 1_000,
}


class LLMCallRecord(BaseModel):
    call_id: str
    timestamp: datetime
    provider: Literal["anthropic", "mistral"]
    model: str
    step: str
    policy_name: str
    status: Literal["success", "error"]
    duration_ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    pages_processed: int = 0
    cost_usd: float = 0.0
    error: str | None = None


def compute_cost(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    rates = PRICING.get(model)
    prompt_tokens = input_tokens + cache_creation_tokens + cache_read_tokens
    if model in PRICING_LONG_PROMPT and prompt_tokens > LONG_PROMPT_THRESHOLD:
        rates = PRICING_LONG_PROMPT[model]
    if rates is None:
        logger.warning("No pricing entry for model %r; recording cost as 0.0", model)
        return 0.0
    return (
        input_tokens * rates["input"]
        + output_tokens * rates["output"]
        + cache_creation_tokens * rates["cache_write"]
        + cache_read_tokens * rates["cache_read"]
    )


def compute_page_cost(model: str, pages_processed: int) -> float:
    rate = PAGE_PRICING.get(model)
    if rate is None:
        logger.warning("No per-page pricing entry for model %r; recording cost as 0.0", model)
        return 0.0
    return pages_processed * rate


def record_llm_call(record: LLMCallRecord) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(USAGE_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(record.model_dump_json())
        f.write("\n")
    logger.info(
        "[llm] step=%s model=%s status=%s tokens=%d/%d cache=%d/%d cost=$%.4f (%dms)",
        record.step,
        record.model,
        record.status,
        record.input_tokens,
        record.output_tokens,
        record.cache_creation_tokens,
        record.cache_read_tokens,
        record.cost_usd,
        record.duration_ms,
    )


class _CallHandle:
    """Mutable handle passed into a `track_llm_call` block to report usage."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_creation_tokens = 0
        self.cache_read_tokens = 0
        self.pages_processed = 0
        self.record: LLMCallRecord | None = None

    def set_usage_from_response(self, response) -> None:
        """Read usage off an Anthropic Message (streaming or non-streaming)."""
        usage = response.usage
        self.input_tokens = getattr(usage, "input_tokens", 0) or 0
        self.output_tokens = getattr(usage, "output_tokens", 0) or 0
        self.cache_creation_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0

    def set_usage(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
    ) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_tokens = cache_creation_tokens
        self.cache_read_tokens = cache_read_tokens

    def set_page_usage(self, pages_processed: int) -> None:
        """Record page-based usage for providers priced per page (e.g. OCR)."""
        self.pages_processed = pages_processed


@contextmanager
def track_llm_call(step: str, policy_name: str, provider: str, model: str):
    """Context manager that times a call, then logs an `LLMCallRecord`.

    Usage::

        with track_llm_call("classify_themes:proposal", policy_name, "anthropic", model) as call:
            response = client.messages.create(...)
            call.set_usage_from_response(response)
    """
    handle = _CallHandle()
    start = time.monotonic()
    status: Literal["success", "error"] = "success"
    error_message: str | None = None
    try:
        yield handle
    except Exception as exc:
        status = "error"
        error_message = str(exc)
        raise
    finally:
        duration_ms = int((time.monotonic() - start) * 1000)
        if handle.pages_processed:
            cost = compute_page_cost(model, handle.pages_processed)
        else:
            cost = compute_cost(
                model,
                input_tokens=handle.input_tokens,
                output_tokens=handle.output_tokens,
                cache_creation_tokens=handle.cache_creation_tokens,
                cache_read_tokens=handle.cache_read_tokens,
            )
        record = LLMCallRecord(
            call_id=uuid.uuid4().hex,
            timestamp=datetime.now(timezone.utc),
            provider=provider,
            model=model,
            step=step,
            policy_name=policy_name,
            status=status,
            duration_ms=duration_ms,
            input_tokens=handle.input_tokens,
            output_tokens=handle.output_tokens,
            cache_creation_tokens=handle.cache_creation_tokens,
            cache_read_tokens=handle.cache_read_tokens,
            pages_processed=handle.pages_processed,
            cost_usd=cost,
            error=error_message,
        )
        handle.record = record
        record_llm_call(record)


def write_run_summary(policy_name: str, records: list[LLMCallRecord]) -> Path:
    """Write a per-run totals file under web_app/logs/ (not results/)."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    summary_dir = LOGS_DIR / "run_summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)

    totals = {
        "policy_name": policy_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "call_count": len(records),
        "total_input_tokens": sum(r.input_tokens for r in records),
        "total_output_tokens": sum(r.output_tokens for r in records),
        "total_cache_creation_tokens": sum(r.cache_creation_tokens for r in records),
        "total_cache_read_tokens": sum(r.cache_read_tokens for r in records),
        "total_cost_usd": sum(r.cost_usd for r in records),
        "calls": [r.model_dump(mode="json") for r in records],
    }

    summary_path = summary_dir / f"{policy_name}.json"
    summary_path.write_text(json.dumps(totals, indent=2) + "\n", encoding="utf-8")
    return summary_path
