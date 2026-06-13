"""Core cost estimation and optional OpenRouter pricing discovery."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from typing import Dict, Optional

import httpx


@dataclass(frozen=True)
class CachePricing:
    read_multiplier: Optional[float] = None
    write_multiplier_5m: Optional[float] = None
    write_multiplier_1h: Optional[float] = None
    write_multiplier_default: Optional[float] = None


@dataclass(frozen=True)
class ModelPricing:
    """Token prices are expressed per one million tokens."""

    input_per_million: float
    output_per_million: float
    request_price: float = 0.0
    cache: Optional[CachePricing] = None
    currency: str = "USD"
    source: str = "static"


STATIC_MODEL_PRICING: Dict[str, ModelPricing] = {
    "anthropic/claude-sonnet-4.6": ModelPricing(
        input_per_million=3.0,
        output_per_million=15.0,
        cache=CachePricing(0.1, 1.25, 2.0, 1.25),
    ),
    "anthropic/claude-opus-4.8": ModelPricing(
        input_per_million=15.0,
        output_per_million=75.0,
        cache=CachePricing(0.1, 1.25, 2.0, 1.25),
    ),
}


def attach_cache_rules(model: str, pricing: ModelPricing) -> ModelPricing:
    lower = model.lower()
    if lower.startswith("anthropic/claude"):
        cache = CachePricing(0.1, 1.25, 2.0, 1.25)
    elif lower.startswith("x-ai/grok"):
        cache = CachePricing(read_multiplier=0.25, write_multiplier_default=0.0)
    elif lower.startswith("deepseek/"):
        cache = CachePricing(read_multiplier=0.1, write_multiplier_default=1.0)
    elif lower.startswith("google/gemini"):
        cache = CachePricing(read_multiplier=0.25, write_multiplier_default=0.0)
    else:
        return pricing
    return replace(pricing, cache=cache)


def estimate_cost(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
    cache_write_tokens: int = 0,
    prompt_cache_ttl: Optional[str] = None,
    pricing_map: Optional[Dict[str, ModelPricing]] = None,
) -> Optional[float]:
    pricing = (pricing_map or STATIC_MODEL_PRICING).get(model)
    if pricing is None:
        return None

    input_tokens = max(0, int(input_tokens or 0))
    output_tokens = max(0, int(output_tokens or 0))
    cached_input_tokens = max(0, min(input_tokens, int(cached_input_tokens or 0)))
    cache_write_tokens = max(
        0,
        min(input_tokens - cached_input_tokens, int(cache_write_tokens or 0)),
    )
    regular_input_tokens = max(0, input_tokens - cached_input_tokens - cache_write_tokens)

    input_cost = regular_input_tokens * pricing.input_per_million / 1_000_000
    output_cost = output_tokens * pricing.output_per_million / 1_000_000
    cache_read_cost = cached_input_tokens * pricing.input_per_million / 1_000_000
    cache_write_cost = cache_write_tokens * pricing.input_per_million / 1_000_000

    if pricing.cache:
        if pricing.cache.read_multiplier is not None:
            cache_read_cost *= pricing.cache.read_multiplier
        ttl = str(prompt_cache_ttl or "").strip().lower()
        if ttl in {"1h", "1hr", "1hour", "1-hour", "60m", "60min"}:
            multiplier = pricing.cache.write_multiplier_1h
        else:
            multiplier = pricing.cache.write_multiplier_5m
        if multiplier is None:
            multiplier = pricing.cache.write_multiplier_default
        if multiplier is not None:
            cache_write_cost *= multiplier

    return input_cost + output_cost + cache_read_cost + cache_write_cost + pricing.request_price


def load_openrouter_pricing(
    api_key: Optional[str] = None,
    *,
    timeout: float = 20.0,
) -> Dict[str, ModelPricing]:
    """Fetch current base token prices. This is explicit and never runs during inference."""

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    response = httpx.get(
        "https://openrouter.ai/api/v1/models",
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()

    result: Dict[str, ModelPricing] = {}
    for item in response.json().get("data", []) or []:
        model_id = item.get("id")
        raw = item.get("pricing") or {}
        if not model_id:
            continue
        try:
            prompt = Decimal(str(raw.get("prompt", "0") or "0"))
            completion = Decimal(str(raw.get("completion", "0") or "0"))
            request = Decimal(str(raw.get("request", "0") or "0"))
        except (InvalidOperation, TypeError, ValueError):
            continue
        pricing = ModelPricing(
            input_per_million=float(prompt * Decimal("1000000")),
            output_per_million=float(completion * Decimal("1000000")),
            request_price=float(request),
            source="openrouter_models_api",
        )
        result[model_id] = attach_cache_rules(model_id, pricing)
    return result
