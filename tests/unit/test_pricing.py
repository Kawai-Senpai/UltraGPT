import pytest
from langchain_core.messages import AIMessage

from ultragpt.core.pricing import (
    CachePricing,
    ModelPricing,
    estimate_cost,
    load_openrouter_pricing,
)
from ultragpt.providers import OpenRouterProvider


def test_estimate_cost_accounts_for_cached_and_written_tokens():
    pricing = {
        "model": ModelPricing(
            input_per_million=3.0,
            output_per_million=15.0,
            cache=CachePricing(0.1, 1.25, 2.0, 1.25),
        )
    }
    cost = estimate_cost(
        model="model",
        input_tokens=10_000,
        output_tokens=1_000,
        cached_input_tokens=8_000,
        cache_write_tokens=1_000,
        pricing_map=pricing,
    )
    assert cost == pytest.approx(0.02415)


def test_one_hour_cache_write_uses_two_times_input_rate():
    pricing = {
        "model": ModelPricing(
            input_per_million=3.0,
            output_per_million=0.0,
            cache=CachePricing(0.1, 1.25, 2.0, 1.25),
        )
    }
    assert estimate_cost(
        model="model",
        input_tokens=1_000,
        output_tokens=0,
        cache_write_tokens=1_000,
        prompt_cache_ttl="1h",
        pricing_map=pricing,
    ) == pytest.approx(0.006)


def test_provider_reported_cost_wins_over_estimate():
    message = AIMessage(
        content="ok",
        response_metadata={
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "total_tokens": 110,
                "cost": 0.123,
                "cost_details": {"upstream_inference_cost": 0.1},
            }
        },
    )
    provider = OpenRouterProvider("test")
    details = provider._extract_usage_details(message)
    finalized = provider._finalize_usage_cost(details, "claude-sonnet-4.6")
    assert finalized["actual_cost"] == 0.123
    assert finalized["cost"] == 0.123
    assert finalized["cost_source"] == "provider_reported"
    assert finalized["cost_details"]["upstream_inference_cost"] == 0.1


def test_estimate_is_used_when_provider_cost_is_absent():
    provider = OpenRouterProvider("test")
    details = {
        "input_tokens": 1_000,
        "output_tokens": 100,
        "cached_input_tokens": 0,
        "cache_write_tokens": 0,
        "actual_cost": None,
    }
    finalized = provider._finalize_usage_cost(
        details,
        "anthropic/claude-sonnet-4.6",
    )
    assert finalized["estimated_cost"] == pytest.approx(0.0045)
    assert finalized["cost_source"] == "estimated"
    assert finalized["cost_currency"] == "USD"


def test_dynamic_pricing_loader_converts_per_token_to_per_million(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": [
                    {
                        "id": "anthropic/claude-sonnet-4.6",
                        "pricing": {
                            "prompt": "0.000003",
                            "completion": "0.000015",
                            "request": "0",
                        },
                    }
                ]
            }

    monkeypatch.setattr(
        "ultragpt.core.pricing.httpx.get",
        lambda *args, **kwargs: Response(),
    )
    pricing = load_openrouter_pricing("key")
    model = pricing["anthropic/claude-sonnet-4.6"]
    assert model.input_per_million == 3.0
    assert model.output_per_million == 15.0
    assert model.cache.read_multiplier == 0.1
