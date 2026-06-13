import pytest

from ultragpt.providers import OpenRouterProvider


@pytest.fixture
def provider():
    return OpenRouterProvider("test")


def test_exact_aliases_win_over_loose_aliases(provider):
    assert provider._transform_model_name("gpt-5.5-pro") == "openai/gpt-5.5-pro"
    assert provider._transform_model_name("claude-sonnet-4.6") == "anthropic/claude-sonnet-4.6"
    assert provider._transform_model_name("sonnet") == "anthropic/claude-sonnet-4.6"


def test_openrouter_body_contains_session_and_optional_provider(provider):
    body = provider._build_extra_body(
        "claude-sonnet-4.6",
        None,
        False,
        {
            "session_id": "thread-123",
            "provider": {"order": ["anthropic"], "allow_fallbacks": False},
            "prompt_cache": True,
            "prompt_cache_mode": "auto",
        },
    )
    assert body["session_id"] == "thread-123"
    assert body["provider"]["order"] == ["anthropic"]
    assert body["cache_control"] == {"type": "ephemeral"}


def test_response_cache_is_off_unless_explicitly_requested(provider):
    assert provider._build_openrouter_headers({}) == {}
    assert provider._build_openrouter_headers({"response_cache": False}) == {
        "X-OpenRouter-Cache": "false"
    }


def test_session_id_limit_is_validated(provider):
    with pytest.raises(ValueError, match="256"):
        provider._build_extra_body(
            "gpt-5.5",
            None,
            False,
            {"session_id": "x" * 257},
        )
