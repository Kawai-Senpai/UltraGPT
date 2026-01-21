"""Provider abstractions and management for UltraGPT."""

# Apply LangChain patches for OpenRouter compatibility FIRST
# This must be done before any LangChain imports to ensure patches are in place
from . import _langchain_patches  # noqa: F401

from .providers import (
    BaseProvider,
    BaseOpenAICompatibleProvider,
    OpenRouterProvider,
    ProviderManager,
    is_rate_limit_error,
)

__all__ = [
    "BaseProvider",
    "BaseOpenAICompatibleProvider",
    "OpenRouterProvider",
    "ProviderManager",
    "is_rate_limit_error",
]
