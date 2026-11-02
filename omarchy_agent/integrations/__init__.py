"""External integrations. Currently: the local LocalAI inference server."""

from .localai import (
    DEFAULT_BASE_URL,
    ConnectionStatus,
    LocalAIClient,
    LocalAIConfigurationError,
    LocalAIError,
    ModelInfo,
)

__all__ = [
    "DEFAULT_BASE_URL",
    "ConnectionStatus",
    "LocalAIClient",
    "LocalAIConfigurationError",
    "LocalAIError",
    "ModelInfo",
]
