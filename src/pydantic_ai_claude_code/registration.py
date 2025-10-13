"""Registration of Claude Code model with Pydantic AI."""

import logging
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_ai.models import Model

logger = logging.getLogger(__name__)


def register_claude_code_model() -> None:
    """Register claude-code provider with Pydantic AI's model inference.

    This patches pydantic_ai.models.infer_model to recognize 'claude-code:model_name'
    strings and return a ClaudeCodeModel instance.

    After calling this (or importing pydantic_ai_claude_code), you can use:
        Agent('claude-code:sonnet')
    instead of:
        Agent(ClaudeCodeModel('sonnet'))
    """
    try:
        from pydantic_ai import models

        # Save the original infer_model function
        _original_infer_model = models.infer_model

        def _patched_infer_model(model: "Model | str") -> "Model":
            """Patched version of infer_model that supports claude-code provider."""
            # If it's already a Model instance, just return it
            if isinstance(model, models.Model):
                return model

            # Check if it's a claude-code provider string
            if isinstance(model, str):
                try:
                    provider, model_name = model.split(":", maxsplit=1)
                    if provider == "claude-code":
                        from .model import ClaudeCodeModel

                        logger.debug(
                            "Creating ClaudeCodeModel for model: %s", model_name
                        )
                        return ClaudeCodeModel(model_name)
                except ValueError:
                    # Not a provider:model format, fall through to original
                    pass

            # Fall back to original implementation
            return _original_infer_model(model)

        # Replace the function
        models.infer_model = _patched_infer_model
        logger.info(
            "Successfully registered claude-code model provider with Pydantic AI"
        )

    except ImportError as e:
        # pydantic_ai not installed, skip registration
        logger.warning("Failed to register claude-code provider: %s", e)
        warnings.warn(
            "pydantic_ai not found - claude-code provider not registered. "
            "Install pydantic-ai to use 'claude-code:model' strings.",
            ImportWarning,
            stacklevel=2,
        )
