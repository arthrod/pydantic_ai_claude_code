"""Register claude-code provider with pydantic_ai.

This module patches pydantic_ai's model inference to recognize
'claude-code:*' model strings.

Supported formats:
- 'claude-code:sonnet' → Anthropic Claude Sonnet
- 'claude-code:deepseek:sonnet' → DeepSeek via preset
- 'claude-code:kimi:moonshot' → Kimi via preset
"""

from __future__ import annotations

import logging

from pydantic_ai import models

logger = logging.getLogger(__name__)


def register_claude_code_model() -> None:
    """Register 'claude-code:*' model strings with pydantic_ai.

    This patches pydantic_ai.models.infer_model to recognize model strings
    that start with 'claude-code:' and create appropriate ClaudeCodeModel
    instances.

    Examples:
        >>> register_claude_code_model()
        >>> # Now you can use:
        >>> agent = Agent(model='claude-code:sonnet')
        >>> agent2 = Agent(model='claude-code:deepseek:sonnet')
    """
    _original_infer = models.infer_model

    def _patched_infer(model: models.Model | str) -> models.Model:
        if isinstance(model, models.Model):
            return model

        if isinstance(model, str) and model.startswith("claude-code:"):
            # Import here to avoid circular dependency
            from .provider import ClaudeCodeProvider

            parts = model.split(":", 2)
            provider = ClaudeCodeProvider()

            if len(parts) == 2:
                # Format: claude-code:model_name
                model_name = parts[1]
                if not model_name:
                    raise ValueError(
                        f"Invalid model string '{model}': model name cannot be empty"
                    )
                logger.debug("Creating ClaudeCodeModel with model_name=%s", model_name)
                return provider.create_model(model_name)

            elif len(parts) == 3:
                # Format: claude-code:preset:model_name
                provider_preset = parts[1]
                model_name = parts[2]
                if not provider_preset:
                    raise ValueError(
                        f"Invalid model string '{model}': preset cannot be empty"
                    )
                if not model_name:
                    raise ValueError(
                        f"Invalid model string '{model}': model name cannot be empty"
                    )
                logger.debug(
                    "Creating ClaudeCodeModel with preset=%s, model_name=%s",
                    provider_preset,
                    model_name,
                )
                return provider.create_model(model_name, provider_preset=provider_preset)

        return _original_infer(model)

    models.infer_model = _patched_infer
    logger.info("Registered claude-code model provider")
