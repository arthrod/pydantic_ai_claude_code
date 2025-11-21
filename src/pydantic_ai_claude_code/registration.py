"""Registration of Claude Code model with Pydantic AI."""

import logging
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_ai.models import Model

logger = logging.getLogger(__name__)


def register_claude_code_model() -> None:
    """Register claude-code provider with Pydantic AI's model inference.

    This patches pydantic_ai.models.infer_model to recognize various provider strings:

    Formats supported:
    - 'claude-code:sonnet' - Standard Claude model (sonnet, opus, haiku)
    - 'claude-code:custom' - Custom model (uses provider's default model)
    - 'claude-code:preset_id:model' - Provider preset with model alias
      Example: 'claude-code:deepseek:sonnet'

    After calling this (or importing pydantic_ai_claude_code), you can use:
        Agent('claude-code:sonnet')
        Agent('claude-code:deepseek:sonnet')
    instead of:
        Agent(ClaudeCodeModel('sonnet'))
    """
    try:
        from pydantic_ai import models

        # Save the original infer_model function
        _original_infer_model = models.infer_model

        def _patched_infer_model(model: "Model | str") -> "Model":
            """
            Enable recognition of 'claude-code' provider strings when inferring a model.
            
            Recognizes these string formats and produces the corresponding ClaudeCodeModel:
            - "claude-code:<model_name>" → returns ClaudeCodeModel constructed with <model_name>.
            - "claude-code:<preset_id>:<model_alias>" → constructs a ClaudeCodeProvider using <preset_id>, resolves the actual model name from <model_alias>, and returns ClaudeCodeModel(actual_model, provider=provider).
            
            @param model: A Model instance or a provider string. If a Model is passed, it is returned unchanged. If a string matches the supported 'claude-code' formats, a ClaudeCodeModel is created as described above; otherwise inference is delegated to the original framework resolver.
            
            @returns: A resolved Model instance: the original Model if passed, a ClaudeCodeModel for supported 'claude-code' strings, or whatever the original infer_model would return for other inputs.
            """
            # If it's already a Model instance, just return it
            if isinstance(model, models.Model):
                return model

            # Check if it's a claude-code provider string
            if isinstance(model, str):
                try:
                    parts = model.split(":")

                    if len(parts) >= 2 and parts[0] == "claude-code":
                        from .model import ClaudeCodeModel
                        from .provider import ClaudeCodeProvider

                        if len(parts) == 2:
                            # Format: claude-code:model_name
                            # e.g., claude-code:sonnet, claude-code:custom
                            model_name = parts[1]

                            # Validate non-empty model name
                            if not model_name:
                                raise ValueError(
                                    f"Invalid model string '{model}': model name cannot be empty. "
                                    "Use format 'claude-code:model_name'"
                                )

                            logger.debug(
                                "Creating ClaudeCodeModel for model: %s", model_name
                            )
                            return ClaudeCodeModel(model_name)

                        elif len(parts) == 3:
                            # Format: claude-code:preset_id:model_alias
                            # e.g., claude-code:deepseek:sonnet
                            preset_id = parts[1]
                            model_alias = parts[2]

                            # Validate non-empty components
                            if not preset_id:
                                raise ValueError(
                                    f"Invalid model string '{model}': preset_id cannot be empty. "
                                    "Use format 'claude-code:preset_id:model_alias'"
                                )
                            if not model_alias:
                                raise ValueError(
                                    f"Invalid model string '{model}': model_alias cannot be empty. "
                                    "Use format 'claude-code:preset_id:model_alias'"
                                )

                            # Create provider with preset
                            provider = ClaudeCodeProvider(settings={
                                "provider_preset": preset_id
                            })

                            # Get actual model name from preset
                            actual_model = provider.get_model_name(model_alias)

                            logger.debug(
                                "Creating ClaudeCodeModel with preset=%s, "
                                "alias=%s, actual_model=%s",
                                preset_id, model_alias, actual_model
                            )
                            return ClaudeCodeModel(actual_model, provider=provider)

                except ValueError:
                    # Not a valid format, fall through to original
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