"""Pydantic AI Claude Code - Use local Claude CLI as a Pydantic AI model.

This package provides a Pydantic AI model implementation that wraps the local
Claude Code CLI, enabling you to use Claude locally with all Pydantic AI features
including structured responses, tool calling, and streaming.

Example:
    ```python
    from pydantic_ai import Agent
    from pydantic_ai_claude_code import ClaudeCodeModel, ClaudeCodeProvider

    # Create a provider with custom settings
    provider = ClaudeCodeProvider(
        working_directory="/path/to/project",
        allowed_tools=["Read", "Edit", "Bash"],
    )

    # Create a model
    model = ClaudeCodeModel("sonnet", provider=provider)

    # Use with Pydantic AI Agent
    agent = Agent(model)
    result = agent.run_sync("What is 2+2?")
    print(result.data)
    ```

Error Handling:
    For long-running processes (>7 hours), handle OAuth token expiration:
    ```python
    from pydantic_ai import Agent
    from pydantic_ai_claude_code import ClaudeOAuthError

    agent = Agent('claude-code:sonnet')

    try:
        result = agent.run_sync("Long running task")
    except ClaudeOAuthError as e:
        print(f"Auth expired: {e.reauth_instruction}")
        # User runs /login, then retry
    ```

Logging:
    To enable debug logging in your application:
    ```python
    import logging
    logging.getLogger('pydantic_ai_claude_code').setLevel(logging.DEBUG)
    ```
"""

import logging

from .exceptions import ClaudeOAuthError
from .model import ClaudeCodeModel
from .provider import ClaudeCodeProvider
from .provider_presets import (
    ProviderPreset,
    get_preset,
    get_presets_by_category,
    list_presets,
    load_all_presets,
)
from .registration import register_claude_code_model
from .types import ClaudeCodeSettings

# Configure module-level logger
logger = logging.getLogger(__name__)
# Use NullHandler by default - consuming applications configure as needed
logger.addHandler(logging.NullHandler())

# Auto-register on import so users can use Agent('claude-code:sonnet')
register_claude_code_model()

# Get version from package metadata (single source of truth in pyproject.toml)
try:
    from importlib.metadata import version

    __version__ = version("pydantic-ai-claude-code")
except Exception:
    # Fallback for development environments where package isn't installed
    __version__ = "0.0.0.dev"

__all__ = [
    "ClaudeCodeModel",
    "ClaudeCodeProvider",
    "ClaudeCodeSettings",
    "ClaudeOAuthError",
    "ProviderPreset",
    "get_preset",
    "get_presets_by_category",
    "list_presets",
    "load_all_presets",
]
