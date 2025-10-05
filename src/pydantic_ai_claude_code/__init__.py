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

Logging:
    To enable debug logging in your application:
    ```python
    import logging
    logging.getLogger('pydantic_ai_claude_code').setLevel(logging.DEBUG)
    ```
"""

import logging

from .model import ClaudeCodeModel
from .provider import ClaudeCodeProvider
from .registration import register_claude_code_model
from .types import ClaudeCodeSettings

# Configure module-level logger
logger = logging.getLogger(__name__)
# Use NullHandler by default - consuming applications configure as needed
logger.addHandler(logging.NullHandler())

# Auto-register on import so users can use Agent('claude-code:sonnet')
register_claude_code_model()

__version__ = "0.1.0"

__all__ = [
    "ClaudeCodeModel",
    "ClaudeCodeProvider",
    "ClaudeCodeSettings",
]
