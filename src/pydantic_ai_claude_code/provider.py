"""Provider for Claude Code CLI model - infrastructure only.

This provider is stateless and handles ONLY infrastructure concerns:
- Finding/configuring the CLI binary
- Creating model instances via factory method

Model configuration, tools, and prompts are specified at Agent level.
Provider presets (deepseek, kimi) are part of the model string.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import ClaudeCodeModel

logger = logging.getLogger(__name__)


class ClaudeCodeProvider:
    """Provider for Claude Code CLI - infrastructure only.

    This provider is stateless and handles ONLY infrastructure concerns.
    Model configuration, tools, and prompts are specified at Agent level.
    Provider presets are part of the model string.

    Examples:
        >>> provider = ClaudeCodeProvider()
        >>> agent = Agent(model='claude-code:sonnet', ...)  # Uses Anthropic
        >>> agent2 = Agent(model='claude-code:deepseek:sonnet', ...)  # Uses DeepSeek

        # Or create model directly
        >>> model = provider.create_model('sonnet')
        >>> model_with_preset = provider.create_model('sonnet', provider_preset='deepseek')
    """

    def __init__(
        self,
        *,
        cli_path: str | Path | None = None,
    ):
        """Initialize provider with optional CLI path.

        Args:
            cli_path: Path to claude CLI binary. If not provided, searches PATH.
        """
        self._cli_path = str(cli_path) if cli_path else None

        logger.debug(
            "Initialized ClaudeCodeProvider with cli_path=%s",
            self._cli_path,
        )

    @property
    def name(self) -> str:
        """Provider name identifier."""
        return "claude-code"

    @property
    def cli_path(self) -> str | None:
        """Get configured CLI path."""
        return self._cli_path

    def create_model(
        self,
        model_name: str,
        *,
        provider_preset: str | None = None,
    ) -> "ClaudeCodeModel":
        """Create a ClaudeCodeModel instance.

        Called by registration logic when parsing 'claude-code:*' strings.

        Args:
            model_name: Model alias (sonnet, opus, haiku) or full model name
            provider_preset: Optional preset ID (deepseek, kimi, etc.)

        Returns:
            Configured ClaudeCodeModel instance
        """
        settings: dict[str, Any] = {
            "working_directory": str(self.working_directory)
            if self.working_directory
            else None,
            "allowed_tools": self.allowed_tools,
            "disallowed_tools": self.disallowed_tools,
            "append_system_prompt": self.append_system_prompt,
            "permission_mode": self.permission_mode,
            "model": self.model,
            "fallback_model": self.fallback_model,
            "verbose": self.verbose,
            "dangerously_skip_permissions": self.dangerously_skip_permissions,
            "retry_on_rate_limit": self.retry_on_rate_limit,
            "timeout_seconds": self.timeout_seconds,
            "claude_cli_path": self.claude_cli_path,
            "extra_cli_args": self.extra_cli_args,
            "use_sandbox_runtime": self.use_sandbox_runtime,
            "sandbox_runtime_path": self.sandbox_runtime_path,
        }

        # Apply overrides
        for key, value in overrides.items():
            settings[key] = value

        # Remove None values and return as ClaudeCodeSettings
        # TypedDict expects specific keys, but total=False allows partial dicts
        final_settings = {k: v for k, v in settings.items() if v is not None}

        if overrides:
            logger.debug("Generated settings with overrides: %s", overrides)

        return cast(ClaudeCodeSettings, final_settings)

    def to_sdk_options(self, **overrides: Any) -> "ClaudeAgentOptions":
        """Convert provider settings to SDK options format.

        Args:
            **overrides: Override specific settings

        Returns:
            SDK agent options dictionary
        """
        from .sdk_original_files.types import ClaudeAgentOptions

        settings = self.get_settings(**overrides)

        options: ClaudeAgentOptions = ClaudeAgentOptions(
            model=settings.get("model", "sonnet"),
            cwd=settings.get("working_directory"),
            allowed_tools=settings.get("allowed_tools", []),
            disallowed_tools=settings.get("disallowed_tools", []),
            permission_mode=settings.get("permission_mode", "bypassPermissions"),
            cli_path=settings.get("claude_cli_path"),
            timeout_ms=settings.get("timeout_seconds", 900) * 1000,
            verbose=settings.get("verbose", False),
            append_system_prompt=settings.get("append_system_prompt"),
        )

        # Build extra args from extra_cli_args
        extra_args = settings.get("extra_cli_args", [])
        if extra_args:
            options["extra_args"] = {}
            i = 0
            while i < len(extra_args):
                arg = extra_args[i]
                if arg.startswith("--"):
                    key = arg[2:]
                    # Check if next arg is a value
                    if i + 1 < len(extra_args) and not extra_args[i + 1].startswith("--"):
                        options["extra_args"][key] = extra_args[i + 1]
                        i += 2
                    else:
                        options["extra_args"][key] = None
                        i += 1
                else:
                    i += 1

        return options
