"""Provider for Claude Code CLI model."""

import logging
import tempfile
from pathlib import Path
from typing import Any, cast

from typing_extensions import Self

from .provider_presets import (
    ProviderPreset,
    apply_provider_environment,
    get_preset,
)
from .types import ClaudeCodeSettings

logger = logging.getLogger(__name__)


class ClaudeCodeProvider:
    """Provider for managing Claude Code CLI execution.

    This class handles configuration and execution of the Claude CLI,
    including working directory management, tool permissions, and
    temporary file handling.
    """

    def __init__(self, settings: ClaudeCodeSettings | None = None):
        """Initialize Claude Code provider.

        Args:
            settings: Configuration dict. Supported keys:
                - working_directory: Directory for Claude to operate in
                - allowed_tools: List of tool names to allow
                - disallowed_tools: List of tool names to disallow
                - append_system_prompt: Additional system prompt to append
                - permission_mode: Permission mode ("acceptEdits", "bypassPermissions", "default", "plan")
                - model: Model to use (e.g., "sonnet", "opus")
                - fallback_model: Fallback model when primary is overloaded
                - verbose: Enable verbose output (default: False)
                - dangerously_skip_permissions: Skip permission checks (default: True)
                - use_temp_workspace: Create temporary workspace (default: True)
                - retry_on_rate_limit: Auto-retry on rate limits (default: True)
                - timeout_seconds: Subprocess timeout in seconds (default: 900)
                - claude_cli_path: Path to claude CLI binary (auto-resolved if not provided)
                - extra_cli_args: Additional CLI arguments to pass through (e.g., ["--debug", "--mcp-config", "config.json"])
                - use_sandbox_runtime: Enable sandbox-runtime wrapping with IS_SANDBOX=1 (default: True)
                - sandbox_runtime_path: Path to srt binary (auto-resolved if not provided)
                - provider_preset: Provider preset ID (e.g., "deepseek", "zhipu_glm")
                - provider_api_key: API key for the provider
                - provider_template_vars: Template variable values for preset configuration
                - provider_override_env: Override existing environment variables (default: False)
        """
        config = settings or {}

        # Load provider preset if specified
        self.provider_preset_id = config.get("provider_preset")
        self.provider_preset: ProviderPreset | None = None
        self._applied_env_vars: dict[str, str] = {}

        if self.provider_preset_id:
            self.provider_preset = get_preset(self.provider_preset_id)
            if self.provider_preset:
                # Apply provider environment variables
                self._applied_env_vars = apply_provider_environment(
                    self.provider_preset,
                    api_key=config.get("provider_api_key"),
                    template_vars=config.get("provider_template_vars"),
                    override_existing=config.get("provider_override_env", False),
                )
                logger.info(
                    "Applied provider preset '%s' with %d environment variables",
                    self.provider_preset_id,
                    len(self._applied_env_vars),
                )
            else:
                logger.warning(
                    "Provider preset '%s' not found. "
                    "Available presets can be listed with list_presets()",
                    self.provider_preset_id,
                )

        self.working_directory = config.get("working_directory")
        self.allowed_tools = config.get("allowed_tools")
        self.disallowed_tools = config.get("disallowed_tools")
        self.append_system_prompt = config.get("append_system_prompt")
        self.permission_mode = config.get("permission_mode")
        self.model = config.get("model")
        self.fallback_model = config.get("fallback_model")
        self.verbose = config.get("verbose", False)
        self.dangerously_skip_permissions = config.get("dangerously_skip_permissions", True)
        self.use_temp_workspace = config.get("use_temp_workspace", True)
        self.retry_on_rate_limit = config.get("retry_on_rate_limit", True)
        self.timeout_seconds = config.get("timeout_seconds", 900)
        self.claude_cli_path = config.get("claude_cli_path")
        self.extra_cli_args = config.get("extra_cli_args")
        self.use_sandbox_runtime = config.get("use_sandbox_runtime", True)
        self.sandbox_runtime_path = config.get("sandbox_runtime_path")
        self._temp_dir: Path | None = None

        logger.debug(
            "Initialized ClaudeCodeProvider with model=%s, "
            "working_directory=%s, use_temp_workspace=%s, "
            "dangerously_skip_permissions=%s, retry_on_rate_limit=%s, timeout_seconds=%s, "
            "claude_cli_path=%s, use_sandbox_runtime=%s, provider_preset=%s",
            self.model,
            self.working_directory,
            self.use_temp_workspace,
            self.dangerously_skip_permissions,
            self.retry_on_rate_limit,
            self.timeout_seconds,
            self.claude_cli_path,
            self.use_sandbox_runtime,
            self.provider_preset_id,
        )

    def get_model_name(self, model_alias: str) -> str:
        """Get the actual model name for a given alias.

        If a provider preset is loaded, maps the alias to the provider's model name.

        Args:
            model_alias: Model alias (e.g., "sonnet", "haiku", "opus", "custom")

        Returns:
            Actual model name to use
        """
        if self.provider_preset:
            return self.provider_preset.get_model_name(model_alias)
        return model_alias

    def get_applied_env_vars(self) -> dict[str, str]:
        """Get the environment variables that were applied from the preset.

        Returns:
            Dictionary of environment variable names to values
        """
        return self._applied_env_vars.copy()

    def __enter__(self) -> Self:
        """Context manager entry - creates temp directory if needed."""
        if self.use_temp_workspace and self.working_directory is None:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="claude_code_"))
            self.working_directory = str(self._temp_dir)
            logger.debug("Created temporary workspace: %s", self._temp_dir)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Context manager exit - temp directories are NOT cleaned up for debugging."""
        if self._temp_dir:
            logger.debug("Temp workspace preserved for inspection: %s", self._temp_dir)
            self._temp_dir = None

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        self.__exit__(exc_type, exc_val, exc_tb)

    def get_settings(self, **overrides: Any) -> ClaudeCodeSettings:
        """Get settings dictionary for Claude Code execution.

        Args:
            **overrides: Override specific settings for this execution

        Returns:
            Settings dictionary
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
