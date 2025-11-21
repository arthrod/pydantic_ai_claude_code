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
        """
        Initialize the Claude Code provider and apply configuration and provider preset environment variables.

        Parameters:
            settings (ClaudeCodeSettings | None): Optional configuration mapping. Recognized keys include common runtime settings (working_directory, allowed_tools, disallowed_tools, append_system_prompt, permission_mode, model, fallback_model, verbose, timeout_seconds, claude_cli_path, extra_cli_args) and flags controlling behavior:
                - use_temp_workspace: create a temporary working directory when no working_directory is provided
                - dangerously_skip_permissions: bypass permission checks
                - retry_on_rate_limit: enable automatic retries on rate limits
                - use_sandbox_runtime / sandbox_runtime_path: wrap execution with a sandbox runtime
                - provider_preset: identifier of a provider preset to load
                - provider_api_key: API key to supply to the provider preset
                - provider_template_vars: template variable values to apply to the preset
                - provider_override_env: if true, override existing environment variables when applying the preset

        Warning:
            If a provider_preset is specified, this constructor modifies the global ``os.environ``
            dictionary by setting environment variables from the preset (e.g., ANTHROPIC_BASE_URL).
            These changes persist for the lifetime of the process and may affect other code.
            Use ``get_applied_env_vars()`` to see which variables were set.

        Notes:
            If a provider_preset is specified and found, its environment variables are applied and stored internally; if not found, a warning is logged.
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
        """
        Return the provider-specific model name for a given alias.
        
        If a provider preset is loaded, returns the preset's mapping for the alias; otherwise returns the alias unchanged.
        
        Args:
            model_alias: Model alias (e.g., "sonnet", "haiku", "opus", "custom")
        
        Returns:
            The actual model name to use, or the original `model_alias` if no preset mapping exists.
        """
        if self.provider_preset:
            return self.provider_preset.get_model_name(model_alias)
        return model_alias

    def get_applied_env_vars(self) -> dict[str, str]:
        """
        Provide a copy of environment variables applied from the active provider preset.
        
        Returns:
            dict[str, str]: Mapping of environment variable names to their applied values.
        """
        return self._applied_env_vars.copy()

    def __enter__(self) -> Self:
        """
        Prepare the provider for context usage by creating and assigning a temporary working directory when configured to use a temp workspace.
        
        Returns:
            self: The provider instance with `working_directory` set to the temporary path if one was created.
        """
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