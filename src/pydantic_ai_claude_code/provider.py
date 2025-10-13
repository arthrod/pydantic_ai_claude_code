"""Provider for Claude Code CLI model."""

import logging
import tempfile
from pathlib import Path
from typing import Any, cast

from .types import ClaudeCodeSettings

logger = logging.getLogger(__name__)


class ClaudeCodeProvider:
    """Provider for managing Claude Code CLI execution.

    This class handles configuration and execution of the Claude CLI,
    including working directory management, tool permissions, and
    temporary file handling.
    """

    def __init__(
        self,
        *,
        working_directory: str | Path | None = None,
        allowed_tools: list[str] | None = None,
        disallowed_tools: list[str] | None = None,
        append_system_prompt: str | None = None,
        permission_mode: str | None = None,
        model: str | None = None,
        fallback_model: str | None = None,
        max_turns: int | None = None,
        verbose: bool = False,
        dangerously_skip_permissions: bool = True,
        use_temp_workspace: bool = True,
        retry_on_rate_limit: bool = True,
    ):
        """Initialize Claude Code provider.

        Args:
            working_directory: Directory for Claude to operate in. If None and
                use_temp_workspace is False, uses current directory.
            allowed_tools: List of tool names to allow (e.g., ["Bash", "Edit", "Read"])
            disallowed_tools: List of tool names to disallow
            append_system_prompt: Additional system prompt to append
            permission_mode: Permission mode ("acceptEdits", "bypassPermissions", "default", "plan")
            model: Model to use (e.g., "sonnet", "opus", or full model name)
            fallback_model: Fallback model when primary is overloaded
            max_turns: Maximum number of agentic turns (default: None for no limit)
            verbose: Enable verbose output
            dangerously_skip_permissions: Skip all permission checks (default: True for non-interactive use)
            use_temp_workspace: If True, creates a temporary workspace directory (default: True to mimic cloud providers)
            retry_on_rate_limit: If True, automatically wait and retry when rate limits are hit (default: True)
        """
        self.working_directory = working_directory
        self.allowed_tools = allowed_tools
        self.disallowed_tools = disallowed_tools
        self.append_system_prompt = append_system_prompt
        self.permission_mode = permission_mode
        self.model = model
        self.fallback_model = fallback_model
        self.max_turns = max_turns
        self.verbose = verbose
        self.dangerously_skip_permissions = dangerously_skip_permissions
        self.use_temp_workspace = use_temp_workspace
        self.retry_on_rate_limit = retry_on_rate_limit
        self._temp_dir: Path | None = None

        logger.debug(
            "Initialized ClaudeCodeProvider with model=%s, working_directory=%s, "
            "use_temp_workspace=%s, dangerously_skip_permissions=%s, retry_on_rate_limit=%s",
            model,
            working_directory,
            use_temp_workspace,
            dangerously_skip_permissions,
            retry_on_rate_limit,
        )

    def __enter__(self):
        """Context manager entry - creates temp directory if needed."""
        if self.use_temp_workspace and self.working_directory is None:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="claude_code_"))
            self.working_directory = self._temp_dir
            logger.debug("Created temporary workspace: %s", self._temp_dir)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleans up temp directory if created."""
        if self._temp_dir and self._temp_dir.exists():
            import shutil

            logger.debug("Cleaning up temporary workspace: %s", self._temp_dir)
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        return self.__exit__(exc_type, exc_val, exc_tb)

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
            "max_turns": self.max_turns,
            "verbose": self.verbose,
            "dangerously_skip_permissions": self.dangerously_skip_permissions,
            "retry_on_rate_limit": self.retry_on_rate_limit,
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
