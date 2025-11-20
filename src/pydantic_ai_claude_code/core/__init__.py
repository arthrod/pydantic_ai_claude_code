"""Core modules for Claude Code model.

This package contains core functionality that the Claude Agent SDK doesn't provide,
including OAuth handling, retry logic, sandbox runtime support, and debug saving.
"""

from .oauth_handler import detect_oauth_error
from .retry_logic import (
    detect_rate_limit,
    calculate_wait_time,
    detect_cli_infrastructure_failure,
)
from .sandbox_runtime import (
    resolve_sandbox_runtime_path,
    build_sandbox_config,
    wrap_command_with_sandbox,
)
from .debug_saver import (
    get_debug_dir,
    save_prompt_debug,
    save_response_debug,
    save_raw_response_to_working_dir,
)

__all__ = [
    # OAuth
    "detect_oauth_error",
    # Retry
    "detect_rate_limit",
    "calculate_wait_time",
    "detect_cli_infrastructure_failure",
    # Sandbox
    "resolve_sandbox_runtime_path",
    "build_sandbox_config",
    "wrap_command_with_sandbox",
    # Debug
    "get_debug_dir",
    "save_prompt_debug",
    "save_response_debug",
    "save_raw_response_to_working_dir",
]
