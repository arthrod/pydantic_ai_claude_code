"""Utility functions for Claude Code model.

This module maintains backward compatibility by re-exporting functions
from the new modular structure. New code should import directly from
the submodules (utils/, core/, structured/, transport/).

For new code, use:
    from pydantic_ai_claude_code.utils import strip_markdown_code_fence
    from pydantic_ai_claude_code.core import detect_oauth_error
    from pydantic_ai_claude_code.structured import read_structure_from_filesystem
    from pydantic_ai_claude_code.transport import EnhancedCLITransport
"""

# Re-export from new utils modules
from ._utils.json_utils import strip_markdown_code_fence, extract_json_from_text
from ._utils.type_utils import convert_primitive_value, get_type_description
from ._utils.file_utils import get_next_call_subdirectory, copy_additional_files

# Re-export from core modules
from .core.oauth_handler import detect_oauth_error
from .core.retry_logic import (
    detect_rate_limit,
    calculate_wait_time,
    detect_cli_infrastructure_failure,
)
from .core.sandbox_runtime import (
    resolve_sandbox_runtime_path,
    build_sandbox_config,
    wrap_command_with_sandbox,
)
from .core.debug_saver import (
    get_debug_dir,
    save_prompt_debug,
    save_response_debug,
    save_raw_response_to_working_dir,
)

# Keep imports from legacy for functions not yet refactored
from .utils_legacy import (
    # CLI path resolution
    resolve_claude_cli_path,
    # Command building
    build_claude_command,
    # Subprocess helpers
    create_subprocess_async,
    # Working directory management
    _determine_working_directory,
    _setup_working_directory_and_prompt,
    _log_prompt_info,
    # Command execution
    _execute_sync_command,
    _execute_async_command,
    # Response handling
    _parse_json_response,
    _validate_claude_response,
    _process_successful_response,
    # Error handling
    _format_cli_error_message,
    _check_rate_limit,
    _handle_command_failure,
    _classify_execution_error,
    # Main execution functions
    run_claude_sync,
    run_claude_async,
    _try_sync_execution_with_rate_limit_retry,
    _try_async_execution_with_rate_limit_retry,
    # Streaming
    parse_stream_json_line,
    # Constants
    LONG_RUNTIME_THRESHOLD_SECONDS,
    MAX_CLI_RETRIES,
    RETRY_BACKOFF_BASE,
)

# Re-export debug functions with original names (for backward compat)
_get_debug_dir = get_debug_dir
_save_prompt_debug = save_prompt_debug
_save_response_debug = save_response_debug
_save_raw_response_to_working_dir = save_raw_response_to_working_dir
_get_next_call_subdirectory = get_next_call_subdirectory
_copy_additional_files = copy_additional_files

__all__ = [
    # Utility functions
    "strip_markdown_code_fence",
    "extract_json_from_text",
    "convert_primitive_value",
    "get_type_description",
    "get_next_call_subdirectory",
    "copy_additional_files",
    # OAuth
    "detect_oauth_error",
    # Retry logic
    "detect_rate_limit",
    "calculate_wait_time",
    "detect_cli_infrastructure_failure",
    # Sandbox runtime
    "resolve_sandbox_runtime_path",
    "build_sandbox_config",
    "wrap_command_with_sandbox",
    # Debug saving
    "get_debug_dir",
    "save_prompt_debug",
    "save_response_debug",
    "save_raw_response_to_working_dir",
    # CLI functions (from legacy)
    "resolve_claude_cli_path",
    "build_claude_command",
    "create_subprocess_async",
    "run_claude_sync",
    "run_claude_async",
    "parse_stream_json_line",
    # Constants
    "LONG_RUNTIME_THRESHOLD_SECONDS",
    "MAX_CLI_RETRIES",
    "RETRY_BACKOFF_BASE",
    # Private functions exposed for backward compatibility
    "_determine_working_directory",
    "_setup_working_directory_and_prompt",
    "_log_prompt_info",
    "_execute_sync_command",
    "_execute_async_command",
    "_parse_json_response",
    "_validate_claude_response",
    "_process_successful_response",
    "_format_cli_error_message",
    "_check_rate_limit",
    "_handle_command_failure",
    "_classify_execution_error",
    "_try_sync_execution_with_rate_limit_retry",
    "_try_async_execution_with_rate_limit_retry",
    "_get_debug_dir",
    "_save_prompt_debug",
    "_save_response_debug",
    "_save_raw_response_to_working_dir",
    "_get_next_call_subdirectory",
    "_copy_additional_files",
]
