"""Utility functions for Claude Code model."""

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

from .exceptions import ClaudeOAuthError
from .types import ClaudeCodeSettings, ClaudeJSONResponse, ClaudeStreamEvent

logger = logging.getLogger(__name__)

# Constants
LONG_RUNTIME_THRESHOLD_SECONDS = 600  # 10 minutes threshold for long runtime warnings
MAX_CLI_RETRIES = 3  # Maximum retries for transient CLI infrastructure failures
RETRY_BACKOFF_BASE = 2  # Exponential backoff base (seconds)


def convert_primitive_value(
    value: str, field_type: str
) -> int | float | bool | str | None:
    """
    Convert a string into a primitive value for the given JSON schema type.
    
    Supported field_type values are "integer", "number", "boolean", and "string". For booleans, the strings "true", "1", and "yes" (case-insensitive) are treated as True; other values are False. Returns None when conversion fails or when an unsupported field_type is provided.
    
    Parameters:
        value (str): The string to convert.
        field_type (str): Target type name ("integer", "number", "boolean", "string").
    
    Returns:
        int | float | bool | str | None: Converted value of the requested type, or `None` if conversion is not possible.
    """
    try:
        if field_type == "integer":
            return int(value)
        elif field_type == "number":
            # Preserve integer vs float distinction
            if "." in value or "e" in value.lower():
                return float(value)
            return int(value)
        elif field_type == "boolean":
            return value.lower() in ("true", "1", "yes")
        elif field_type == "string":
            return value
    except (ValueError, AttributeError):
        pass

    return None


def strip_markdown_code_fence(text: str) -> str:
    """
    Remove surrounding Markdown code fences and return the inner text.
    
    Removes a leading ```json or ``` fence and a trailing ``` fence if present, then trims surrounding whitespace.
    
    Parameters:
        text (str): Text that may be wrapped in Markdown code fences.
    
    Returns:
        str: The input text with surrounding Markdown code fences and surrounding whitespace removed.
    """
    cleaned = text.strip()

    # Remove starting code fence
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]

    # Remove ending code fence
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    return cleaned.strip()


async def create_subprocess_async(
    cmd: list[str], cwd: str | None = None, env: dict[str, str] | None = None
) -> asyncio.subprocess.Process:
    """Create an async subprocess with standard configuration.

    Args:
        cmd: Command and arguments to execute
        cwd: Working directory for the process
        env: Optional environment variables

    Returns:
        Started subprocess with stdout/stderr piped and stdin as PIPE
    """
    # stdin=PIPE to allow passing prompt via stdin (avoids command-line quoting issues)
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.PIPE,  # Changed from DEVNULL to PIPE
        cwd=cwd,
        env=env,
    )
    return process


def _format_cli_error_message(elapsed: float, returncode: int, stderr_text: str) -> str:
    """
    Create a human-readable error message for a failed Claude CLI invocation.
    
    Parameters:
        elapsed (float): Time elapsed in seconds between command start and completion.
        returncode (int): Process exit code returned by the CLI.
        stderr_text (str): Captured standard error output from the CLI.
    
    Returns:
        str: Formatted error message that includes elapsed time and stderr. If elapsed exceeds the long-runtime threshold, the message includes an advisory suggesting the task may be too large and could be split.
    """
    if elapsed > LONG_RUNTIME_THRESHOLD_SECONDS:
        return (
            f"Claude CLI failed after {elapsed:.1f}s with return code {returncode}: {stderr_text}\n"
            f"Long runtime suggests task complexity - consider breaking into smaller tasks."
        )
    else:
        return f"Claude CLI error after {elapsed:.1f}s: {stderr_text}"


def resolve_claude_cli_path(settings: ClaudeCodeSettings | None = None) -> str:
    """
    Finds the filesystem path to the Claude CLI binary.
    
    Resolution priority:
    1. `claude_cli_path` in the provided settings
    2. `CLAUDE_CLI_PATH` environment variable
    3. auto-resolve `claude` from the system PATH via `shutil.which`
    
    Parameters:
        settings (ClaudeCodeSettings | None): Optional settings object that may provide `claude_cli_path`.
    
    Returns:
        str: Filesystem path to the Claude CLI binary.
    
    Raises:
        RuntimeError: If the CLI binary cannot be located by any of the resolution methods.
    """
    # Priority 1: Settings
    if settings and settings.get("claude_cli_path"):
        cli_path = cast(str, settings["claude_cli_path"])
        logger.debug("Using claude CLI from settings: %s", cli_path)
        return cli_path

    # Priority 2: Environment variable
    env_path = os.environ.get("CLAUDE_CLI_PATH")
    if env_path:
        logger.debug("Using claude CLI from CLAUDE_CLI_PATH env var: %s", env_path)
        return env_path

    # Priority 3: Auto-resolve from PATH
    which_path = shutil.which("claude")
    if which_path:
        logger.debug("Auto-resolved claude CLI from PATH: %s", which_path)
        return which_path

    # Not found
    logger.error("Could not find claude CLI binary")
    raise RuntimeError(
        "Could not find claude CLI binary. Please either:\n"
        "1. Install Claude Code (see claude.com/claude-code)\n"
        "2. Set claude_cli_path in ClaudeCodeSettings\n"
        "3. Set CLAUDE_CLI_PATH environment variable\n"
        "4. Add claude binary to your PATH"
    )


def resolve_sandbox_runtime_path(settings: ClaudeCodeSettings | None = None) -> str:
    """Resolve path to sandbox-runtime (srt) binary.

    Resolution priority:
    1. sandbox_runtime_path from settings (if provided)
    2. SANDBOX_RUNTIME_PATH environment variable
    3. shutil.which('srt') - auto-resolve from PATH

    Args:
        settings: Optional settings containing sandbox_runtime_path

    Returns:
        Path to srt binary

    Raises:
        RuntimeError: If srt binary cannot be found
    """
    # Priority 1: Settings
    if settings and settings.get("sandbox_runtime_path"):
        srt_path = cast(str, settings["sandbox_runtime_path"])
        logger.debug("Using sandbox-runtime from settings: %s", srt_path)
        return srt_path

    # Priority 2: Environment variable
    env_path = os.environ.get("SANDBOX_RUNTIME_PATH")
    if env_path:
        logger.debug("Using sandbox-runtime from SANDBOX_RUNTIME_PATH env var: %s", env_path)
        return env_path

    # Priority 3: Auto-resolve from PATH
    which_path = shutil.which("srt")
    if which_path:
        logger.debug("Auto-resolved sandbox-runtime from PATH: %s", which_path)
        return which_path

    # Not found
    logger.error("Could not find sandbox-runtime (srt) binary")
    raise RuntimeError(
        "Could not find sandbox-runtime (srt) binary. Please either:\n"
        "1. Install sandbox-runtime: npm install -g @anthropic-ai/sandbox-runtime\n"
        "2. Set sandbox_runtime_path in ClaudeCodeSettings\n"
        "3. Set SANDBOX_RUNTIME_PATH environment variable\n"
        "4. Add srt binary to your PATH"
    )


def detect_rate_limit(error_output: str) -> tuple[bool, str | None]:
    """
    Detects a rate-limit condition in CLI output and extracts the reported reset time.
    
    Parameters:
        error_output (str): Combined stdout and stderr text from the Claude CLI to scan for rate-limit messages.
    
    Returns:
        tuple[bool, str | None]: `True` if a rate limit was detected, `False` otherwise; the second element is the reset time string (e.g., "3PM") when detected, or `None` when not.
    """
    # Pattern matches: "limit reached.*resets 3PM" or similar
    rate_limit_match = re.search(
        r"limit reached.*resets\s+(\d{1,2}[AP]M)", error_output, re.IGNORECASE
    )

    if rate_limit_match:
        reset_time_str = rate_limit_match.group(1).strip()
        logger.info("Rate limit detected, reset time: %s", reset_time_str)
        return True, reset_time_str

    return False, None


def calculate_wait_time(reset_time_str: str) -> int:
    """
    Compute the number of seconds to wait until the specified reset time.
    
    Parameters:
        reset_time_str (str): Reset time in 12-hour format with AM/PM (e.g., "3PM", "11AM").
    
    Returns:
        int: Seconds to wait until the reset time plus a 1-minute buffer. Returns 300 if the input cannot be parsed.
    """
    try:
        now = datetime.now()
        # Parse time like "3PM" or "11AM"
        reset_time_obj = datetime.strptime(reset_time_str, "%I%p")
        reset_datetime = now.replace(
            hour=reset_time_obj.hour,
            minute=reset_time_obj.minute,
            second=0,
            microsecond=0,
        )

        # If reset time is in the past, add a day
        if reset_datetime < now:
            reset_datetime += timedelta(days=1)

        # Add 1-minute buffer
        wait_until = reset_datetime + timedelta(minutes=1)
        wait_seconds = int((wait_until - now).total_seconds())

        logger.info(
            "Rate limit resets at %s, waiting until %s (%d seconds)",
            reset_datetime.strftime("%I:%M%p"),
            wait_until.strftime("%I:%M%p"),
            wait_seconds,
        )

        return max(0, wait_seconds)

    except ValueError as e:
        # Fallback: wait 5 minutes if we can't parse the time
        logger.warning(
            "Could not parse reset time '%s': %s. Defaulting to 5-minute wait.",
            reset_time_str,
            e,
        )
        return 300


def detect_cli_infrastructure_failure(stderr: str) -> bool:
    """
    Detect transient Claude CLI infrastructure failures that should trigger a retry.
    
    Parameters:
        stderr (str): Error output from the Claude CLI.
    
    Returns:
        True if the error indicates a retryable infrastructure failure, False otherwise.
    """
    # Node.js module loading errors (e.g., missing yoga.wasm)
    if "Cannot find module" in stderr:
        return True

    # Node.js module resolution errors
    if "MODULE_NOT_FOUND" in stderr:
        return True

    # Other transient errors
    return bool("ENOENT" in stderr or "EACCES" in stderr)


def detect_oauth_error(stdout: str, stderr: str) -> tuple[bool, str | None]:
    """Detect OAuth authentication errors from Claude CLI output.

    The CLI returns OAuth errors in the JSON response (stdout), not stderr.
    Example response:
    {
        "type": "result",
        "subtype": "success",
        "is_error": true,
        "result": "OAuth token revoked · Please run /login"
    }

    Args:
        stdout: Standard output from Claude CLI (may contain JSON response)
        stderr: Standard error output (typically empty for OAuth errors)

    Returns:
        Tuple of (is_oauth_error, error_message)
        - is_oauth_error: True if this is an OAuth/authentication error
        - error_message: The error message from the CLI, or None if not an OAuth error
    """
    if not stdout:
        return False, None

    try:
        # Try to parse JSON from first line of stdout
        first_line = stdout.strip().split("\n")[0]
        response = json.loads(first_line)

        # Check if this is an error response
        if not isinstance(response, dict):
            return False, None

        if not response.get("is_error"):
            return False, None

        # Check the error message for OAuth/authentication indicators
        result_msg = response.get("result", "")
        error_msg = response.get("error", "")
        combined_msg = f"{result_msg} {error_msg}".lower()

        # OAuth token errors
        oauth_indicators = [
            "oauth token",
            "oauth_token",
            "/login",
            "authentication",
            "auth expired",
            "auth failed",
            "token expired",
            "token revoked",
            "please login",
            "please log in",
        ]

        for indicator in oauth_indicators:
            if indicator in combined_msg:
                # Return the actual error message from the result field
                actual_message = result_msg or error_msg or "Authentication error"
                logger.info("Detected OAuth error: %s", actual_message)
                return True, actual_message

    except (json.JSONDecodeError, KeyError, IndexError):
        # Not a valid JSON response, continue to check stderr
        pass

    return False, None


def _add_tool_permission_flags(cmd: list[str], settings: ClaudeCodeSettings) -> None:
    """
    Append allowed and disallowed tool flags from settings to the CLI command list.
    
    If `settings` contains an `allowed_tools` list, appends `--allowed-tools` followed by the tool names to `cmd`.
    If `settings` contains a `disallowed_tools` list, appends `--disallowed-tools` followed by the tool names to `cmd`.
    The `cmd` list is modified in place.
    
    Parameters:
        cmd (list[str]): Command argument list to be extended.
        settings (ClaudeCodeSettings): Settings providing `allowed_tools` and/or `disallowed_tools` keys.
    """
    allowed_tools = settings.get("allowed_tools")
    if allowed_tools:
        cmd.append("--allowed-tools")
        cmd.extend(allowed_tools)

    disallowed_tools = settings.get("disallowed_tools")
    if disallowed_tools:
        cmd.append("--disallowed-tools")
        cmd.extend(disallowed_tools)


def _add_model_flags(cmd: list[str], settings: ClaudeCodeSettings) -> None:
    """
    Append model-related CLI flags to `cmd` when corresponding settings are present.
    
    Parameters:
        cmd (list[str]): Mutable command argument list to extend.
        settings (ClaudeCodeSettings): Settings mapping; if `model` is set, adds `--model <value>`; if `fallback_model` is set, adds `--fallback-model <value>`; if `session_id` is set, adds `--session-id <value>`.
    """
    model = settings.get("model")
    if model:
        cmd.extend(["--model", model])

    fallback_model = settings.get("fallback_model")
    if fallback_model:
        cmd.extend(["--fallback-model", fallback_model])

    session_id = settings.get("session_id")
    if session_id:
        cmd.extend(["--session-id", session_id])


def _add_settings_flags(cmd: list[str], settings: ClaudeCodeSettings) -> None:
    """
    Add flags derived from runtime settings to an existing Claude CLI command list.
    
    This mutates the provided `cmd` list by appending flags for tool permissions, optional system prompt appending, permission mode (defaults to "bypassPermissions" when not set), an optional dangerous permission bypass flag, and model-related flags.
    
    Parameters:
        cmd (list[str]): The command argument list to be modified in-place.
        settings (ClaudeCodeSettings): Mapping-like settings containing keys such as
            "append_system_prompt", "permission_mode", and "dangerously_skip_permissions",
            as well as model and tool permission configuration consumed by helper functions.
    """
    # Tool permissions
    _add_tool_permission_flags(cmd, settings)

    # System prompt
    append_system_prompt = settings.get("append_system_prompt")
    if append_system_prompt:
        cmd.extend(["--append-system-prompt", append_system_prompt])

    # Permission mode (default to bypassPermissions for non-interactive use)
    permission_mode = settings.get("permission_mode") or "bypassPermissions"
    cmd.extend(["--permission-mode", permission_mode])

    # Permission bypass
    if settings.get("dangerously_skip_permissions"):
        cmd.append("--dangerously-skip-permissions")

    # Model settings
    _add_model_flags(cmd, settings)


def build_claude_command(
    *,
    settings: ClaudeCodeSettings | None = None,
    input_format: str = "text",
    output_format: str = "json",
) -> list[str]:
    """
    Construct the Claude CLI command with the appropriate flags and optional sandbox-runtime (srt) wrapper.
    
    The command will include input/output format flags and any settings-derived flags and extra CLI arguments. The prompt is not added to the command line (it is passed via stdin). If sandbox runtime is enabled in settings, a temporary srt config file is created, user Claude credentials and settings are copied into a sandbox config directory, and the returned command is wrapped with srt.
    
    Parameters:
        settings (ClaudeCodeSettings | None): Optional settings object or mapping used to influence flags and sandbox behavior. If sandboxing is used, this mapping will be mutated to include a "__sandbox_env" dictionary with environment variables to apply when launching the subprocess.
        input_format (str): Input format to request from the CLI (commonly "text" or "stream-json").
        output_format (str): Output format to request from the CLI (commonly "text", "json", or "stream-json").
    
    Returns:
        list[str]: The list of command arguments to execute. When sandboxing is enabled, this is the srt-wrapped command; otherwise it is the direct Claude CLI command.
    """
    settings = settings or {}
    claude_path = resolve_claude_cli_path(settings)
    cmd = [claude_path, "--print"]

    # Add format flags
    cmd.extend(["--output-format", output_format])
    if input_format != "text":
        cmd.extend(["--input-format", input_format])
    if output_format == "stream-json":
        cmd.append("--include-partial-messages")
        cmd.append("--verbose")  # Required for stream-json output

    # Add settings-based flags
    _add_settings_flags(cmd, settings)

    # Add extra CLI arguments (pass-through for any CLI flags)
    extra_args = settings.get("extra_cli_args")
    if extra_args:
        cmd.extend(extra_args)
        logger.debug("Added %d extra CLI arguments: %s", len(extra_args), extra_args)

    # Do NOT add prompt to command line - it will be passed via stdin
    # This avoids quoting issues when wrapping with sandbox-runtime

    # Wrap with sandbox-runtime if enabled
    if settings.get("use_sandbox_runtime"):
        srt_path = resolve_sandbox_runtime_path(settings)

        # Sandbox config: Allow full /tmp access as required by user
        # "claude code should be able to do any write and read operation in /tmp"
        # Network allowed for Claude API calls to Anthropic
        config = {
            "permissions": {
                "allow": [
                    "Bash(*)",                          # Allow bash (OS sandbox blocks dangerous filesystem ops)
                    "Write(/tmp/**)",                   # Allow writes to /tmp (for outputs, debug logs, etc.)
                    "Read(/tmp/**)",                    # Allow reads from /tmp
                    "Edit(/tmp/**)",                    # Allow edits to /tmp files
                    "WebFetch(domain:api.anthropic.com)",  # Allow API calls to Anthropic (required for Claude to work)
                ]
            }
        }

        # Write config to temp file
        config_fd, config_path = tempfile.mkstemp(suffix=".json", prefix="srt_config_")
        try:
            with os.fdopen(config_fd, 'w') as f:
                json.dump(config, f)

            # Redirect Claude config/debug to /tmp to avoid ~/.claude/ writes
            claude_config_dir = "/tmp/claude_sandbox_config"
            os.makedirs(claude_config_dir, exist_ok=True)

            # Copy OAuth credentials from ~/.claude/ to sandbox config dir
            # This allows Claude to authenticate while keeping debug logs in /tmp
            home_claude_dir = Path.home() / ".claude"
            credentials_file = home_claude_dir / ".credentials.json"
            settings_file = home_claude_dir / "settings.json"

            if credentials_file.exists():
                shutil.copy2(credentials_file, Path(claude_config_dir) / ".credentials.json")
                logger.debug("Copied credentials to sandbox config dir")

            if settings_file.exists():
                shutil.copy2(settings_file, Path(claude_config_dir) / "settings.json")
                logger.debug("Copied settings to sandbox config dir")

            # Build wrapper: srt -- <claude command>
            # Environment variables (IS_SANDBOX=1, CLAUDE_CONFIG_DIR) will be set via subprocess env parameter
            wrapped_cmd = [
                srt_path,
                "--settings", config_path,
                "--",
            ] + cmd

            # Store sandbox env vars in settings so subprocess can use them
            if settings is not None:
                settings["__sandbox_env"] = {
                    "IS_SANDBOX": "1",
                    "CLAUDE_CONFIG_DIR": claude_config_dir,
                }

            logger.info("Wrapped Claude command with sandbox (IS_SANDBOX=1, CLAUDE_CONFIG_DIR=%s)", claude_config_dir)
            logger.debug("Full sandboxed command: %s", " ".join(wrapped_cmd))
            return wrapped_cmd
        except Exception:
            # Clean up config file on error
            try:
                os.unlink(config_path)
            except Exception:
                pass
            raise

    logger.debug("Built Claude command: %s", " ".join(cmd))
    return cmd


def _get_next_call_subdirectory(base_dir: str) -> Path:
    """
    Create and return the next numeric subdirectory under the given base working directory to avoid overwriting previous runs.
    
    Parameters:
        base_dir (str): Base working directory under which a new numeric subdirectory will be created.
    
    Returns:
        Path: Path to the newly created numeric subdirectory (e.g., base_dir/1, base_dir/2).
    """
    base_path = Path(base_dir)
    existing_subdirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.isdigit()]
    next_num = len(existing_subdirs) + 1

    subdir = base_path / str(next_num)
    subdir.mkdir(parents=True, exist_ok=True)
    logger.debug("Created call subdirectory: %s", subdir)

    return subdir


def _copy_additional_files(cwd: str, additional_files: dict[str, Path]) -> None:
    """
    Copy the given additional files into the specified working directory, creating any destination subdirectories as needed.
    
    Parameters:
        cwd (str): Destination working directory path.
        additional_files (dict[str, Path]): Mapping from destination relative path (within cwd) to source Path.
    
    Raises:
        FileNotFoundError: If a source path does not exist.
        ValueError: If a source path exists but is not a regular file.
    """
    for dest_name, source_path in additional_files.items():
        # Resolve relative paths from current working directory
        resolved_source = source_path.resolve()

        if not resolved_source.exists():
            raise FileNotFoundError(
                f"Additional file source not found: {source_path} "
                f"(resolved to {resolved_source})"
            )

        if not resolved_source.is_file():
            raise ValueError(
                f"Additional file source is not a file: {source_path} "
                f"(resolved to {resolved_source})"
            )

        # Create destination path (may include subdirectories)
        dest_path = Path(cwd) / dest_name

        # Create parent directories if needed
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy file (binary mode, preserves permissions and timestamps)
        shutil.copy2(resolved_source, dest_path)

        logger.info("Copied additional file: %s -> %s", source_path, dest_path)


def _determine_working_directory(settings: ClaudeCodeSettings | None) -> str:
    """
    Determine the working directory path to use for a Claude run.
    
    If settings contains a "working_directory" value, ensure that base directory exists and return the path for the next numbered subdirectory (the numeric subdirectory itself is not created). If no working_directory is provided, create and return a new temporary directory (created immediately).
    
    Parameters:
        settings (ClaudeCodeSettings | None): Optional settings object or mapping; the function reads the "working_directory" key if present.
    
    Returns:
        str: Filesystem path to use as the working directory for this run.
    """
    base_dir = settings.get("working_directory") if settings else None

    if not base_dir:
        # Will create temp directory later
        # For now, just return a path that will be created
        return tempfile.mkdtemp(prefix="claude_prompt_")
    else:
        # User-specified directory - will create numbered subdirectory later
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        existing_subdirs = [d for d in Path(base_dir).iterdir() if d.is_dir() and d.name.isdigit()]
        next_num = len(existing_subdirs) + 1
        return str(Path(base_dir) / str(next_num))


def _log_prompt_info(prompt_file: Path, prompt: str) -> None:
    """
    Record debugging details about a prompt and its file location to the logger.
    
    Parameters:
        prompt_file (Path): Filesystem path where the prompt was written.
        prompt (str): The full prompt text; its length and content are logged.
    """
    logger.info("=" * 80)
    logger.info("PROMPT WRITTEN TO: %s", prompt_file)
    logger.info("PROMPT LENGTH: %d chars", len(prompt))
    logger.info("=" * 80)
    logger.info("COMPLETE PROMPT CONTENT:")
    logger.info("=" * 80)
    logger.info("%s", prompt)
    logger.info("=" * 80)


def _setup_working_directory_and_prompt(
    prompt: str, settings: ClaudeCodeSettings | None
) -> str:
    """
    Prepare a per-call working directory, write the prompt to a prompt.md file, and record metadata in settings.
    
    Parameters:
        prompt (str): The prompt text to write into the working directory as "prompt.md".
        settings (ClaudeCodeSettings | None): Optional mutable settings mapping. When provided, this function may:
            - use or create a base working directory,
            - store "__working_directory" (str) as the call-specific directory,
            - store "__response_file_path" (str) pointing to "response.json" inside the call directory,
            - store "__prompt_text" (str) with the prompt content,
            - create and reuse "__temp_base_directory" for temporary sessions,
            - read "additional_files" (dict[str, Path]) to copy into the working directory before writing the prompt.
    
    Returns:
        str: Path to the created or selected working directory for this call.
    """
    # Check if we already determined the working directory for this call
    # (happens when we pre-create tool result files or binary content files)
    existing_working_dir = settings.get("__working_directory") if settings else None
    if existing_working_dir:
        # Use the pre-determined working directory - don't create a new one
        cwd = existing_working_dir
        logger.debug("Using pre-determined working directory: %s", cwd)
        # Ensure it exists
        Path(cwd).mkdir(parents=True, exist_ok=True)
    else:
        # Determine base directory from settings
        base_dir = settings.get("working_directory") if settings else None

        if not base_dir:
            # No working directory specified - create temp directory for this session
            # Check if we already created a temp base directory for these settings
            existing_temp_base = settings.get("__temp_base_directory") if settings else None

            if not existing_temp_base:
                # First call with these settings - create new temp base directory
                existing_temp_base = tempfile.mkdtemp(prefix="claude_prompt_")
                if settings is not None:
                    settings["__temp_base_directory"] = existing_temp_base
                logger.debug("Created temporary base directory: %s", existing_temp_base)

            base_dir = existing_temp_base

        # At this point base_dir is guaranteed to be a string
        assert isinstance(base_dir, str), "base_dir should be a string by now"

        # Always create numbered subdirectory to prevent overwrites across multiple calls
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        cwd_path = _get_next_call_subdirectory(base_dir)
        cwd = str(cwd_path)

    # Copy additional files if specified (before writing prompt.md so they can be referenced)
    additional_files = settings.get("additional_files") if settings else None
    if additional_files:
        _copy_additional_files(cwd, additional_files)

    prompt_file = Path(cwd) / "prompt.md"
    prompt_file.write_text(prompt)
    _log_prompt_info(prompt_file, prompt)

    # Save prompt for debugging if enabled
    _save_prompt_debug(prompt, settings)

    # Store working directory, response filename, and prompt text in settings for later
    if settings is not None:
        settings["__working_directory"] = cwd
        settings["__response_file_path"] = str(Path(cwd) / "response.json")
        settings["__prompt_text"] = prompt  # Store for stdin transmission

    return cwd


def _execute_sync_command(
    cmd: list[str], cwd: str, timeout_seconds: int, settings: ClaudeCodeSettings | None = None
) -> subprocess.CompletedProcess[str]:
    """
    Execute the given CLI command synchronously with a wall-clock timeout and optional sandboxed environment.
    
    Parameters:
        cmd (list[str]): Command and arguments to execute.
        cwd (str): Working directory for the command.
        timeout_seconds (int): Maximum seconds to allow the process to run before raising.
        settings (ClaudeCodeSettings | None): Optional settings object; if it contains a "__sandbox_env" mapping those variables will be merged into the subprocess environment, and if it contains "__prompt_text" that string will be passed to the process via stdin.
    
    Returns:
        subprocess.CompletedProcess[str]: The completed process result containing stdout, stderr, and return code.
    
    Raises:
        RuntimeError: If the command exceeds the specified timeout.
    """
    start_time = time.time()

    # Get sandbox environment variables if present
    env = None
    if settings and settings.get("__sandbox_env"):
        env = os.environ.copy()
        env.update(settings["__sandbox_env"])
        logger.debug("Using sandbox environment: %s", settings["__sandbox_env"])

    try:
        logger.info("Running Claude CLI synchronously in %s", cwd)

        # Get prompt from settings to pass via stdin (avoids quoting issues with srt)
        prompt_input = None
        if settings and settings.get("__prompt_text"):
            prompt_input = settings["__prompt_text"]
            logger.debug("Passing prompt via stdin (%d chars)", len(prompt_input))

        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            cwd=cwd,
            timeout=timeout_seconds,
            env=env,
            input=prompt_input,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        logger.error(
            "Claude CLI timeout after %.1fs (limit: %ds). Working directory: %s",
            elapsed,
            timeout_seconds,
            cwd,
        )
        raise RuntimeError(
            f"Claude CLI timeout after {elapsed:.1f}s (limit: {timeout_seconds}s). "
            f"Task took too long - consider breaking into smaller tasks or increasing timeout_seconds in settings."
        ) from None


def _check_rate_limit(
    stdout_text: str, stderr_text: str, returncode: int, retry_enabled: bool
) -> tuple[bool, int]:
    """
    Determine whether a CLI invocation failed due to a rate limit and, if so, how long to wait before retrying.
    
    Parameters:
        stdout_text (str): Decoded standard output from the process.
        stderr_text (str): Decoded standard error from the process.
        returncode (int): Process exit code.
        retry_enabled (bool): Whether retry behavior is enabled.
    
    Returns:
        tuple[bool, int]: `True` and the number of seconds to wait if a rate limit was detected and retry is enabled; `False, 0` otherwise.
    """
    if returncode != 0 and retry_enabled:
        error_output = stdout_text + "\n" + stderr_text
        is_rate_limited, reset_time = detect_rate_limit(error_output)

        if is_rate_limited and reset_time:
            wait_seconds = calculate_wait_time(reset_time)
            wait_minutes = wait_seconds // 60
            logger.info("Rate limit hit. Waiting %d minutes until reset...", wait_minutes)
            return True, wait_seconds

    return False, 0


def _handle_command_failure(
    stdout_text: str,
    stderr_text: str,
    returncode: int,
    elapsed: float,
    prompt_len: int,
    cwd: str,
) -> None:
    """Handle failed command execution with generic error.

    Note: Specific errors (OAuth, rate limit, infrastructure) should be checked
    before calling this function. This handles remaining generic errors.

    Args:
        stdout_text: Standard output text (decoded)
        stderr_text: Standard error text (decoded)
        returncode: Process return code
        elapsed: Elapsed time in seconds
        prompt_len: Length of prompt
        cwd: Working directory

    Raises:
        RuntimeError: Always raises with appropriate error message
    """
    stderr = stderr_text if stderr_text else "(no error output)"

    logger.error(
        "Claude CLI failed after %.1fs with return code %d\n"
        "Prompt length: %d chars\n"
        "Working dir: %s\n"
        "Stderr: %s\n"
        "Stdout (first 500 chars): %s",
        elapsed,
        returncode,
        prompt_len,
        cwd,
        stderr,
        stdout_text[:500] if stdout_text else "",
    )

    # Generic error handling
    error_msg = _format_cli_error_message(elapsed, returncode, stderr)
    raise RuntimeError(error_msg)


def _parse_json_response(raw_stdout: str) -> ClaudeJSONResponse:
    """
    Parse the Claude CLI JSON response and return the final result event when verbose output is used.
    
    If the stdout begins with a sandbox-runtime diagnostic line ("Running: ..."), that line is removed before parsing. If the parsed JSON is a list (verbose/stream form), the function returns the first object whose `"type"` is `"result"`. If the JSON is a single object, that object is returned.
    
    Parameters:
        raw_stdout (str): Raw stdout from the Claude CLI, possibly including a sandbox-runtime diagnostic line.
    
    Returns:
        ClaudeJSONResponse: The parsed response dictionary (the `"result"` event for verbose/list output, or the parsed object for single-object output).
    
    Raises:
        RuntimeError: If the parsed JSON is a list but no event with `"type": "result"` is found.
    """
    # Strip srt diagnostic output if present (when using sandbox-runtime)
    # srt outputs "Running: <command>" on first line before actual JSON
    if raw_stdout.startswith("Running: "):
        # Skip first line
        first_newline = raw_stdout.find('\n')
        if first_newline > 0:
            raw_stdout = raw_stdout[first_newline + 1:]
            logger.debug("Stripped srt diagnostic line from stdout")

    raw_response = json.loads(raw_stdout)

    if isinstance(raw_response, list):
        logger.debug("Received verbose JSON output with %d events", len(raw_response))
        for event in raw_response:
            if isinstance(event, dict) and event.get("type") == "result":
                return cast(ClaudeJSONResponse, event)

        logger.error("No result event found in verbose output")
        raise RuntimeError("No result event in Claude CLI output")
    else:
        return cast(ClaudeJSONResponse, raw_response)


def _validate_claude_response(response: ClaudeJSONResponse) -> None:
    """Validate response and raise on errors.

    Args:
        response: Response to validate

    Raises:
        RuntimeError: If response contains error
    """
    logger.debug(
        "Received response with %d tokens",
        response.get("usage", {}).get("output_tokens", 0)
        if isinstance(response.get("usage"), dict)
        else 0,
    )

    if response.get("is_error"):
        error_msg = response.get("error", "Unknown error")
        logger.error("Claude CLI returned error: %s", error_msg)
        raise RuntimeError(f"Claude CLI error: {error_msg}")


def _classify_execution_error(
    stdout_text: str,
    stderr_text: str,
    returncode: int,
    elapsed: float,
    retry_enabled: bool,
    cwd: str,
) -> tuple[str, float]:
    """
    Classifies a Claude CLI execution failure and determines whether to retry or raise an error.
    
    Parameters:
        stdout_text (str): Captured standard output from the command.
        stderr_text (str): Captured standard error from the command.
        returncode (int): Process exit code.
        elapsed (float): Elapsed execution time in seconds.
        retry_enabled (bool): Whether automatic rate-limit retries are permitted.
        cwd (str): Working directory used when the command was executed (used for context in errors).
    
    Returns:
        tuple[str, float]: A pair (action, wait_seconds).
            - action is "retry_rate_limit" to indicate a rate-limit retry, or "retry_infra" to indicate a transient infrastructure retry.
            - wait_seconds is the number of seconds to wait before retrying (non-zero only for rate-limit retries; otherwise 0.0).
    
    Raises:
        ClaudeOAuthError: If the output indicates an OAuth/authentication error that requires reauthentication.
        RuntimeError: For other non-retriable command failures.
    """
    # Priority 1: Check OAuth errors first (most specific)
    is_oauth_error, oauth_message = detect_oauth_error(stdout_text, stderr_text)
    if is_oauth_error and oauth_message:
        raise ClaudeOAuthError(
            f"Claude CLI authentication expired after {elapsed:.1f}s: {oauth_message}",
            reauth_instruction=oauth_message if "/login" in oauth_message else "Please run /login"
        )

    # Priority 2: Check rate limit (less specific, could have false positives)
    should_retry, wait_seconds = _check_rate_limit(stdout_text, stderr_text, returncode, retry_enabled)
    if should_retry:
        return ("retry_rate_limit", wait_seconds)

    # Priority 3: Check for infrastructure failures
    if detect_cli_infrastructure_failure(stderr_text):
        return ("retry_infra", 0.0)

    # Priority 4: Generic error handling (raises exception)
    _handle_command_failure(stdout_text, stderr_text, returncode, elapsed, 0, cwd)
    # If we get here, _handle_command_failure should have raised an exception
    raise RuntimeError("Unexpected: _handle_command_failure should have raised")


def _process_successful_response(
    stdout_text: str,
    settings: ClaudeCodeSettings | None = None,
) -> ClaudeJSONResponse:
    """
    Parse, validate, and persist a successful Claude CLI JSON response.
    
    Parameters:
        stdout_text (str): Raw stdout text produced by the Claude CLI.
        settings (ClaudeCodeSettings | None): Optional settings used to determine where the raw response is saved.
    
    Returns:
        ClaudeJSONResponse: The parsed and validated Claude JSON response.
    """
    response = _parse_json_response(stdout_text)
    _validate_claude_response(response)
    _save_raw_response_to_working_dir(response, settings)
    return response


def _try_sync_execution_with_rate_limit_retry(
    cmd: list[str],
    cwd: str,
    timeout_seconds: int,
    retry_enabled: bool,
    settings: ClaudeCodeSettings | None = None,
) -> tuple[ClaudeJSONResponse | None, bool]:
    """
    Execute a Claude CLI command synchronously, retrying when a rate-limit reset is detected.
    
    This function runs the provided command in a loop: on a non-zero exit it classifies the failure and
    will sleep and retry if the error is a rate-limit; it will signal an infrastructure retry when the
    error classification indicates a transient CLI infrastructure failure. On success it parses and
    returns the CLI JSON response.
    
    Returns:
        (response, should_retry_infra): `response` is the parsed ClaudeJSONResponse when the command
        succeeds, or `None` when no successful response was produced. `should_retry_infra` is `True`
        when the caller should retry the overall operation due to a transient infrastructure failure,
        `False` otherwise.
    """
    while True:
        start_time = time.time()
        result = _execute_sync_command(cmd, cwd, timeout_seconds, settings)
        elapsed = time.time() - start_time

        # Check for errors if command failed
        if result.returncode != 0:
            stdout_text = result.stdout if result.stdout else ""
            stderr_text = result.stderr if result.stderr else ""

            # Classify error and get action (may raise exception)
            action, wait_seconds = _classify_execution_error(
                stdout_text, stderr_text, result.returncode, elapsed, retry_enabled, cwd
            )

            if action == "retry_rate_limit":
                time.sleep(int(wait_seconds))
                logger.info("Wait complete, retrying...")
                continue
            elif action == "retry_infra":
                return None, True

        # Success - process and return
        response = _process_successful_response(result.stdout, settings)
        return response, False


def run_claude_sync(
    prompt: str,
    *,
    settings: ClaudeCodeSettings | None = None,
) -> ClaudeJSONResponse:
    """
    Execute the Claude CLI with the provided prompt and return the parsed Claude JSON response.
    
    The call may retry on rate limits and transient CLI infrastructure failures according to settings.
    
    Parameters:
        prompt (str): Prompt text to send to Claude.
        settings (ClaudeCodeSettings | None): Optional execution settings (e.g., timeouts, retry behavior, sandboxing).
    
    Returns:
        ClaudeJSONResponse: Parsed JSON response produced by the Claude CLI.
    
    Raises:
        RuntimeError: When the CLI fails after retry attempts or a persistent infrastructure failure occurs.
        ClaudeOAuthError: If an authentication/OAuth error is detected that requires reauthorization.
        json.JSONDecodeError: If the CLI output cannot be parsed as valid JSON.
    """
    retry_enabled = settings.get("retry_on_rate_limit", True) if settings else True
    timeout_seconds = settings.get("timeout_seconds", 900) if settings else 900

    cwd = _setup_working_directory_and_prompt(prompt, settings)
    cmd = build_claude_command(settings=settings, output_format="json")

    # Outer retry loop for infrastructure failures
    for attempt in range(MAX_CLI_RETRIES):
        try:
            response, should_retry_infra = _try_sync_execution_with_rate_limit_retry(
                cmd, cwd, timeout_seconds, retry_enabled, settings
            )
            if response:
                _save_response_debug(response, settings)
                return response

            # Infrastructure failure detected
            if should_retry_infra and attempt < MAX_CLI_RETRIES - 1:
                backoff_seconds = RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    "Claude CLI infrastructure failure detected (attempt %d/%d). "
                    "Retrying in %d seconds...",
                    attempt + 1,
                    MAX_CLI_RETRIES,
                    backoff_seconds,
                )
                time.sleep(backoff_seconds)
                continue
            elif should_retry_infra:
                # Final attempt failed
                logger.error(
                    "Claude CLI infrastructure failure persisted after %d attempts",
                    MAX_CLI_RETRIES,
                )
                raise RuntimeError("Claude CLI infrastructure failure persisted")

        except RuntimeError as e:
            # If this is the last attempt or not an infrastructure error, re-raise
            if attempt >= MAX_CLI_RETRIES - 1:
                raise
            # Check if the error is due to infrastructure failure
            error_str = str(e)
            if detect_cli_infrastructure_failure(error_str):
                backoff_seconds = RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    "Claude CLI infrastructure failure in execution (attempt %d/%d). "
                    "Retrying in %d seconds...",
                    attempt + 1,
                    MAX_CLI_RETRIES,
                    backoff_seconds,
                )
                time.sleep(backoff_seconds)
                continue
            # Not an infrastructure error - re-raise immediately
            raise

    # Should never reach here, but just in case
    raise RuntimeError("Claude CLI failed after maximum retry attempts")


async def _execute_async_command(
    cmd: list[str], cwd: str, timeout_seconds: int, settings: ClaudeCodeSettings | None = None
) -> tuple[bytes, bytes, int]:
    """
    Run a subprocess command within a working directory, optionally using sandbox environment and prompt from settings, and enforce a timeout.
    
    Parameters:
        cmd (list[str]): Command and arguments to execute.
        cwd (str): Working directory for the subprocess.
        timeout_seconds (int): Number of seconds to wait before timing out.
        settings (ClaudeCodeSettings | None): Optional settings; when provided, may supply a "__sandbox_env" mapping to use as environment variables and a "__prompt_text" string that will be passed to the process via stdin.
    
    Returns:
        tuple[bytes, bytes, int]: A tuple containing (stdout bytes, stderr bytes, return code).
    
    Raises:
        RuntimeError: If the command does not complete within timeout_seconds.
    """
    start_time = time.time()
    logger.info("Running Claude CLI asynchronously in %s", cwd)

    # Get sandbox environment variables if present
    env = None
    if settings and settings.get("__sandbox_env"):
        env = os.environ.copy()
        env.update(settings["__sandbox_env"])
        logger.debug("Using sandbox environment: %s", settings["__sandbox_env"])

    process = await create_subprocess_async(cmd, cwd, env)

    try:
        # Get prompt from settings to pass via stdin (avoids quoting issues with srt)
        prompt_input = None
        if settings and settings.get("__prompt_text"):
            prompt_input = settings["__prompt_text"].encode('utf-8')
            logger.debug("Passing prompt via stdin (%d chars)", len(settings["__prompt_text"]))

        stdout, stderr = await asyncio.wait_for(
            process.communicate(input=prompt_input),
            timeout=timeout_seconds,
        )
        return stdout, stderr, process.returncode or 0

    except asyncio.TimeoutError:
        try:
            process.kill()
            await process.wait()
        except Exception:
            pass

        elapsed = time.time() - start_time
        logger.error(
            "Claude CLI timeout after %.1fs (limit: %ds). Working directory: %s",
            elapsed,
            timeout_seconds,
            cwd,
        )
        raise RuntimeError(
            f"Claude CLI timeout after {elapsed:.1f}s (limit: {timeout_seconds}s). "
            f"Task took too long - consider breaking into smaller tasks or increasing timeout_seconds in settings."
        ) from None


async def _try_async_execution_with_rate_limit_retry(
    cmd: list[str],
    cwd: str,
    timeout_seconds: int,
    retry_enabled: bool,
    settings: ClaudeCodeSettings | None = None,
) -> tuple[ClaudeJSONResponse | None, bool]:
    """
    Execute a Claude CLI command asynchronously, automatically retrying when a rate-limit reset is detected.
    
    This function runs the command once per loop, classifies failures, and:
    - waits and retries when a rate limit is detected,
    - returns (None, True) to signal an infrastructure retry when a transient CLI infrastructure failure is detected,
    - on success parses and returns the Claude JSON response.
    
    Parameters:
        retry_enabled (bool): If True, rate-limit errors will trigger automatic wait-and-retry behavior; if False, rate-limit errors will not be retried.
    
    Returns:
        tuple[ClaudeJSONResponse | None, bool]: A pair where the first element is the parsed Claude response on success (or None if an infrastructure retry is required),
        and the second element is True when the caller should perform an infrastructure retry, False otherwise.
    
    Raises:
        ClaudeOAuthError: If an OAuth/authentication error is detected and reauthentication is required.
        RuntimeError: For other unrecoverable CLI errors.
    """
    while True:
        start_time = time.time()
        process_output = await _execute_async_command(cmd, cwd, timeout_seconds, settings)
        elapsed = time.time() - start_time
        stdout, stderr, returncode = process_output

        # Check for errors if command failed
        if returncode != 0:
            stdout_text = stdout.decode() if stdout else ""
            stderr_text = stderr.decode() if stderr else ""

            # Classify error and get action (may raise exception)
            action, wait_seconds = _classify_execution_error(
                stdout_text, stderr_text, returncode, elapsed, retry_enabled, cwd
            )

            if action == "retry_rate_limit":
                await asyncio.sleep(int(wait_seconds))
                logger.info("Wait complete, retrying...")
                continue
            elif action == "retry_infra":
                return None, True

        # Success - process and return
        response = _process_successful_response(stdout.decode(), settings)
        return response, False


async def run_claude_async(
    prompt: str,
    *,
    settings: ClaudeCodeSettings | None = None,
) -> ClaudeJSONResponse:
    """
    Run the Claude CLI with the given prompt and return the parsed JSON response.
    
    This starts a CLI invocation (possibly wrapped in a sandbox if configured), handles rate-limit and transient infrastructure retries according to settings, and saves debug output when enabled. The call writes the prompt and auxiliary files to a working directory and may populate working-directory related fields on `settings`.
    
    Parameters:
        prompt (str): The prompt text to send to Claude.
        settings (ClaudeCodeSettings | None): Optional settings that influence execution. Recognized keys include `retry_on_rate_limit` (default True) and `timeout_seconds` (default 900). The function may also add working-directory metadata to this object.
    
    Returns:
        ClaudeJSONResponse: The parsed JSON response produced by the Claude CLI.
    
    Raises:
        RuntimeError: If the CLI repeatedly fails due to persistent infrastructure problems or the maximum retry attempts are exhausted.
    """
    retry_enabled = settings.get("retry_on_rate_limit", True) if settings else True
    timeout_seconds = settings.get("timeout_seconds", 900) if settings else 900

    cwd = _setup_working_directory_and_prompt(prompt, settings)
    cmd = build_claude_command(settings=settings, output_format="json")

    # Outer retry loop for infrastructure failures
    for attempt in range(MAX_CLI_RETRIES):
        try:
            response, should_retry_infra = await _try_async_execution_with_rate_limit_retry(
                cmd, cwd, timeout_seconds, retry_enabled, settings
            )
            if response:
                _save_response_debug(response, settings)
                return response

            # Infrastructure failure detected
            if should_retry_infra and attempt < MAX_CLI_RETRIES - 1:
                backoff_seconds = RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    "Claude CLI infrastructure failure detected (attempt %d/%d). "
                    "Retrying in %d seconds...",
                    attempt + 1,
                    MAX_CLI_RETRIES,
                    backoff_seconds,
                )
                await asyncio.sleep(backoff_seconds)
                continue
            elif should_retry_infra:
                # Final attempt failed
                logger.error(
                    "Claude CLI infrastructure failure persisted after %d attempts",
                    MAX_CLI_RETRIES,
                )
                raise RuntimeError("Claude CLI infrastructure failure persisted")

        except RuntimeError as e:
            # If this is the last attempt or not an infrastructure error, re-raise
            if attempt >= MAX_CLI_RETRIES - 1:
                raise
            # Check if the error is due to infrastructure failure
            error_str = str(e)
            if detect_cli_infrastructure_failure(error_str):
                backoff_seconds = RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    "Claude CLI infrastructure failure in execution (attempt %d/%d). "
                    "Retrying in %d seconds...",
                    attempt + 1,
                    MAX_CLI_RETRIES,
                    backoff_seconds,
                )
                await asyncio.sleep(backoff_seconds)
                continue
            # Not an infrastructure error - re-raise immediately
            raise

    # Should never reach here, but just in case
    raise RuntimeError("Claude CLI failed after maximum retry attempts")


def parse_stream_json_line(line: str) -> ClaudeStreamEvent | None:
    """
    Parse a single line of stream-json output into a ClaudeStreamEvent.
    
    Parameters:
        line (str): One line from the stream-json output.
    
    Returns:
        ClaudeStreamEvent | None: The parsed event if the line is valid JSON, `None` if the line is empty or cannot be parsed.
    """
    line = line.strip()
    if not line:
        return None

    try:
        event = json.loads(line)
        if event.get("type"):
            logger.debug("Parsed stream event: type=%s", event["type"])
        return cast(ClaudeStreamEvent, event)
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse stream JSON line: %s", e)
        return None


# Counter for sequential numbering of prompts/responses
_debug_counter = 0


def _get_debug_dir(settings: ClaudeCodeSettings | None) -> Path | None:
    """
    Return the debug directory path when prompt/response debug saving is enabled.
    
    Parameters:
        settings (ClaudeCodeSettings | None): Mapping-like settings where the key "debug_save_prompts"
            controls debug saving. If the value is True, the default directory /tmp/claude_debug is used.
            If the value is a string or path-like, that path is used. Any other falsy value disables debug saving.
    
    Returns:
        Path | None: A Path to the ensured debug directory when enabled, or `None` if debug saving is disabled
    """
    if not settings:
        return None

    debug_setting = settings.get("debug_save_prompts")
    if not debug_setting:
        return None

    if debug_setting is True:
        debug_dir = Path("/tmp/claude_debug")
    else:
        debug_dir = Path(str(debug_setting))

    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir


def _save_prompt_debug(prompt: str, settings: ClaudeCodeSettings | None) -> None:
    """
    Save the prompt to a timestamped debug file when debug saving is enabled.
    
    If debug saving is enabled in `settings`, writes the prompt to the debug directory determined by settings using a filename of the form `<counter>_<YYYYMMDD_HHMMSS>_prompt.md`. Does nothing if debug saving is disabled or no debug directory is available.
    
    Parameters:
        prompt (str): The prompt text to save.
        settings (ClaudeCodeSettings | None): Optional settings object that controls debug saving and debug directory location.
    """
    debug_dir = _get_debug_dir(settings)
    if not debug_dir:
        return

    global _debug_counter
    _debug_counter += 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{_debug_counter:03d}_{timestamp}_prompt.md"
    filepath = debug_dir / filename

    filepath.write_text(prompt)
    logger.info("Saved prompt to: %s", filepath)


def _save_response_debug(response: ClaudeJSONResponse, settings: ClaudeCodeSettings | None) -> None:
    """
    Save the parsed Claude JSON response to a timestamped debug file when debug saving is enabled.
    
    Parameters:
        response (ClaudeJSONResponse): The parsed Claude response to write to disk.
        settings (ClaudeCodeSettings | None): Settings that control debug saving; if `None` or debug saving is disabled, no file is written.
    """
    debug_dir = _get_debug_dir(settings)
    if not debug_dir:
        return

    global _debug_counter

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{_debug_counter:03d}_{timestamp}_response.json"
    filepath = debug_dir / filename

    filepath.write_text(json.dumps(response, indent=2))
    logger.info("Saved response to: %s", filepath)


def _save_raw_response_to_working_dir(
    response: ClaudeJSONResponse, settings: ClaudeCodeSettings | None
) -> None:
    """
    Save the parsed Claude JSON response to the response file configured for the current working directory.
    
    If `settings` is None or does not contain a "__response_file_path" entry, the function returns without action. When a path is present, the response is written as pretty-printed JSON to that file. Failures during writing are caught and logged; no exception is raised.
    
    Parameters:
        response: The parsed Claude JSON response to persist.
        settings: A mapping-like settings object expected to contain "__response_file_path" with the destination file path.
    """
    if not settings:
        return

    response_file = settings.get("__response_file_path")
    if not response_file:
        logger.debug("No response file path configured, skipping save")
        return

    try:
        response_path = Path(response_file)
        response_path.write_text(json.dumps(response, indent=2))
        logger.info("Saved raw response to: %s", response_path)
    except Exception as e:
        logger.warning("Failed to save raw response to working directory: %s", e)