"""Utility functions for Claude Code model."""

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


def resolve_claude_cli_path(settings: ClaudeCodeSettings | None = None) -> str:
    """Resolve path to Claude CLI binary.

    Resolution priority:
    1. claude_cli_path from settings (if provided)
    2. CLAUDE_CLI_PATH environment variable
    3. shutil.which('claude') - auto-resolve from PATH

    Args:
        settings: Optional settings containing claude_cli_path

    Returns:
        Path to claude CLI binary

    Raises:
        RuntimeError: If claude CLI cannot be found
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


def detect_rate_limit(error_output: str) -> tuple[bool, str | None]:
    """Detect rate limit error and extract reset time.

    Args:
        error_output: Combined stdout + stderr from Claude CLI

    Returns:
        Tuple of (is_rate_limited, reset_time_str)
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
    """Calculate seconds to wait until reset time.

    Args:
        reset_time_str: Time string like "3PM" or "11AM"

    Returns:
        Seconds to wait (with 1-minute buffer)
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
    """Detect transient Claude CLI infrastructure failures that should trigger retry.

    Args:
        stderr: Error output from Claude CLI

    Returns:
        True if error indicates retryable infrastructure failure
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
    """Add tool permission flags to command."""
    allowed_tools = settings.get("allowed_tools")
    if allowed_tools:
        cmd.append("--allowed-tools")
        cmd.extend(allowed_tools)

    disallowed_tools = settings.get("disallowed_tools")
    if disallowed_tools:
        cmd.append("--disallowed-tools")
        cmd.extend(disallowed_tools)


def _add_model_flags(cmd: list[str], settings: ClaudeCodeSettings) -> None:
    """Add model-related flags to command."""
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
    """Add settings-based flags to command.

    Args:
        cmd: Command list to modify
        settings: Settings dict
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
    """Build Claude CLI command with appropriate flags.

    Args:
        settings: Optional settings for Claude Code execution
        input_format: Input format ('text' or 'stream-json')
        output_format: Output format ('text', 'json', or 'stream-json')

    Returns:
        List of command arguments
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

    # Add prompt reference
    cmd.append("Follow the instructions in prompt.md")

    logger.debug("Built Claude command: %s", " ".join(cmd))
    return cmd


def _get_next_call_subdirectory(base_dir: str) -> Path:
    """Get next numbered subdirectory for this CLI call to avoid overwrites.

    Args:
        base_dir: Base working directory

    Returns:
        Path to numbered subdirectory (e.g., base_dir/1/, base_dir/2/, etc.)
    """
    base_path = Path(base_dir)
    existing_subdirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.isdigit()]
    next_num = len(existing_subdirs) + 1

    subdir = base_path / str(next_num)
    subdir.mkdir(parents=True, exist_ok=True)
    logger.debug("Created call subdirectory: %s", subdir)

    return subdir


def _copy_additional_files(cwd: str, additional_files: dict[str, Path]) -> None:
    """Copy additional files into working directory.

    Args:
        cwd: Working directory path
        additional_files: Dict mapping destination filename to source Path

    Raises:
        FileNotFoundError: If source file doesn't exist
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


def _setup_working_directory_and_prompt(
    prompt: str, settings: ClaudeCodeSettings | None
) -> str:
    """Setup working directory and write prompt file.

    Args:
        prompt: The prompt text
        settings: Optional settings

    Returns:
        Working directory path (including call subdirectory)
    """
    base_dir = settings.get("working_directory") if settings else None

    if not base_dir:
        # Create temp directory - this will be the only call, so no subdirectory needed
        cwd = tempfile.mkdtemp(prefix="claude_prompt_")
        logger.debug("Created temporary working directory: %s", cwd)
    else:
        # User-specified directory - create numbered subdirectory to avoid overwrites
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        cwd_path = _get_next_call_subdirectory(base_dir)
        cwd = str(cwd_path)

    # Copy additional files if specified (before writing prompt.md so they can be referenced)
    additional_files = settings.get("additional_files") if settings else None
    if additional_files:
        _copy_additional_files(cwd, additional_files)

    prompt_file = Path(cwd) / "prompt.md"
    prompt_file.write_text(prompt)
    logger.info("=" * 80)
    logger.info("PROMPT WRITTEN TO: %s", prompt_file)
    logger.info("PROMPT LENGTH: %d chars", len(prompt))
    logger.info("=" * 80)
    logger.info("COMPLETE PROMPT CONTENT:")
    logger.info("=" * 80)
    logger.info("%s", prompt)
    logger.info("=" * 80)

    # Save prompt for debugging if enabled
    _save_prompt_debug(prompt, settings)

    # Store response filename in settings for later
    if settings is not None:
        settings["__response_file_path"] = str(Path(cwd) / "response.json")

    return cwd


def _execute_sync_command(
    cmd: list[str], cwd: str, timeout_seconds: int
) -> subprocess.CompletedProcess[str]:
    """Execute command synchronously with timeout.

    Args:
        cmd: Command to execute
        cwd: Working directory
        timeout_seconds: Timeout in seconds

    Returns:
        Completed process result

    Raises:
        RuntimeError: On timeout
    """
    start_time = time.time()

    try:
        logger.info("Running Claude CLI synchronously in %s", cwd)
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            cwd=cwd,
            timeout=timeout_seconds,
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


def _check_rate_limit_and_retry(
    result: subprocess.CompletedProcess[str], retry_enabled: bool
) -> tuple[bool, int]:
    """Check if command hit rate limit and should retry.

    Args:
        result: Subprocess result
        retry_enabled: Whether retry is enabled

    Returns:
        Tuple of (should_retry, wait_seconds)
    """
    if result.returncode != 0 and retry_enabled:
        error_output = result.stdout + "\n" + result.stderr
        is_rate_limited, reset_time = detect_rate_limit(error_output)

        if is_rate_limited and reset_time:
            wait_seconds = calculate_wait_time(reset_time)
            wait_minutes = wait_seconds // 60
            logger.info("Rate limit hit. Waiting %d minutes until reset...", wait_minutes)
            return True, wait_seconds

    return False, 0


def _handle_sync_command_failure(
    result: subprocess.CompletedProcess[str], elapsed: float, prompt_len: int, cwd: str
) -> None:
    """Handle failed command execution with generic error.

    Note: Specific errors (OAuth, rate limit, infrastructure) should be checked
    before calling this function. This handles remaining generic errors.

    Args:
        result: Failed subprocess result
        elapsed: Elapsed time in seconds
        prompt_len: Length of prompt
        cwd: Working directory

    Raises:
        RuntimeError: Always raises with appropriate error message
    """
    stderr_text = result.stderr if result.stderr else "(no error output)"
    stdout_text = result.stdout if result.stdout else ""

    logger.error(
        "Claude CLI failed after %.1fs with return code %d\n"
        "Prompt length: %d chars\n"
        "Working dir: %s\n"
        "Stderr: %s\n"
        "Stdout (first 500 chars): %s",
        elapsed,
        result.returncode,
        prompt_len,
        cwd,
        stderr_text,
        stdout_text[:500],
    )

    # Generic error handling
    if elapsed > LONG_RUNTIME_THRESHOLD_SECONDS:
        raise RuntimeError(
            f"Claude CLI failed after {elapsed:.1f}s with return code {result.returncode}: {stderr_text}\n"
            f"Long runtime suggests task complexity - consider breaking into smaller tasks."
        )
    else:
        raise RuntimeError(f"Claude CLI error after {elapsed:.1f}s: {stderr_text}")


def _parse_sync_json_response(raw_stdout: str) -> ClaudeJSONResponse:
    """Parse JSON response from Claude CLI output.

    Args:
        raw_stdout: Raw stdout from CLI

    Returns:
        Parsed response

    Raises:
        RuntimeError: If no result event found
    """
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


def _try_sync_execution_with_rate_limit_retry(
    cmd: list[str],
    cwd: str,
    timeout_seconds: int,
    retry_enabled: bool,
    settings: ClaudeCodeSettings | None = None,
) -> tuple[ClaudeJSONResponse | None, bool]:
    """Try command execution with rate limit retry.

    Error detection priority (most specific to least specific):
    1. OAuth errors (requires JSON + specific keywords) - raise immediately
    2. Rate limit errors (regex pattern) - wait and retry
    3. Infrastructure failures (stderr patterns) - signal retry
    4. Generic errors - raise

    Returns:
        Tuple of (response if successful or None, should_retry_infra)
    """
    while True:
        start_time = time.time()
        result = _execute_sync_command(cmd, cwd, timeout_seconds)
        elapsed = time.time() - start_time

        # Check for errors if command failed
        if result.returncode != 0:
            stdout_text = result.stdout if result.stdout else ""
            stderr_text = result.stderr if result.stderr else ""

            # Priority 1: Check OAuth errors first (most specific)
            is_oauth_error, oauth_message = detect_oauth_error(stdout_text, stderr_text)
            if is_oauth_error and oauth_message:
                raise ClaudeOAuthError(
                    f"Claude CLI authentication expired after {elapsed:.1f}s: {oauth_message}",
                    reauth_instruction=oauth_message if "/login" in oauth_message else "Please run /login"
                )

            # Priority 2: Check rate limit (less specific, could have false positives)
            should_retry, wait_seconds = _check_rate_limit_and_retry(result, retry_enabled)
            if should_retry:
                time.sleep(wait_seconds)
                logger.info("Wait complete, retrying...")
                continue

            # Priority 3: Check for infrastructure failures
            if detect_cli_infrastructure_failure(stderr_text):
                # Signal to retry in outer loop
                return None, True

            # Priority 4: Generic error handling
            _handle_sync_command_failure(result, elapsed, 0, cwd)

        # Success - parse and return
        response = _parse_sync_json_response(result.stdout)
        _validate_claude_response(response)
        _save_raw_response_to_working_dir(response, settings)
        return response, False


def run_claude_sync(
    prompt: str,
    *,
    settings: ClaudeCodeSettings | None = None,
) -> ClaudeJSONResponse:
    """Run Claude CLI synchronously and return JSON response.

    Automatically retries on rate limit if retry_on_rate_limit is True (default).
    Also retries on transient CLI infrastructure failures (e.g., missing modules).

    Args:
        prompt: The prompt to send to Claude
        settings: Optional settings for Claude Code execution

    Returns:
        Claude JSON response

    Raises:
        subprocess.CalledProcessError: If Claude CLI fails
        json.JSONDecodeError: If response is not valid JSON
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
    cmd: list[str], cwd: str, timeout_seconds: int
) -> tuple[bytes, bytes, int]:
    """Execute command asynchronously with timeout.

    Args:
        cmd: Command to execute
        cwd: Working directory
        timeout_seconds: Timeout in seconds

    Returns:
        Tuple of (stdout, stderr, returncode)

    Raises:
        RuntimeError: On timeout
    """
    import asyncio

    start_time = time.time()
    logger.info("Running Claude CLI asynchronously in %s", cwd)

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.DEVNULL,
        cwd=cwd,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
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


async def _check_rate_limit_and_retry_async(
    stdout: bytes, stderr: bytes, returncode: int, retry_enabled: bool
) -> tuple[bool, int]:
    """Check if command hit rate limit and should retry (async version).

    Args:
        stdout: Process stdout
        stderr: Process stderr
        returncode: Process return code
        retry_enabled: Whether retry is enabled

    Returns:
        Tuple of (should_retry, wait_seconds)
    """
    if returncode != 0 and retry_enabled:
        error_output = stdout.decode() + "\n" + stderr.decode()
        is_rate_limited, reset_time = detect_rate_limit(error_output)

        if is_rate_limited and reset_time:
            wait_seconds = calculate_wait_time(reset_time)
            wait_minutes = wait_seconds // 60
            logger.info("Rate limit hit. Waiting %d minutes until reset...", wait_minutes)
            return True, wait_seconds

    return False, 0


def _handle_async_command_failure(
    process_output: tuple[bytes, bytes, int], elapsed: float, prompt_len: int, cwd: str
) -> None:
    """Handle failed async command execution with generic error.

    Note: Specific errors (OAuth, rate limit, infrastructure) should be checked
    before calling this function. This handles remaining generic errors.

    Args:
        process_output: Tuple of (stdout, stderr, returncode) from process
        elapsed: Elapsed time in seconds
        prompt_len: Length of prompt
        cwd: Working directory

    Raises:
        RuntimeError: Always raises with appropriate error message
    """
    stdout, stderr, returncode = process_output
    stderr_text = stderr.decode() if stderr else "(no error output)"
    stdout_text = stdout.decode() if stdout else ""

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
        stderr_text,
        stdout_text[:500],
    )

    # Generic error handling
    if elapsed > LONG_RUNTIME_THRESHOLD_SECONDS:
        raise RuntimeError(
            f"Claude CLI failed after {elapsed:.1f}s with return code {returncode}: {stderr_text}\n"
            f"Long runtime suggests task complexity - consider breaking into smaller tasks."
        )
    else:
        raise RuntimeError(f"Claude CLI error after {elapsed:.1f}s: {stderr_text}")


async def _try_async_execution_with_rate_limit_retry(
    cmd: list[str],
    cwd: str,
    timeout_seconds: int,
    retry_enabled: bool,
    settings: ClaudeCodeSettings | None = None,
) -> tuple[ClaudeJSONResponse | None, bool]:
    """Try async command execution with rate limit retry.

    Error detection priority (most specific to least specific):
    1. OAuth errors (requires JSON + specific keywords) - raise immediately
    2. Rate limit errors (regex pattern) - wait and retry
    3. Infrastructure failures (stderr patterns) - signal retry
    4. Generic errors - raise

    Returns:
        Tuple of (response if successful or None, should_retry_infra)
    """
    import asyncio

    while True:
        start_time = time.time()
        process_output = await _execute_async_command(cmd, cwd, timeout_seconds)
        elapsed = time.time() - start_time
        stdout, stderr, returncode = process_output

        # Check for errors if command failed
        if returncode != 0:
            stdout_text = stdout.decode() if stdout else ""
            stderr_text = stderr.decode() if stderr else ""

            # Priority 1: Check OAuth errors first (most specific)
            is_oauth_error, oauth_message = detect_oauth_error(stdout_text, stderr_text)
            if is_oauth_error and oauth_message:
                raise ClaudeOAuthError(
                    f"Claude CLI authentication expired after {elapsed:.1f}s: {oauth_message}",
                    reauth_instruction=oauth_message if "/login" in oauth_message else "Please run /login"
                )

            # Priority 2: Check rate limit (less specific, could have false positives)
            should_retry, wait_seconds = await _check_rate_limit_and_retry_async(
                stdout, stderr, returncode, retry_enabled
            )
            if should_retry:
                await asyncio.sleep(wait_seconds)
                logger.info("Wait complete, retrying...")
                continue

            # Priority 3: Check for infrastructure failures
            if detect_cli_infrastructure_failure(stderr_text):
                # Signal to retry in outer loop
                return None, True

            # Priority 4: Generic error handling
            _handle_async_command_failure(process_output, elapsed, 0, cwd)

        # Success - parse and return
        response = _parse_sync_json_response(stdout.decode())
        _validate_claude_response(response)
        _save_raw_response_to_working_dir(response, settings)
        return response, False


async def run_claude_async(
    prompt: str,
    *,
    settings: ClaudeCodeSettings | None = None,
) -> ClaudeJSONResponse:
    """Run Claude CLI asynchronously and return JSON response.

    Automatically retries on rate limit if retry_on_rate_limit is True (default).
    Also retries on transient CLI infrastructure failures (e.g., missing modules).

    Args:
        prompt: The prompt to send to Claude
        settings: Optional settings for Claude Code execution

    Returns:
        Claude JSON response

    Raises:
        subprocess.CalledProcessError: If Claude CLI fails
        json.JSONDecodeError: If response is not valid JSON
    """
    import asyncio

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
    """Parse a single line of stream-json output.

    Args:
        line: A line of JSON output

    Returns:
        Parsed event or None if line is empty or invalid
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
    """Get debug directory path if debug saving is enabled.

    Args:
        settings: Settings dict

    Returns:
        Path to debug directory or None if disabled
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
    """Save prompt to debug file if enabled.

    Args:
        prompt: Prompt text to save
        settings: Settings dict
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
    """Save response to debug file if enabled.

    Args:
        response: Claude response to save
        settings: Settings dict
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
    """Save raw response to working directory (always-on feature).

    Args:
        response: Claude response to save
        settings: Settings dict containing __response_file_path
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
