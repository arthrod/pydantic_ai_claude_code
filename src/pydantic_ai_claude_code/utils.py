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

from .types import ClaudeCodeSettings, ClaudeJSONResponse, ClaudeStreamEvent

logger = logging.getLogger(__name__)

# Constants
LONG_RUNTIME_THRESHOLD_SECONDS = 600  # 10 minutes threshold for long runtime warnings


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
        cli_path = settings["claude_cli_path"]
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


def _setup_working_directory_and_prompt(
    prompt: str, settings: ClaudeCodeSettings | None
) -> str:
    """Setup working directory and write prompt file.

    Args:
        prompt: The prompt text
        settings: Optional settings

    Returns:
        Working directory path
    """
    cwd = settings.get("working_directory") if settings else None

    if not cwd:
        cwd = tempfile.mkdtemp(prefix="claude_prompt_")
        logger.debug("Created temporary working directory: %s", cwd)

    Path(cwd).mkdir(parents=True, exist_ok=True)

    prompt_file = Path(cwd) / "prompt.md"
    prompt_file.write_text(prompt)
    logger.debug("Wrote prompt (%d chars) to %s", len(prompt), prompt_file)

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
    """Handle failed command execution.

    Args:
        result: Failed subprocess result
        elapsed: Elapsed time in seconds
        prompt_len: Length of prompt
        cwd: Working directory

    Raises:
        RuntimeError: Always raises with appropriate error message
    """
    stderr_text = result.stderr if result.stderr else "(no error output)"
    stdout_text = result.stdout[:500] if result.stdout else "(no stdout)"

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
        stdout_text,
    )

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


def run_claude_sync(
    prompt: str,
    *,
    settings: ClaudeCodeSettings | None = None,
) -> ClaudeJSONResponse:
    """Run Claude CLI synchronously and return JSON response.

    Automatically retries on rate limit if retry_on_rate_limit is True (default).

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

    while True:
        start_time = time.time()
        result = _execute_sync_command(cmd, cwd, timeout_seconds)
        elapsed = time.time() - start_time

        should_retry, wait_seconds = _check_rate_limit_and_retry(result, retry_enabled)
        if should_retry:
            time.sleep(wait_seconds)
            logger.info("Wait complete, retrying...")
            continue

        if result.returncode != 0:
            _handle_sync_command_failure(result, elapsed, len(prompt), cwd)

        response = _parse_sync_json_response(result.stdout)
        _validate_claude_response(response)
        return response


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
    """Handle failed async command execution.

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
    stdout_text = stdout.decode()[:500] if stdout else "(no stdout)"

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
        stdout_text,
    )

    if elapsed > LONG_RUNTIME_THRESHOLD_SECONDS:
        raise RuntimeError(
            f"Claude CLI failed after {elapsed:.1f}s with return code {returncode}: {stderr_text}\n"
            f"Long runtime suggests task complexity - consider breaking into smaller tasks."
        )
    else:
        raise RuntimeError(f"Claude CLI error after {elapsed:.1f}s: {stderr_text}")


async def run_claude_async(
    prompt: str,
    *,
    settings: ClaudeCodeSettings | None = None,
) -> ClaudeJSONResponse:
    """Run Claude CLI asynchronously and return JSON response.

    Automatically retries on rate limit if retry_on_rate_limit is True (default).

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

    while True:
        start_time = time.time()
        process_output = await _execute_async_command(cmd, cwd, timeout_seconds)
        elapsed = time.time() - start_time
        stdout, stderr, returncode = process_output

        should_retry, wait_seconds = await _check_rate_limit_and_retry_async(
            stdout, stderr, returncode, retry_enabled
        )
        if should_retry:
            await asyncio.sleep(wait_seconds)
            logger.info("Wait complete, retrying...")
            continue

        if returncode != 0:
            _handle_async_command_failure(process_output, elapsed, len(prompt), cwd)

        response = _parse_sync_json_response(stdout.decode())
        _validate_claude_response(response)
        return response


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
