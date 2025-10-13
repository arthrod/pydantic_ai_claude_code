"""Utility functions for Claude Code model."""

import json
import logging
import re
import subprocess
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

from .types import ClaudeCodeSettings, ClaudeJSONResponse, ClaudeStreamEvent

logger = logging.getLogger(__name__)


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
    cmd = ["claude", "--print"]

    # Add output format
    cmd.extend(["--output-format", output_format])

    # Add input format if not text
    if input_format != "text":
        cmd.extend(["--input-format", input_format])

    # For stream-json, add include-partial-messages for continuous progress
    if output_format == "stream-json":
        cmd.append("--include-partial-messages")

    # Add settings
    if settings.get("working_directory"):
        # Claude CLI doesn't have a --cwd flag, so we'll handle this via subprocess cwd
        pass

    allowed_tools = settings.get("allowed_tools")
    if allowed_tools:
        cmd.append("--allowed-tools")
        cmd.extend(allowed_tools)

    disallowed_tools = settings.get("disallowed_tools")
    if disallowed_tools:
        cmd.append("--disallowed-tools")
        cmd.extend(disallowed_tools)

    append_system_prompt = settings.get("append_system_prompt")
    if append_system_prompt:
        cmd.extend(["--append-system-prompt", append_system_prompt])

    # Always set permission mode for non-interactive use
    # Default to bypassPermissions since we cannot provide input (stdin=DEVNULL)
    permission_mode = settings.get("permission_mode") or "bypassPermissions"
    cmd.extend(["--permission-mode", permission_mode])

    model = settings.get("model")
    if model:
        cmd.extend(["--model", model])

    fallback_model = settings.get("fallback_model")
    if fallback_model:
        cmd.extend(["--fallback-model", fallback_model])

    max_turns = settings.get("max_turns")
    if max_turns:
        cmd.extend(["--max-turns", str(max_turns)])

    session_id = settings.get("session_id")
    if session_id:
        cmd.extend(["--session-id", session_id])

    if settings.get("dangerously_skip_permissions"):
        cmd.append("--dangerously-skip-permissions")

    # Always reference prompt.md file
    cmd.append("Follow the instructions in prompt.md")

    logger.debug("Built Claude command: %s", " ".join(cmd))

    return cmd


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
    # Check if retry is enabled
    retry_enabled = settings.get("retry_on_rate_limit", True) if settings else True

    # Determine working directory
    cwd = settings.get("working_directory") if settings else None

    # If no working directory, create a temp one
    if not cwd:
        cwd = tempfile.mkdtemp(prefix="claude_prompt_")
        logger.debug("Created temporary working directory: %s", cwd)

    # Ensure working directory exists
    Path(cwd).mkdir(parents=True, exist_ok=True)

    # Write prompt to prompt.md in the working directory
    prompt_file = Path(cwd) / "prompt.md"
    prompt_file.write_text(prompt)
    logger.debug("Wrote prompt (%d chars) to %s", len(prompt), prompt_file)

    # Build command (now references prompt.md)
    cmd = build_claude_command(settings=settings, output_format="json")

    # Retry loop for rate limiting
    while True:
        try:
            # Run command
            logger.info("Running Claude CLI synchronously in %s", cwd)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,  # Don't raise on non-zero exit
                cwd=cwd,
            )

            # Check for rate limit error
            if result.returncode != 0 and retry_enabled:
                error_output = result.stdout + "\n" + result.stderr
                is_rate_limited, reset_time = detect_rate_limit(error_output)

                if is_rate_limited and reset_time:
                    wait_seconds = calculate_wait_time(reset_time)
                    wait_minutes = wait_seconds // 60
                    logger.info(
                        "Rate limit hit. Waiting %d minutes until reset...",
                        wait_minutes,
                    )
                    time.sleep(wait_seconds)
                    logger.info("Wait complete, retrying...")
                    continue  # Retry the loop

            # If not rate-limited or retry disabled, check for errors
            if result.returncode != 0:
                logger.error(
                    "Claude CLI failed with return code %d: %s",
                    result.returncode,
                    result.stderr,
                )
                raise RuntimeError(f"Claude CLI error: {result.stderr}")

            # Parse JSON response
            raw_response = json.loads(result.stdout)

            if isinstance(raw_response, list):
                # Verbose mode: array of events - find the result event (usually last)
                logger.debug(
                    "Received verbose JSON output with %d events", len(raw_response)
                )
                response = None
                for event in raw_response:
                    if isinstance(event, dict) and event.get("type") == "result":
                        response = cast(ClaudeJSONResponse, event)
                        break

                if not response:
                    logger.error("No result event found in verbose output")
                    raise RuntimeError("No result event in Claude CLI output")
            else:
                # Non-verbose mode: single result object
                response = cast(ClaudeJSONResponse, raw_response)

            logger.debug(
                "Received response with %d tokens",
                response.get("usage", {}).get("output_tokens", 0)
                if isinstance(response.get("usage"), dict)
                else 0,
            )

            # Check for error
            if response.get("is_error"):
                error_msg = response.get("error", "Unknown error")
                logger.error("Claude CLI returned error: %s", error_msg)
                raise RuntimeError(f"Claude CLI error: {error_msg}")

            return response

        except (json.JSONDecodeError, KeyError) as e:
            # JSON parsing error - not a rate limit, raise immediately
            logger.error("Failed to parse Claude CLI response: %s", e)
            raise


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

    # Check if retry is enabled
    retry_enabled = settings.get("retry_on_rate_limit", True) if settings else True

    # Determine working directory
    cwd = settings.get("working_directory") if settings else None

    # If no working directory, create a temp one
    if not cwd:
        cwd = tempfile.mkdtemp(prefix="claude_prompt_")
        logger.debug("Created temporary working directory: %s", cwd)

    # Ensure working directory exists
    Path(cwd).mkdir(parents=True, exist_ok=True)

    # Write prompt to prompt.md in the working directory
    prompt_file = Path(cwd) / "prompt.md"
    prompt_file.write_text(prompt)
    logger.debug("Wrote prompt (%d chars) to %s", len(prompt), prompt_file)

    # Build command (now references prompt.md)
    cmd = build_claude_command(settings=settings, output_format="json")

    # Retry loop for rate limiting
    while True:
        try:
            # Run command asynchronously
            # stdin=DEVNULL because the CLI is non-interactive and should not read from stdin
            logger.info("Running Claude CLI asynchronously in %s", cwd)
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
                cwd=cwd,
            )

            stdout, stderr = await process.communicate()

            # Check for rate limit error
            if process.returncode != 0 and retry_enabled:
                error_output = stdout.decode() + "\n" + stderr.decode()
                is_rate_limited, reset_time = detect_rate_limit(error_output)

                if is_rate_limited and reset_time:
                    wait_seconds = calculate_wait_time(reset_time)
                    wait_minutes = wait_seconds // 60
                    logger.info(
                        "Rate limit hit. Waiting %d minutes until reset...",
                        wait_minutes,
                    )
                    await asyncio.sleep(wait_seconds)
                    logger.info("Wait complete, retrying...")
                    continue  # Retry the loop

            # If not rate-limited or retry disabled, check for errors
            if process.returncode != 0:
                logger.error(
                    "Claude CLI failed with return code %d: %s",
                    process.returncode,
                    stderr.decode(),
                )
                raise RuntimeError(f"Claude CLI error: {stderr.decode()}")

            # Parse JSON response
            raw_response = json.loads(stdout.decode())

            if isinstance(raw_response, list):
                # Verbose mode: array of events - find the result event (usually last)
                logger.debug(
                    "Received verbose JSON output with %d events", len(raw_response)
                )
                response = None
                for event in raw_response:
                    if isinstance(event, dict) and event.get("type") == "result":
                        response = cast(ClaudeJSONResponse, event)
                        break

                if not response:
                    logger.error("No result event found in verbose output")
                    raise RuntimeError("No result event in Claude CLI output")
            else:
                # Non-verbose mode: single result object
                response = cast(ClaudeJSONResponse, raw_response)

            logger.debug(
                "Received response with %d output tokens",
                response.get("usage", {}).get("output_tokens", 0)
                if isinstance(response.get("usage"), dict)
                else 0,
            )

            # Check for error
            if response.get("is_error"):
                error_msg = response.get("error", "Unknown error")
                logger.error("Claude CLI returned error: %s", error_msg)
                raise RuntimeError(f"Claude CLI error: {error_msg}")

            return response

        except (json.JSONDecodeError, KeyError) as e:
            # JSON parsing error - not a rate limit, raise immediately
            logger.error("Failed to parse Claude CLI response: %s", e)
            raise


def create_temp_workspace() -> Path:
    """Create a temporary workspace directory for Claude Code.

    Returns:
        Path to temporary directory
    """
    temp_dir = tempfile.mkdtemp(prefix="claude_code_")
    return Path(temp_dir)


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
        return event
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse stream JSON line: %s", e)
        return None


async def run_claude_with_prefill(
    prompt: str,
    prefill_text: str = "{",
    *,
    settings: ClaudeCodeSettings | None = None,
) -> ClaudeJSONResponse:
    """Run Claude CLI with assistant response prefilling to force JSON output.

    Args:
        prompt: The user prompt
        prefill_text: Text to prefill assistant response with (default: "{")
        settings: Optional settings

    Returns:
        Claude JSON response
    """
    import asyncio
    import uuid

    session_id = str(uuid.uuid4())

    logger.info("Running Claude with prefill: %r", prefill_text)

    # Build messages with user prompt and assistant prefill
    messages = [
        {
            "type": "user",
            "message": {"role": "user", "content": prompt},
            "session_id": session_id,
            "parent_tool_use_id": None,
        },
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": prefill_text},
            "session_id": session_id,
            "parent_tool_use_id": None,
        },
    ]

    # Build command for stream-json input/output
    cmd = build_claude_command(
        settings=settings, input_format="stream-json", output_format="stream-json"
    )

    cwd = settings.get("working_directory") if settings else None

    # Prepare input (newline-delimited JSON)
    input_data = "\n".join(json.dumps(msg) for msg in messages) + "\n"

    # Run command
    logger.debug("Running Claude CLI with stream-json input/output")
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )

    stdout, stderr = await process.communicate(input=input_data.encode())

    if process.returncode != 0:
        logger.error(
            "Claude CLI failed with return code %d: %s",
            process.returncode,
            stderr.decode(),
        )
        raise RuntimeError(f"Claude CLI error: {stderr.decode()}")

    # Parse stream-json output to extract final result
    lines = stdout.decode().strip().split("\n")
    result_text = None

    for line in lines:
        event = parse_stream_json_line(line)
        if event and isinstance(event, dict):
            event_type = event.get("type")
            if event_type == "result":
                # Return the result event as ClaudeJSONResponse (compatible structure)
                # Cast to ClaudeJSONResponse type
                return cast(ClaudeJSONResponse, event)
            elif event_type == "assistant":
                # Extract text from assistant message
                msg = event.get("message", {})
                if isinstance(msg, dict):
                    content = msg.get("content", [])
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                result_text = part.get("text", "")

    # Fallback: construct response from accumulated text
    if result_text:
        return {
            "result": result_text,
            "is_error": False,
            "usage": {},
            "total_cost_usd": 0.0,
        }

    raise RuntimeError("No result found in stream-json output")


async def run_claude_with_jq_pipeline(
    prompt: str,
    *,
    settings: ClaudeCodeSettings | None = None,
) -> ClaudeJSONResponse:
    """Run Claude CLI and extract JSON using jq pipeline for robustness.

    This uses jq to aggressively extract JSON from Claude's response,
    handling markdown blocks, mixed text, and other edge cases.

    Args:
        prompt: The user prompt
        settings: Optional settings

    Returns:
        Claude JSON response with cleaned result

    Raises:
        RuntimeError: If jq is not available or JSON extraction fails
    """
    import asyncio

    logger.info("Running Claude with jq JSON extraction pipeline")

    # First check if jq is available
    try:
        jq_check = await asyncio.create_subprocess_exec(
            "which",
            "jq",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await jq_check.communicate()
        if jq_check.returncode != 0:
            logger.error("jq is not installed")
            raise RuntimeError(
                "jq is not installed. Install with: sudo apt-get install jq"
            )
    except FileNotFoundError as e:
        logger.error("jq is not available in PATH")
        raise RuntimeError(
            "jq is not installed. Install with: sudo apt-get install jq"
        ) from e

    # Determine working directory
    cwd = settings.get("working_directory") if settings else None

    # If no working directory, create a temp one
    if not cwd:
        cwd = tempfile.mkdtemp(prefix="claude_prompt_")

    # Ensure working directory exists
    Path(cwd).mkdir(parents=True, exist_ok=True)

    # Write prompt to prompt.md in the working directory
    prompt_file = Path(cwd) / "prompt.md"
    prompt_file.write_text(prompt)

    # Run Claude CLI
    cmd = build_claude_command(settings=settings, output_format="json")

    # stdin=DEVNULL because the CLI is non-interactive and should not read from stdin
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.DEVNULL,
        cwd=cwd,
    )

    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        raise RuntimeError(f"Claude CLI error: {stderr.decode()}")

    response: ClaudeJSONResponse = json.loads(stdout.decode())

    if response.get("is_error"):
        error_msg = response.get("error", "Unknown error")
        raise RuntimeError(f"Claude CLI error: {error_msg}")

    result_text = response.get("result", "")

    # Use jq pipeline to extract JSON
    # Strategy: Try multiple jq filters in order of preference
    jq_filters = [
        # 1. Try parsing directly
        ".",
        # 2. Remove markdown code blocks and parse
        'gsub("```json\\n"; "") | gsub("\\n```"; "") | gsub("```"; "") | fromjson',
        # 3. Extract first JSON object
        'match("\\{[^}]+\\}") | .string | fromjson',
        # 4. Extract from array of matches
        '[match("\\{[^}]+\\}"; "g")] | .[0].string | fromjson',
    ]

    cleaned_result = None

    for jq_filter in jq_filters:
        try:
            # Use jq to process
            logger.debug("Trying jq filter: %s", jq_filter)
            jq_process = await asyncio.create_subprocess_exec(
                "jq",
                "-r",
                jq_filter,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            jq_stdout, jq_stderr = await jq_process.communicate(
                input=result_text.encode()
            )

            if jq_process.returncode == 0:
                cleaned_result = jq_stdout.decode().strip()
                # Verify it's valid JSON
                json.loads(cleaned_result)
                logger.debug(
                    "Successfully extracted JSON using jq filter: %s", jq_filter
                )
                break
        except (json.JSONDecodeError, Exception) as e:
            logger.debug("jq filter failed: %s - %s", jq_filter, e)
            continue

    if cleaned_result:
        response["result"] = cleaned_result
        logger.info("jq pipeline successfully cleaned JSON response")
        return response

    # If all jq strategies fail, return original
    logger.warning("All jq extraction strategies failed, returning original response")
    return response
