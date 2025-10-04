"""Utility functions for Claude Code model."""

import json
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .types import ClaudeCodeSettings, ClaudeJSONResponse, ClaudeStreamEvent


def build_claude_command(
    prompt: str | None = None,
    *,
    settings: ClaudeCodeSettings | None = None,
    input_format: str = "text",
    output_format: str = "json",
) -> list[str]:
    """Build Claude CLI command with appropriate flags.

    Args:
        prompt: The prompt to send to Claude (for non-streaming input)
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

    # Add verbose flag if using stream-json output
    if output_format == "stream-json":
        cmd.append("--verbose")

    # Add settings
    if settings.get("working_directory"):
        # Claude CLI doesn't have a --cwd flag, so we'll handle this via subprocess cwd
        pass

    if settings.get("allowed_tools"):
        cmd.append("--allowed-tools")
        cmd.extend(settings["allowed_tools"])

    if settings.get("disallowed_tools"):
        cmd.append("--disallowed-tools")
        cmd.extend(settings["disallowed_tools"])

    if settings.get("append_system_prompt"):
        cmd.extend(["--append-system-prompt", settings["append_system_prompt"]])

    if settings.get("permission_mode"):
        cmd.extend(["--permission-mode", settings["permission_mode"]])

    if settings.get("model"):
        cmd.extend(["--model", settings["model"]])

    if settings.get("fallback_model"):
        cmd.extend(["--fallback-model", settings["fallback_model"]])

    if settings.get("max_turns"):
        cmd.extend(["--max-turns", str(settings["max_turns"])])

    if settings.get("session_id"):
        cmd.extend(["--session-id", settings["session_id"]])

    if settings.get("dangerously_skip_permissions"):
        cmd.append("--dangerously-skip-permissions")

    # Add prompt if provided (for non-streaming input)
    if prompt:
        cmd.append(prompt)

    return cmd


def run_claude_sync(
    prompt: str,
    *,
    settings: ClaudeCodeSettings | None = None,
) -> ClaudeJSONResponse:
    """Run Claude CLI synchronously and return JSON response.

    Args:
        prompt: The prompt to send to Claude
        settings: Optional settings for Claude Code execution

    Returns:
        Claude JSON response

    Raises:
        subprocess.CalledProcessError: If Claude CLI fails
        json.JSONDecodeError: If response is not valid JSON
    """
    # Build command with prompt as argument
    cmd = build_claude_command(prompt, settings=settings, output_format="json")

    # Determine working directory
    cwd = settings.get("working_directory") if settings else None

    # Run command
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
        cwd=cwd,
    )

    # Parse JSON response
    response: ClaudeJSONResponse = json.loads(result.stdout)

    # Check for error
    if response.get("is_error"):
        error_msg = response.get("error", "Unknown error")
        raise RuntimeError(f"Claude CLI error: {error_msg}")

    return response


async def run_claude_async(
    prompt: str,
    *,
    settings: ClaudeCodeSettings | None = None,
) -> ClaudeJSONResponse:
    """Run Claude CLI asynchronously and return JSON response.

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

    cmd = build_claude_command(prompt, settings=settings, output_format="json")

    # Determine working directory
    cwd = settings.get("working_directory") if settings else None

    # Run command asynchronously with stdin explicitly closed
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

    # Parse JSON response
    response: ClaudeJSONResponse = json.loads(stdout.decode())

    # Check for error
    if response.get("is_error"):
        error_msg = response.get("error", "Unknown error")
        raise RuntimeError(f"Claude CLI error: {error_msg}")

    return response


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
        return json.loads(line)
    except json.JSONDecodeError:
        return None
