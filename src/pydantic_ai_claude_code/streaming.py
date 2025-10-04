"""Streaming utilities for Claude Code model."""

import asyncio
import json
from typing import AsyncIterator

from .types import ClaudeStreamEvent


async def run_claude_streaming(
    cmd: list[str],
    *,
    cwd: str | None = None,
) -> AsyncIterator[ClaudeStreamEvent]:
    """Run Claude CLI in streaming mode and yield events.

    Args:
        cmd: Command to execute (should include --output-format stream-json)
        cwd: Working directory

    Yields:
        Parsed stream events
    """
    # Start the process
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.DEVNULL,
        cwd=cwd,
    )

    assert process.stdout is not None

    # Read and parse JSON lines
    while True:
        line = await process.stdout.readline()
        if not line:
            break

        try:
            event = json.loads(line.decode().strip())
            yield event
        except json.JSONDecodeError:
            # Skip invalid JSON lines
            continue

    # Wait for process to complete
    await process.wait()

    if process.returncode != 0:
        stderr = await process.stderr.read() if process.stderr else b""
        raise RuntimeError(f"Claude CLI error: {stderr.decode()}")


def extract_text_from_stream_event(event: ClaudeStreamEvent) -> str | None:
    """Extract text content from a stream event.

    Args:
        event: Stream event from Claude CLI

    Returns:
        Extracted text or None if no text in event
    """
    event_type = event.get("type")

    if event_type == "assistant":
        # Extract text from assistant message
        message = event.get("message", {})
        content = message.get("content", [])

        # Concatenate all text parts
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))

        return "".join(text_parts) if text_parts else None

    elif event_type == "result":
        # Extract final result
        return event.get("result")

    return None
