"""Streaming utilities for Claude Code model."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from .types import ClaudeStreamEvent

logger = logging.getLogger(__name__)


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
    logger.info("Starting Claude CLI streaming in %s", cwd or "current directory")
    logger.debug("Streaming command: %s", " ".join(cmd))

    # Start the process
    # stdin=DEVNULL because the CLI is non-interactive and should not read from stdin
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.DEVNULL,
        cwd=cwd,
    )

    assert process.stdout is not None

    event_count = 0

    # Read and parse JSON lines
    while True:
        line = await process.stdout.readline()
        if not line:
            break

        try:
            event = json.loads(line.decode().strip())
            event_count += 1
            if event.get("type"):
                logger.debug("Streaming event #%d: type=%s", event_count, event["type"])
            yield event
        except json.JSONDecodeError as e:
            # Skip invalid JSON lines
            logger.warning("Skipping invalid JSON line in stream: %s", e)
            continue

    # Wait for process to complete
    await process.wait()

    logger.info("Stream completed with %d events", event_count)

    if process.returncode != 0:
        stderr = await process.stderr.read() if process.stderr else b""
        logger.error(
            "Claude CLI streaming failed with return code %d: %s",
            process.returncode,
            stderr.decode(),
        )
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
        if not isinstance(message, dict):
            return None

        content = message.get("content", [])
        if not isinstance(content, list):
            return None

        # Concatenate all text parts
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_value = part.get("text", "")
                if isinstance(text_value, str):
                    text_parts.append(text_value)

        text = "".join(text_parts) if text_parts else None
        if text:
            logger.debug("Extracted %d chars of text from assistant event", len(text))
        return text

    elif event_type == "result":
        # Extract final result
        result = event.get("result")
        if result and isinstance(result, str):
            logger.debug("Extracted result: %d chars", len(result))
            return result
        return None

    return None
