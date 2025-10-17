"""Tests for streaming behavior - verifying incremental delivery of chunks."""

from datetime import datetime

import pytest
from pydantic_ai import Agent

import pydantic_ai_claude_code  # noqa: F401 - triggers registration

# Test constants
MIN_CHUNKS_FOR_STREAMING = 5
MAX_TIME_TO_FIRST_CHUNK_MS = 5000
MIN_STREAMING_TIMESPAN_MS = 100
MIN_CHUNK_GAP_MS = 50
MIN_SIGNIFICANT_GAPS = 2
MIN_CONTENT_LENGTH = 5  # Reduced to allow concise responses like "1\n2\n3\n4\n5"
MIN_CHUNKS_EXPECTED = 2
MIN_SUBSTANTIAL_CONTENT_LENGTH = 50
MAX_TIME_TO_FIRST_IN_BACKGROUND_MS = 1000


def get_timestamp_ms():
    """Get current timestamp in milliseconds."""
    return datetime.now().timestamp() * 1000


@pytest.mark.asyncio
async def test_streaming_delivers_chunks_incrementally():
    """Verify that chunks arrive incrementally over time, not all at once."""
    agent = Agent("claude-code:sonnet")
    prompt = "Write a medium-length paragraph (5-6 sentences) about space exploration"

    chunk_timestamps = []
    chunk_count = 0
    context_entered_at = None

    async with agent.run_stream(prompt) as result:
        context_entered_at = get_timestamp_ms()

        # Collect all chunks (don't break early to avoid cleanup issues)
        async for _text in result.stream_text():
            chunk_count += 1
            chunk_timestamps.append(get_timestamp_ms())

    # Verify we got multiple chunks
    assert chunk_count >= MIN_CHUNKS_FOR_STREAMING, f"Expected at least {MIN_CHUNKS_FOR_STREAMING} chunks, got {chunk_count}"

    # Verify first chunk arrived quickly after entering context
    time_to_first_chunk = chunk_timestamps[0] - context_entered_at
    assert (
        time_to_first_chunk < MAX_TIME_TO_FIRST_CHUNK_MS
    ), f"First chunk took {time_to_first_chunk}ms (expected < {MAX_TIME_TO_FIRST_CHUNK_MS}ms)"

    # Verify chunks arrived over time (not all at once)
    # If chunks were instant, all timestamps would be within a few ms
    # Real streaming should have delays between chunks
    time_span = chunk_timestamps[-1] - chunk_timestamps[0]
    assert time_span > MIN_STREAMING_TIMESPAN_MS, f"Chunks arrived in {time_span}ms (expected > {MIN_STREAMING_TIMESPAN_MS}ms for true streaming)"

    # Verify there are gaps between some chunks (not instant delivery)
    gaps = [chunk_timestamps[i + 1] - chunk_timestamps[i] for i in range(len(chunk_timestamps) - 1)]
    significant_gaps = [g for g in gaps if g > MIN_CHUNK_GAP_MS]
    assert (
        len(significant_gaps) >= MIN_SIGNIFICANT_GAPS
    ), f"Expected multiple gaps >{MIN_CHUNK_GAP_MS}ms between chunks, found {len(significant_gaps)}"


@pytest.mark.asyncio
async def test_streaming_filters_tool_use_messages():
    """Verify that tool-use messages are filtered out and only final response is streamed."""
    agent = Agent("claude-code:sonnet")
    # Simple prompt that shouldn't require tools, so response should start immediately
    prompt = "Count from 1 to 5"

    full_text = ""
    async with agent.run_stream(prompt) as result:
        async for text in result.stream_text():
            full_text = text

    # Verify we don't see tool-use messages like "I'll read the prompt.md file"
    # These should be filtered out by the marker detection
    assert "prompt.md" not in full_text.lower(), "Tool-use messages should be filtered out"
    assert "read" not in full_text[:100].lower() or "1" in full_text[:20], (
        "Response should not start with tool-use description"
    )

    # Verify we got actual content
    assert len(full_text) > MIN_CONTENT_LENGTH, "Should have received actual response content"


@pytest.mark.asyncio
async def test_streaming_delivers_complete_response():
    """Verify that streaming delivers the complete response."""
    agent = Agent("claude-code:sonnet")
    prompt = "Write exactly 3 bullet points about Python"

    chunks = []
    async with agent.run_stream(prompt) as result:
        async for text in result.stream_text():
            chunks.append(text)

    # Verify we got multiple chunks
    assert len(chunks) >= MIN_CHUNKS_EXPECTED, f"Expected multiple chunks, got {len(chunks)}"

    # Verify each chunk is cumulative (each chunk contains previous text)
    for i in range(1, len(chunks)):
        assert len(chunks[i]) >= len(chunks[i - 1]), (
            f"Chunk {i} should be >= chunk {i-1} (cumulative)"
        )
        assert chunks[i].startswith(chunks[i - 1][:MIN_SUBSTANTIAL_CONTENT_LENGTH]), (
            f"Chunk {i} should start with previous chunk content"
        )

    # Verify final response contains expected content indicators
    final_text = chunks[-1]
    assert len(final_text) > MIN_SUBSTANTIAL_CONTENT_LENGTH, "Should have substantial content"


@pytest.mark.asyncio
async def test_streaming_usage_available_after_completion():
    """Verify that usage information is available after streaming completes."""
    agent = Agent("claude-code:sonnet")
    prompt = "Say hello"

    async with agent.run_stream(prompt) as result:
        # Consume the stream
        async for _ in result.stream_text():
            pass

        # Usage should be available
        usage = result.usage()
        assert usage.output_tokens > 0, "Should have output tokens after completion"
        assert usage.input_tokens >= 0, "Should have input tokens"


@pytest.mark.asyncio
async def test_background_task_consumption():
    """Verify that the background task consumes events concurrently."""
    agent = Agent("claude-code:sonnet")
    prompt = "Write a short sentence"

    # Track when we enter context vs when we get first chunk
    context_entered_at = None
    first_chunk_at = None
    collected_all = False

    async with agent.run_stream(prompt) as result:
        context_entered_at = get_timestamp_ms()

        async for _text in result.stream_text():
            if first_chunk_at is None:
                first_chunk_at = get_timestamp_ms()
            # Continue collecting to avoid cleanup issues
            collected_all = True

    # Verify we actually got chunks
    assert collected_all, "Should have received chunks"
    assert first_chunk_at is not None, "Should have recorded first chunk time"

    # If background task is working, the gap between entering context
    # and getting first chunk should be minimal
    # because events are being buffered in the background
    time_to_first = first_chunk_at - context_entered_at
    assert time_to_first < MAX_TIME_TO_FIRST_IN_BACKGROUND_MS, (
        f"Time to first chunk was {time_to_first}ms, suggesting background task isn't working"
    )
