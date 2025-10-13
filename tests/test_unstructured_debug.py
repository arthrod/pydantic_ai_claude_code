"""Debug tests for unstructured output to identify issues."""

import logging

import pytest
from pydantic_ai import Agent

# Register the claude-code provider
import pydantic_ai_claude_code  # noqa: F401

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("pydantic_ai_claude_code").setLevel(logging.DEBUG)

# Test constants for minimum output lengths
MIN_MULTILINE_OUTPUT_CHARS = 10  # Minimum chars for multiline responses
MIN_SHORT_OUTPUT_CHARS = 5  # Minimum chars for short responses


def test_multiline_with_logging():
    """Test multiline response with full debug logging."""
    print("\n=== Testing multiline response with debug logging ===")
    agent = Agent("claude-code:haiku")

    try:
        result = agent.run_sync("Write 3 lines: line1, line2, line3")
        print("\n=== SUCCESS ===")
        print(f"Result type: {type(result.output)}")
        print(f"Result content: {result.output}")
        print(f"Result usage: {result.usage()}")
        assert len(str(result.output)) > MIN_MULTILINE_OUTPUT_CHARS
    except Exception as e:
        print("\n=== FAILURE ===")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception: {e}")
        # Check if it's the retry error
        if "Exceeded maximum retries" in str(e):
            print("\n=== Analysis: Model returned empty or invalid response ===")
        raise


def test_simple_one_line():
    """Test simpler single line response."""
    print("\n=== Testing simple one-line response ===")
    agent = Agent("claude-code:haiku")
    result = agent.run_sync("Say: hello")
    print(f"Result: {result.output}")
    assert "hello" in str(result.output).lower()


def test_numbered_list():
    """Test with explicit numbered list format."""
    print("\n=== Testing numbered list ===")
    agent = Agent("claude-code:haiku")
    result = agent.run_sync(
        "Write a numbered list with 3 items: 1. apple 2. banana 3. cherry"
    )
    print(f"Result: {result.output}")
    output = str(result.output)
    assert len(output) > MIN_MULTILINE_OUTPUT_CHARS


def test_newline_explicit():
    """Test with explicit newline mention."""
    print("\n=== Testing explicit newline request ===")
    agent = Agent("claude-code:haiku")
    result = agent.run_sync("Write three words separated by newlines: cat\\ndog\\nbird")
    print(f"Result: {result.output}")
    output = str(result.output)
    assert len(output) > MIN_SHORT_OUTPUT_CHARS


def test_with_direct_model_call():
    """Test by calling the model directly to see raw response."""
    print("\n=== Testing direct model call ===")
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    from pydantic_ai_claude_code.model import ClaudeCodeModel

    model = ClaudeCodeModel("haiku")

    # Create a request
    messages = [
        ModelRequest(
            parts=[UserPromptPart(content="Write 3 lines: line1, line2, line3")]
        )
    ]

    # Call model directly
    from pydantic_ai.models import ModelRequestParameters

    params = ModelRequestParameters()

    import asyncio

    async def call_model():
        response = await model.request(messages, None, params)
        print("\n=== Raw Model Response ===")
        print(f"Response type: {type(response)}")
        print(f"Response parts: {response.parts}")
        for i, part in enumerate(response.parts):
            print(f"Part {i}: type={type(part).__name__}")
            if hasattr(part, "content"):
                print(f"  Content: {part.content}")
            if hasattr(part, "tool_name"):
                print(f"  Tool: {part.tool_name}")
        print(f"Model name: {response.model_name}")
        print(f"Usage: {response.usage}")
        return response

    response = asyncio.run(call_model())

    # Check what we got
    assert len(response.parts) > 0, "Response has no parts!"

    from pydantic_ai.messages import TextPart

    text_parts = [p for p in response.parts if isinstance(p, TextPart)]
    print(f"\n=== Text Parts Found: {len(text_parts)} ===")
    for i, part in enumerate(text_parts):
        print(f"Text Part {i}: '{part.content}'")
        print(f"  Length: {len(part.content)}")
        print(f"  Is empty: {not part.content.strip()}")


@pytest.mark.asyncio
async def test_async_multiline():
    """Test multiline with async to see if it's sync-specific."""
    print("\n=== Testing async multiline response ===")
    agent = Agent("claude-code:haiku")
    result = await agent.run("Write 3 lines: line1, line2, line3")
    print(f"Result: {result.output}")
    assert len(str(result.output)) > MIN_MULTILINE_OUTPUT_CHARS
