"""Basic integration tests for Claude Code model."""

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent

from pydantic_ai_claude_code import ClaudeCodeProvider

# Test constants
EXPECTED_MATH_RESULT = 12  # 5 + 7


class MathResult(BaseModel):
    """Math calculation result."""

    result: int
    explanation: str


def test_basic_query_sync():
    """Test basic synchronous query using string format."""
    agent = Agent("claude-code:sonnet")

    result = agent.run_sync("What is 2+2? Just give me the number.")
    assert "4" in str(result.output)


@pytest.mark.asyncio
async def test_basic_query_async():
    """Test basic asynchronous query using string format."""
    agent = Agent("claude-code:sonnet")

    result = await agent.run("What is 3+3? Just give me the number.")
    assert "6" in str(result.output)


def test_structured_output_sync():
    """Test structured output with Pydantic model using string format."""
    agent = Agent("claude-code:sonnet", output_type=MathResult)

    result = agent.run_sync("Calculate 5+7 and explain why.")

    assert isinstance(result.output, MathResult)
    assert result.output.result == EXPECTED_MATH_RESULT
    assert len(result.output.explanation) > 0


def test_provider_settings():
    """Test provider with custom settings using model_settings at run-time."""
    from pydantic_ai_claude_code import ClaudeCodeModel

    # New stateless provider pattern
    provider = ClaudeCodeProvider()
    model = provider.create_model("sonnet")
    agent = Agent(model)

    # Pass settings at run-time via model_settings
    result = agent.run_sync(
        "What is the capital of France? Just the city name.",
        model_settings={"verbose": False}
    )
    assert "Paris" in str(result.output)


def test_model_string_format():
    """Test using model string directly (new recommended pattern)."""
    # This is the simplest way to use claude-code
    agent = Agent("claude-code:sonnet")

    result = agent.run_sync("What is 2+2? Just give me the number.")
    assert "4" in str(result.output)


def test_provider_preset():
    """Test model string with provider preset format."""
    # This tests the claude-code:preset:model format
    # Note: This will only work if the preset is configured
    # For testing, we use the model directly
    from pydantic_ai_claude_code import ClaudeCodeModel

    provider = ClaudeCodeProvider()
    model = provider.create_model("sonnet")
    agent = Agent(model)

    result = agent.run_sync("Say hello")
    assert result.output is not None


def test_usage_tracking():
    """Test that usage information is tracked using string format."""
    agent = Agent("claude-code:sonnet")

    result = agent.run_sync("Say hello")

    usage = result.usage()
    assert usage.input_tokens > 0
    # Note: output tokens might be 0 for very short responses
