"""Tests for unstructured (plain text) output with diverse query types."""

import pytest
from pydantic_ai import Agent

# Register the claude-code provider
import pydantic_ai_claude_code  # noqa: F401


def test_simple_calculation():
    """Test simple math calculation."""
    agent = Agent('claude-code:haiku')
    result = agent.run_sync("What is 7 * 8? Just the number.")
    assert "56" in str(result.output)


def test_text_generation():
    """Test basic text generation."""
    agent = Agent('claude-code:haiku')
    result = agent.run_sync("Say hello")
    output = str(result.output).lower()
    assert "hello" in output or "hi" in output


def test_list_generation():
    """Test generating a simple list."""
    agent = Agent('claude-code:haiku')
    result = agent.run_sync("List 3 fruits, comma separated")
    output = str(result.output)
    assert "," in output  # Should have comma-separated values


def test_yes_no_question():
    """Test yes/no question."""
    agent = Agent('claude-code:haiku')
    result = agent.run_sync("Is 10 > 5? Answer yes or no.")
    output = str(result.output).lower()
    assert "yes" in output


def test_multiline_response():
    """Test response that should be multiple lines."""
    agent = Agent('claude-code:haiku')
    result = agent.run_sync("Write 3 lines: line1, line2, line3")
    output = str(result.output)
    assert len(output) > 10  # Should have substantial content


@pytest.mark.asyncio
async def test_async_text_output():
    """Test async unstructured output."""
    agent = Agent('claude-code:haiku')
    result = await agent.run("What color is the sky? One word.")
    output = str(result.output).lower()
    assert "blue" in output
