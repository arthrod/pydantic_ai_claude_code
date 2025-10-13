"""Streamlined tests for structured output."""

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent

# Register the claude-code provider
import pydantic_ai_claude_code  # noqa: F401

# Test constants for expected values
EXPECTED_SUM_5_PLUS_3 = 8  # 5 + 3
EXPECTED_TEST_NUMBER = 42  # Test number value
EXPECTED_COLOR_COUNT = 3  # Number of colors in list test


class SingleInt(BaseModel):
    """Single integer field."""

    value: int


class MultiField(BaseModel):
    """Multiple fields with different types."""

    number: int
    text: str
    flag: bool


class ListResult(BaseModel):
    """Result with a list field."""

    items: list[str]


@pytest.mark.asyncio
async def test_single_int():
    """Test single integer field."""
    agent = Agent("claude-code:haiku", output_type=SingleInt)
    result = await agent.run("What is 5 + 3?")
    assert result.output.value == EXPECTED_SUM_5_PLUS_3


@pytest.mark.asyncio
async def test_multi_field():
    """Test multiple fields with different types."""
    agent = Agent("claude-code:haiku", output_type=MultiField)
    result = await agent.run("Number: 42, Text: hello, Flag: true")
    assert result.output.number == EXPECTED_TEST_NUMBER
    assert isinstance(result.output.text, str)
    assert isinstance(result.output.flag, bool)


@pytest.mark.asyncio
async def test_list_result():
    """Test result with list field."""
    agent = Agent("claude-code:haiku", output_type=ListResult)
    result = await agent.run("List 3 colors: red, blue, green")
    assert len(result.output.items) == EXPECTED_COLOR_COUNT
    assert isinstance(result.output.items, list)
