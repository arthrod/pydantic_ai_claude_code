"""Streamlined tests for structured output."""

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent

# Register the claude-code provider
import pydantic_ai_claude_code  # noqa: F401


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
    agent = Agent('claude-code:haiku', output_type=SingleInt)
    result = await agent.run('What is 5 + 3?')
    assert result.output.value == 8


@pytest.mark.asyncio
async def test_multi_field():
    """Test multiple fields with different types."""
    agent = Agent('claude-code:haiku', output_type=MultiField)
    result = await agent.run('Number: 42, Text: hello, Flag: true')
    assert result.output.number == 42
    assert isinstance(result.output.text, str)
    assert isinstance(result.output.flag, bool)


@pytest.mark.asyncio
async def test_list_result():
    """Test result with list field."""
    agent = Agent('claude-code:haiku', output_type=ListResult)
    result = await agent.run('List 3 colors: red, blue, green')
    assert len(result.output.items) == 3
    assert isinstance(result.output.items, list)
