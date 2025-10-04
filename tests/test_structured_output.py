"""Comprehensive tests for structured output with diverse schemas."""

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent

# Register the claude-code provider
import pydantic_ai_claude_code  # noqa: F401


class SingleInt(BaseModel):
    """Single integer field."""
    value: int


class SingleFloat(BaseModel):
    """Single float field."""
    price: float


class SingleString(BaseModel):
    """Single string field."""
    message: str


class SingleBool(BaseModel):
    """Single boolean field."""
    is_valid: bool


class MultiFieldMath(BaseModel):
    """Multiple fields with different types."""
    result: int
    explanation: str
    is_correct: bool


class PersonInfo(BaseModel):
    """Nested person information."""
    name: str
    age: int
    email: str


class ListResult(BaseModel):
    """Result with a list field."""
    items: list[str]


class NestedObject(BaseModel):
    """Nested object structure."""
    title: str
    metadata: dict


class ComplexCalculation(BaseModel):
    """Complex multi-field result."""
    answer: float
    steps: list[str]
    confidence: float


class Classification(BaseModel):
    """Classification result."""
    category: str
    confidence: float
    tags: list[str]


@pytest.mark.asyncio
async def test_single_int():
    """Test single integer field."""
    agent = Agent('claude-code:sonnet', output_type=SingleInt)
    result = await agent.run('What is 15 * 8?')
    assert result.output.value == 120


@pytest.mark.asyncio
async def test_single_float():
    """Test single float field."""
    agent = Agent('claude-code:sonnet', output_type=SingleFloat)
    result = await agent.run('What is the price of pi rounded to 2 decimals?')
    assert abs(result.output.price - 3.14) < 0.01


@pytest.mark.asyncio
async def test_single_string():
    """Test single string field."""
    agent = Agent('claude-code:sonnet', output_type=SingleString)
    result = await agent.run('Say "Hello World"')
    assert "hello" in result.output.message.lower()
    assert "world" in result.output.message.lower()


@pytest.mark.asyncio
async def test_single_bool():
    """Test single boolean field."""
    agent = Agent('claude-code:sonnet', output_type=SingleBool)
    result = await agent.run('Is 100 greater than 50?')
    assert result.output.is_valid is True


@pytest.mark.asyncio
async def test_multi_field_math():
    """Test multiple fields with different types."""
    agent = Agent('claude-code:sonnet', output_type=MultiFieldMath)
    result = await agent.run('Calculate 7 * 6. Provide result, explanation, and confirm if correct.')
    assert result.output.result == 42
    assert isinstance(result.output.explanation, str)
    assert len(result.output.explanation) > 0
    assert result.output.is_correct is True


@pytest.mark.asyncio
async def test_person_info():
    """Test structured person information."""
    agent = Agent('claude-code:sonnet', output_type=PersonInfo)
    result = await agent.run('Create a person: Alice, 30 years old, email alice@example.com')
    assert result.output.name == "Alice"
    assert result.output.age == 30
    assert "@" in result.output.email


@pytest.mark.asyncio
async def test_list_result():
    """Test result with list field."""
    agent = Agent('claude-code:sonnet', output_type=ListResult)
    result = await agent.run('List 3 primary colors')
    assert len(result.output.items) >= 2
    assert isinstance(result.output.items, list)


@pytest.mark.asyncio
async def test_nested_object():
    """Test nested object structure."""
    agent = Agent('claude-code:sonnet', output_type=NestedObject)
    result = await agent.run('Create a report document with some metadata')
    assert len(result.output.title) > 0
    assert isinstance(result.output.metadata, dict)


@pytest.mark.asyncio
async def test_complex_calculation():
    """Test complex multi-field calculation."""
    agent = Agent('claude-code:sonnet', output_type=ComplexCalculation)
    result = await agent.run('What is the square root of 16? Show your work and confidence level.')
    assert abs(result.output.answer - 4.0) < 0.1
    assert isinstance(result.output.steps, list)
    assert 0 <= result.output.confidence <= 1


@pytest.mark.asyncio
async def test_classification():
    """Test classification with multiple fields."""
    agent = Agent('claude-code:sonnet', output_type=Classification)
    result = await agent.run('Classify "Python is a programming language" into category, confidence, and tags')
    assert isinstance(result.output.category, str)
    assert len(result.output.category) > 0
    assert 0 <= result.output.confidence <= 1
    assert isinstance(result.output.tags, list)
    assert len(result.output.tags) > 0
