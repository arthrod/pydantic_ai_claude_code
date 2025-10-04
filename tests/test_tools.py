"""Tests for tool calling functionality."""

import pytest
from pydantic_ai import ToolDefinition
from pydantic_ai.messages import ToolCallPart

from pydantic_ai_claude_code.tools import (
    format_tools_for_prompt,
    is_tool_call_response,
    parse_tool_calls,
)


def test_format_tools_for_prompt_empty():
    """Test formatting with no tools."""
    result = format_tools_for_prompt([])
    assert result == ""


def test_format_tools_for_prompt_single_tool():
    """Test formatting a single tool."""
    tool = ToolDefinition(
        name="get_weather",
        description="Get weather for a city",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    )

    result = format_tools_for_prompt([tool])

    assert "get_weather" in result
    assert "Get weather for a city" in result
    assert '"city"' in result
    assert "tool_calls" in result
    assert "JSON" in result


def test_format_tools_for_prompt_multiple_tools():
    """Test formatting multiple tools."""
    tools = [
        ToolDefinition(
            name="tool1",
            description="First tool",
            parameters_json_schema={"type": "object", "properties": {}},
        ),
        ToolDefinition(
            name="tool2",
            description="Second tool",
            parameters_json_schema={"type": "object", "properties": {}},
        ),
    ]

    result = format_tools_for_prompt(tools)

    assert "tool1" in result
    assert "tool2" in result
    assert "First tool" in result
    assert "Second tool" in result


def test_parse_tool_calls_valid_single():
    """Test parsing a valid single tool call."""
    response = """```json
{
  "type": "tool_calls",
  "calls": [
    {"tool_name": "get_weather", "args": {"city": "London", "units": "celsius"}}
  ]
}
```"""

    result = parse_tool_calls(response)

    assert result is not None
    assert len(result) == 1
    assert isinstance(result[0], ToolCallPart)
    assert result[0].tool_name == "get_weather"
    assert result[0].args == {"city": "London", "units": "celsius"}
    assert result[0].tool_call_id.startswith("call_")


def test_parse_tool_calls_valid_multiple():
    """Test parsing multiple tool calls."""
    response = """{
  "type": "tool_calls",
  "calls": [
    {"tool_name": "tool1", "args": {"param": "value1"}},
    {"tool_name": "tool2", "args": {"param": "value2"}}
  ]
}"""

    result = parse_tool_calls(response)

    assert result is not None
    assert len(result) == 2
    assert result[0].tool_name == "tool1"
    assert result[1].tool_name == "tool2"


def test_parse_tool_calls_plain_text():
    """Test parsing plain text (not tool calls)."""
    response = "This is just a regular response, not a tool call."

    result = parse_tool_calls(response)

    assert result is None


def test_parse_tool_calls_invalid_json():
    """Test parsing invalid JSON."""
    response = """{
  "type": "tool_calls",
  "calls": [
    {"tool_name": "test", "args": invalid}
  ]
}"""

    result = parse_tool_calls(response)

    assert result is None


def test_parse_tool_calls_wrong_type():
    """Test parsing JSON with wrong type field."""
    response = """{
  "type": "something_else",
  "data": "value"
}"""

    result = parse_tool_calls(response)

    assert result is None


def test_parse_tool_calls_missing_calls():
    """Test parsing JSON missing calls array."""
    response = """{
  "type": "tool_calls"
}"""

    result = parse_tool_calls(response)

    assert result is None


def test_parse_tool_calls_empty_calls():
    """Test parsing JSON with empty calls array."""
    response = """{
  "type": "tool_calls",
  "calls": []
}"""

    result = parse_tool_calls(response)

    assert result is None


def test_parse_tool_calls_missing_tool_name():
    """Test parsing call missing tool_name."""
    response = """{
  "type": "tool_calls",
  "calls": [
    {"args": {"param": "value"}}
  ]
}"""

    result = parse_tool_calls(response)

    assert result is None


def test_parse_tool_calls_args_optional():
    """Test parsing call with missing args (should default to empty dict)."""
    response = """{
  "type": "tool_calls",
  "calls": [
    {"tool_name": "test"}
  ]
}"""

    result = parse_tool_calls(response)

    assert result is not None
    assert len(result) == 1
    assert result[0].tool_name == "test"
    assert result[0].args == {}


def test_is_tool_call_response_true():
    """Test detecting tool call response."""
    response = """{
  "type": "tool_calls",
  "calls": [
    {"tool_name": "test", "args": {}}
  ]
}"""

    assert is_tool_call_response(response) is True


def test_is_tool_call_response_false():
    """Test detecting non-tool-call response."""
    response = "This is a regular text response."

    assert is_tool_call_response(response) is False
