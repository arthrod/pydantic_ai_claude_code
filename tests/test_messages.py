"""Tests for message formatting and conversion."""

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from pydantic_ai_claude_code.messages import (
    build_conversation_context,
    format_messages_for_claude,
)

# Test constants for message counts
EXPECTED_TOTAL_MESSAGES = 4  # Total messages in full conversation test
EXPECTED_ASSISTANT_MESSAGES = 2  # Number of assistant messages in full conversation test


def test_format_system_prompt():
    """Test that system prompt is prepended."""
    messages = [
        ModelRequest(
            parts=[
                SystemPromptPart(content="You are a helpful assistant."),
                UserPromptPart(content="What is 2+2?"),
            ]
        ),
    ]

    formatted, additional_files = format_messages_for_claude(messages)

    assert "System: You are a helpful assistant." in formatted
    assert "Request: What is 2+2?" in formatted
    # System should come before Request
    assert formatted.index("System:") < formatted.index("Request:")
    assert additional_files == {}  # No tool results


def test_format_conversation():
    """Test formatting a multi-turn conversation."""
    messages = [
        ModelRequest(parts=[UserPromptPart(content="What is 2+2?")]),
        ModelResponse(parts=[TextPart(content="4")]),
        ModelRequest(parts=[UserPromptPart(content="What about 3+3?")]),
    ]

    formatted, additional_files = format_messages_for_claude(messages)

    assert "Request: What is 2+2?" in formatted
    assert "Assistant: 4" in formatted
    assert "Request: What about 3+3?" in formatted
    assert additional_files == {}  # No tool results


def test_format_tool_call():
    """Test formatting tool calls - tool calls are skipped in history."""
    messages = [
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="calculator",
                    args={"a": 2, "b": 2},
                    tool_call_id="call_1",
                )
            ]
        ),
    ]

    formatted, additional_files = format_messages_for_claude(messages)

    # Tool calls are not included in conversation history
    # (only tool results are shown as "Context: ...")
    assert "calculator" not in formatted or formatted == ""
    assert additional_files == {}  # No tool results


def test_format_tool_return():
    """Test formatting tool returns - results should be written to files."""
    messages = [
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="calculator",
                    content="4",
                    tool_call_id="call_1",
                )
            ]
        ),
    ]

    formatted, additional_files = format_messages_for_claude(messages)

    # Should reference the file instead of embedding content
    assert "tool_result_1_calculator.txt" in formatted
    assert "Additional Information" in formatted
    assert "calculator tool" in formatted

    # Should have created one additional file
    assert len(additional_files) == 1
    assert "tool_result_1_calculator.txt" in additional_files

    # Verify file content
    tool_file = additional_files["tool_result_1_calculator.txt"]
    assert tool_file.exists()
    assert tool_file.read_text() == "4"

    # Clean up temp file
    tool_file.unlink()


def test_build_conversation_context_empty():
    """Test building context from empty messages."""
    context = build_conversation_context([])

    assert context["num_messages"] == 0
    assert context["has_system_prompt"] is False


def test_build_conversation_context_with_system():
    """Test building context with system prompt."""
    messages = [
        ModelRequest(
            parts=[
                SystemPromptPart(content="You are helpful."),
                UserPromptPart(content="Hello"),
            ]
        ),
    ]

    context = build_conversation_context(messages)

    assert context["has_system_prompt"] is True
    assert context["num_user_messages"] == 1


def test_build_conversation_context_full():
    """Test building context from complex conversation."""
    messages = [
        ModelRequest(parts=[UserPromptPart(content="Test 1")]),
        ModelResponse(
            parts=[
                TextPart(content="Response 1"),
                ToolCallPart(
                    tool_name="tool1",
                    args={},
                    tool_call_id="call_1",
                ),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="tool1",
                    content="result",
                    tool_call_id="call_1",
                )
            ]
        ),
        ModelResponse(parts=[TextPart(content="Final response")]),
    ]

    context = build_conversation_context(messages)

    assert context["num_messages"] == EXPECTED_TOTAL_MESSAGES
    assert context["num_user_messages"] == 1
    assert context["num_assistant_messages"] == EXPECTED_ASSISTANT_MESSAGES
    assert context["num_tool_calls"] == 1
    assert context["num_tool_returns"] == 1
