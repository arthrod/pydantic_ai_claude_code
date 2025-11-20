"""Tests for SDK Adapter."""

import pytest
from datetime import datetime, timezone

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)

from pydantic_ai_claude_code.sdk_adapter import SDKAdapter, get_adapter


class TestMessagesToPrompt:
    """Test conversion from Pydantic AI messages to prompt string."""

    def test_user_message(self):
        """Test converting user message."""
        adapter = SDKAdapter()

        messages = [
            ModelRequest(parts=[
                UserPromptPart(content="Hello, world!")
            ])
        ]

        prompt = adapter.messages_to_prompt(messages)

        assert "User: Hello, world!" in prompt

    def test_system_message(self):
        """Test converting system message."""
        adapter = SDKAdapter()

        messages = [
            ModelRequest(parts=[
                SystemPromptPart(content="You are helpful")
            ])
        ]

        prompt = adapter.messages_to_prompt(messages, include_system=True)
        assert "System: You are helpful" in prompt

        prompt = adapter.messages_to_prompt(messages, include_system=False)
        assert "System:" not in prompt

    def test_tool_return_message(self):
        """Test converting tool return message."""
        adapter = SDKAdapter()

        messages = [
            ModelRequest(parts=[
                ToolReturnPart(
                    tool_name="calculator",
                    content="42",
                    tool_call_id="call_123",
                )
            ])
        ]

        prompt = adapter.messages_to_prompt(messages)

        assert "Tool Result (calculator): 42" in prompt

    def test_multiple_messages(self):
        """Test converting multiple messages."""
        adapter = SDKAdapter()

        messages = [
            ModelRequest(parts=[
                SystemPromptPart(content="Be helpful"),
                UserPromptPart(content="What is 2+2?"),
            ]),
        ]

        prompt = adapter.messages_to_prompt(messages)

        assert "System: Be helpful" in prompt
        assert "User: What is 2+2?" in prompt

    def test_empty_messages(self):
        """Test converting empty message list."""
        adapter = SDKAdapter()

        prompt = adapter.messages_to_prompt([])
        assert prompt == ""


class TestSDKToModelResponse:
    """Test conversion from SDK messages to ModelResponse."""

    def test_text_response(self):
        """Test converting text response."""
        adapter = SDKAdapter()

        sdk_messages = [
            {"type": "result", "result": "The answer is 4"}
        ]

        response = adapter.sdk_to_model_response(sdk_messages)

        assert len(response.parts) == 1
        assert isinstance(response.parts[0], TextPart)
        assert response.parts[0].content == "The answer is 4"

    def test_assistant_message(self):
        """Test converting assistant message with content blocks."""
        adapter = SDKAdapter()

        sdk_messages = [
            {
                "type": "assistant",
                "content": [
                    {"type": "text", "text": "Hello!"},
                    {"type": "text", "text": "How can I help?"},
                ]
            }
        ]

        response = adapter.sdk_to_model_response(sdk_messages)

        assert len(response.parts) == 1
        assert "Hello!" in response.parts[0].content
        assert "How can I help?" in response.parts[0].content

    def test_tool_use_message(self):
        """Test converting tool use message."""
        adapter = SDKAdapter()

        sdk_messages = [
            {
                "type": "tool_use",
                "name": "calculator",
                "input": {"expr": "2+2"},
                "id": "call_abc123",
            }
        ]

        response = adapter.sdk_to_model_response(sdk_messages)

        assert len(response.parts) == 1
        assert isinstance(response.parts[0], ToolCallPart)
        assert response.parts[0].tool_name == "calculator"
        assert response.parts[0].args == {"expr": "2+2"}

    def test_usage_extraction(self):
        """Test extracting usage information."""
        adapter = SDKAdapter()

        sdk_messages = [
            {
                "type": "result",
                "result": "Done",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                },
            }
        ]

        response = adapter.sdk_to_model_response(sdk_messages)

        assert response.usage is not None
        assert response.usage.input_tokens == 100
        assert response.usage.output_tokens == 50

    def test_empty_messages_returns_empty_text(self):
        """Test that empty messages returns empty text part."""
        adapter = SDKAdapter()

        response = adapter.sdk_to_model_response([])

        assert len(response.parts) == 1
        assert isinstance(response.parts[0], TextPart)
        assert response.parts[0].content == ""

    def test_none_messages_filtered(self):
        """Test that None messages are filtered."""
        adapter = SDKAdapter()

        sdk_messages = [
            None,
            {"type": "result", "result": "Valid"},
            None,
        ]

        response = adapter.sdk_to_model_response(sdk_messages)

        assert len(response.parts) == 1
        assert response.parts[0].content == "Valid"

    def test_model_name_set(self):
        """Test that model name is set correctly."""
        adapter = SDKAdapter()

        response = adapter.sdk_to_model_response(
            [{"type": "result", "result": "ok"}],
            model_name="claude-code:sonnet"
        )

        assert response.model_name == "claude-code:sonnet"

    def test_timestamp_set(self):
        """Test that timestamp is set."""
        adapter = SDKAdapter()

        response = adapter.sdk_to_model_response(
            [{"type": "result", "result": "ok"}]
        )

        assert response.timestamp is not None


class TestExtractHelpers:
    """Test internal extraction helper methods."""

    def test_extract_assistant_content_string(self):
        """Test extracting string content."""
        adapter = SDKAdapter()

        msg = {"content": "Simple string"}
        result = adapter._extract_assistant_content(msg)

        assert result == "Simple string"

    def test_extract_assistant_content_list(self):
        """Test extracting list content."""
        adapter = SDKAdapter()

        msg = {
            "content": [
                {"type": "text", "text": "First"},
                {"type": "text", "text": "Second"},
            ]
        }
        result = adapter._extract_assistant_content(msg)

        assert "First" in result
        assert "Second" in result

    def test_extract_result_content(self):
        """Test extracting result content."""
        adapter = SDKAdapter()

        msg = {"result": "The result"}
        result = adapter._extract_result_content(msg)

        assert result == "The result"

    def test_generate_tool_call_id(self):
        """Test tool call ID generation."""
        adapter = SDKAdapter()

        id1 = adapter._generate_tool_call_id()
        id2 = adapter._generate_tool_call_id()

        assert id1.startswith("call_")
        assert id2.startswith("call_")
        assert id1 != id2


class TestModelResponseToDict:
    """Test conversion of ModelResponse to dictionary."""

    def test_basic_conversion(self):
        """Test basic response conversion to dict."""
        adapter = SDKAdapter()

        response = ModelResponse(
            parts=[TextPart(content="Hello!")],
            model_name="claude-code:sonnet",
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        )

        result = adapter.model_response_to_dict(response)

        assert result["model_name"] == "claude-code:sonnet"
        assert len(result["parts"]) == 1
        assert result["parts"][0]["type"] == "text"
        assert result["parts"][0]["content"] == "Hello!"

    def test_tool_call_conversion(self):
        """Test tool call conversion to dict."""
        adapter = SDKAdapter()

        response = ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="test",
                    args={"x": 1},
                    tool_call_id="call_123",
                )
            ],
            model_name="test",
            timestamp=datetime.now(timezone.utc),
        )

        result = adapter.model_response_to_dict(response)

        assert result["parts"][0]["type"] == "tool_call"
        assert result["parts"][0]["tool_name"] == "test"
        assert result["parts"][0]["args"] == {"x": 1}

    def test_usage_conversion(self):
        """Test usage info conversion to dict."""
        adapter = SDKAdapter()

        from pydantic_ai.usage import RequestUsage

        response = ModelResponse(
            parts=[TextPart(content="")],
            model_name="test",
            timestamp=datetime.now(timezone.utc),
            usage=RequestUsage(
                input_tokens=100,
                output_tokens=50,
            ),
        )

        result = adapter.model_response_to_dict(response)

        assert result["usage"]["input_tokens"] == 100
        assert result["usage"]["output_tokens"] == 50
        assert result["usage"]["total_tokens"] == 150


class TestGlobalAdapter:
    """Test global adapter singleton."""

    def test_get_adapter_returns_same_instance(self):
        """Test that get_adapter returns singleton."""
        adapter1 = get_adapter()
        adapter2 = get_adapter()

        assert adapter1 is adapter2

    def test_get_adapter_returns_sdk_adapter(self):
        """Test that get_adapter returns SDKAdapter instance."""
        adapter = get_adapter()

        assert isinstance(adapter, SDKAdapter)
