"""Tests for SDK adapter type conversions."""

import pytest
from datetime import datetime, timezone

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from pydantic_ai_claude_code import SDKAdapter
from pydantic_ai_claude_code.sdk_original_files.types import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    UserMessage,
)


class TestMessagesToPrompt:
    """Tests for converting messages to prompt string."""

    def test_simple_user_message(self):
        """Test converting simple user message."""
        adapter = SDKAdapter()

        messages = [
            ModelRequest(parts=[UserPromptPart(content="Hello, Claude!")])
        ]

        prompt = adapter.messages_to_prompt(messages)

        assert "Hello, Claude!" in prompt
        assert "Request:" in prompt

    def test_with_system_prompt(self):
        """Test including system prompt."""
        adapter = SDKAdapter()

        messages = [
            ModelRequest(parts=[
                SystemPromptPart(content="You are a helpful assistant."),
                UserPromptPart(content="Hi!"),
            ])
        ]

        prompt = adapter.messages_to_prompt(messages, include_system=True)

        assert "System:" in prompt
        assert "You are a helpful assistant." in prompt
        assert "Hi!" in prompt

    def test_skip_system_prompt(self):
        """Test skipping system prompt."""
        adapter = SDKAdapter()

        messages = [
            ModelRequest(parts=[
                SystemPromptPart(content="System prompt"),
                UserPromptPart(content="User message"),
            ])
        ]

        prompt = adapter.messages_to_prompt(messages, include_system=False)

        assert "System prompt" not in prompt
        assert "User message" in prompt

    def test_conversation_with_assistant(self):
        """Test conversation with assistant messages."""
        adapter = SDKAdapter()

        messages = [
            ModelRequest(parts=[UserPromptPart(content="What is 2+2?")]),
            ModelResponse(parts=[TextPart(content="The answer is 4.")]),
            ModelRequest(parts=[UserPromptPart(content="Thanks!")]),
        ]

        prompt = adapter.messages_to_prompt(messages)

        assert "What is 2+2?" in prompt
        assert "The answer is 4." in prompt
        assert "Thanks!" in prompt
        assert "Assistant:" in prompt

    def test_tool_result(self):
        """Test including tool results."""
        adapter = SDKAdapter()

        messages = [
            ModelRequest(parts=[
                ToolReturnPart(
                    tool_name="search",
                    content="Found: Python is a programming language",
                    tool_call_id="call_123",
                )
            ])
        ]

        prompt = adapter.messages_to_prompt(messages)

        assert "Tool Result" in prompt
        assert "search" in prompt
        assert "Python is a programming language" in prompt


class TestMessagesToSDKFormat:
    """Tests for converting messages to SDK format."""

    def test_user_message(self):
        """Test converting user message to SDK format."""
        adapter = SDKAdapter()

        messages = [
            ModelRequest(parts=[UserPromptPart(content="Hello")])
        ]

        sdk_messages = adapter.messages_to_sdk_format(messages)

        assert len(sdk_messages) == 1
        assert sdk_messages[0]["role"] == "user"

    def test_assistant_with_text(self):
        """Test converting assistant message with text."""
        adapter = SDKAdapter()

        messages = [
            ModelResponse(parts=[TextPart(content="Hello back!")])
        ]

        sdk_messages = adapter.messages_to_sdk_format(messages)

        assert len(sdk_messages) == 1
        assert sdk_messages[0]["role"] == "assistant"

    def test_tool_call_conversion(self):
        """Test converting tool calls."""
        adapter = SDKAdapter()

        messages = [
            ModelResponse(parts=[
                ToolCallPart(
                    tool_name="search",
                    args={"query": "test"},
                    tool_call_id="call_123",
                )
            ])
        ]

        sdk_messages = adapter.messages_to_sdk_format(messages)

        assert len(sdk_messages) == 1
        content = sdk_messages[0]["content"]
        assert content[0]["type"] == "tool_use"
        assert content[0]["name"] == "search"

    def test_tool_result_conversion(self):
        """Test converting tool results."""
        adapter = SDKAdapter()

        messages = [
            ModelRequest(parts=[
                ToolReturnPart(
                    tool_name="search",
                    content="Result data",
                    tool_call_id="call_123",
                )
            ])
        ]

        sdk_messages = adapter.messages_to_sdk_format(messages)

        assert len(sdk_messages) == 1
        content = sdk_messages[0]["content"]
        assert content[0]["type"] == "tool_result"
        assert content[0]["tool_use_id"] == "call_123"


class TestSDKToModelResponse:
    """Tests for converting SDK messages to ModelResponse."""

    def test_text_response(self):
        """Test converting text response."""
        adapter = SDKAdapter()

        sdk_messages = [
            AssistantMessage(
                role="assistant",
                content=[TextBlock(type="text", text="Hello!")]
            )
        ]

        response = adapter.sdk_to_model_response(sdk_messages)

        assert len(response.parts) == 1
        assert isinstance(response.parts[0], TextPart)
        assert response.parts[0].content == "Hello!"

    def test_tool_use_response(self):
        """Test converting tool use response."""
        adapter = SDKAdapter()

        sdk_messages = [
            AssistantMessage(
                role="assistant",
                content=[
                    ToolUseBlock(
                        type="tool_use",
                        id="call_123",
                        name="search",
                        input={"query": "test"},
                    )
                ]
            )
        ]

        response = adapter.sdk_to_model_response(sdk_messages)

        assert len(response.parts) == 1
        assert isinstance(response.parts[0], ToolCallPart)
        assert response.parts[0].tool_name == "search"
        assert response.parts[0].args == {"query": "test"}

    def test_result_message(self):
        """Test converting result message."""
        adapter = SDKAdapter()

        sdk_messages = [
            ResultMessage(
                type="result",
                subtype="success",
                is_error=False,
                result="Final answer",
                usage={"input_tokens": 100, "output_tokens": 50},
            )
        ]

        response = adapter.sdk_to_model_response(sdk_messages)

        assert len(response.parts) == 1
        assert isinstance(response.parts[0], TextPart)
        assert response.parts[0].content == "Final answer"

    def test_mixed_content(self):
        """Test converting mixed content."""
        adapter = SDKAdapter()

        sdk_messages = [
            AssistantMessage(
                role="assistant",
                content=[
                    TextBlock(type="text", text="Let me search"),
                    ToolUseBlock(
                        type="tool_use",
                        id="call_456",
                        name="search",
                        input={"q": "test"},
                    ),
                ]
            )
        ]

        response = adapter.sdk_to_model_response(sdk_messages)

        assert len(response.parts) == 2
        assert isinstance(response.parts[0], TextPart)
        assert isinstance(response.parts[1], ToolCallPart)

    def test_string_content(self):
        """Test converting string content."""
        adapter = SDKAdapter()

        sdk_messages = [
            AssistantMessage(
                role="assistant",
                content="Simple string response"
            )
        ]

        response = adapter.sdk_to_model_response(sdk_messages)

        assert len(response.parts) == 1
        assert response.parts[0].content == "Simple string response"


class TestOptionsConversion:
    """Tests for converting between options and settings."""

    def test_options_to_settings(self):
        """Test converting SDK options to settings."""
        adapter = SDKAdapter()

        options: ClaudeAgentOptions = ClaudeAgentOptions(
            model="sonnet",
            cwd="/tmp/test",
            allowed_tools=["Read", "Edit"],
            permission_mode="bypassPermissions",
            timeout_ms=120000,
            verbose=True,
        )

        settings = adapter.options_to_settings(options)

        assert settings["model"] == "sonnet"
        assert settings["working_directory"] == "/tmp/test"
        assert settings["allowed_tools"] == ["Read", "Edit"]
        assert settings["timeout_seconds"] == 120

    def test_settings_to_options(self):
        """Test converting settings to SDK options."""
        adapter = SDKAdapter()

        settings = {
            "model": "opus",
            "working_directory": "/home/user",
            "allowed_tools": ["Bash"],
            "disallowed_tools": ["WebFetch"],
            "permission_mode": "acceptEdits",
            "timeout_seconds": 300,
        }

        options = adapter.settings_to_options(settings)

        assert options["model"] == "opus"
        assert options["cwd"] == "/home/user"
        assert options["allowed_tools"] == ["Bash"]
        assert options["disallowed_tools"] == ["WebFetch"]
        assert options["timeout_ms"] == 300000


class TestToolCallIdGeneration:
    """Tests for tool call ID generation."""

    def test_unique_ids(self):
        """Test that generated IDs are unique."""
        adapter = SDKAdapter()

        ids = set()
        for _ in range(100):
            id_ = adapter._generate_tool_call_id()
            assert id_ not in ids
            ids.add(id_)

    def test_id_format(self):
        """Test tool call ID format."""
        adapter = SDKAdapter()

        id_ = adapter._generate_tool_call_id()

        assert id_.startswith("call_")
        assert len(id_) == 21  # "call_" + 16 hex chars


class TestUsageConversion:
    """Tests for usage statistics conversion."""

    def test_build_usage(self):
        """Test building usage from SDK data."""
        adapter = SDKAdapter()

        usage_data = {
            "input_tokens": 100,
            "output_tokens": 50,
        }

        usage = adapter._build_usage(usage_data)

        assert usage.input_tokens == 100
        assert usage.output_tokens == 50

    def test_build_usage_empty(self):
        """Test building usage with empty data."""
        adapter = SDKAdapter()

        usage = adapter._build_usage({})

        assert usage.input_tokens == 0
        assert usage.output_tokens == 0


class TestImports:
    """Tests for package imports."""

    def test_import_sdk_adapter(self):
        """Test importing SDKAdapter from package."""
        from pydantic_ai_claude_code import SDKAdapter

        adapter = SDKAdapter()
        assert adapter is not None
