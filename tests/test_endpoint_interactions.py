"""Tests for the interaction between Claude Code model and Pydantic AI framework.

Uses Pydantic AI's TestModel and FunctionModel to exercise the integration
layer without calling the actual Claude CLI. This enables fast, deterministic
unit tests for:
- Agent creation and model override
- Message conversion and formatting
- Structured output handling
- Tool registration and function selection
- Settings merge logic
- Error handling paths (OAuth, rate limits)
- Response parsing and conversion
- Provider configuration
"""

import json
import tempfile

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent, capture_run_messages, models
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RequestUsage

from pydantic_ai_claude_code import ClaudeCodeModel, ClaudeCodeProvider
from pydantic_ai_claude_code.messages import (
    build_conversation_context,
    extract_text_from_response,
    format_messages_for_claude,
)
from pydantic_ai_claude_code.response_utils import (
    create_tool_call_part,
    extract_model_parameters,
    get_working_directory,
)
from pydantic_ai_claude_code.types import ClaudeCodeSettings, ClaudeJSONResponse

# Block all real model requests during tests
models.ALLOW_MODEL_REQUESTS = False

# Test constants
EXPECTED_ANSWER = 42
EXPECTED_MULTI_TURN_COUNT = 2
EXPECTED_MIN_MESSAGES = 2
DEFAULT_TIMEOUT = 900
RATE_LIMIT_FALLBACK_SECONDS = 300
EXPECTED_INPUT_TOKENS = 100
EXPECTED_OUTPUT_TOKENS = 50
EXPECTED_DURATION_MS = 1500
EXPECTED_ALICE_AGE = 30
CUSTOM_TIMEOUT = 300


# ===== Fixtures =====


@pytest.fixture
def test_model():
    """Provide a fresh TestModel instance."""
    return TestModel()


@pytest.fixture
def claude_code_model():
    """Provide a ClaudeCodeModel for testing internal methods."""
    return ClaudeCodeModel(model_name="sonnet")


@pytest.fixture
def sample_claude_response() -> ClaudeJSONResponse:
    """Provide a sample Claude CLI JSON response."""
    return ClaudeJSONResponse(
        type="result",
        subtype="success",
        is_error=False,
        duration_ms=1500,
        duration_api_ms=1200,
        num_turns=1,
        result="The answer is 42.",
        session_id="test-session-001",
        total_cost_usd=0.003,
        usage={
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
        modelUsage={
            "claude-sonnet-4-20250514": {
                "inputTokens": 100,
                "outputTokens": 50,
                "cacheReadInputTokens": 0,
                "cacheCreationInputTokens": 0,
                "costUSD": 0.003,
            }
        },
    )


# ===== Section 1: Agent Override with TestModel =====


class TestAgentOverrideWithTestModel:
    """Test that Agent.override works correctly with claude-code agents."""

    async def test_override_replaces_model(self, test_model: TestModel):
        """Verify Agent.override replaces the model for the agent run."""
        agent = Agent("claude-code:sonnet")

        with agent.override(model=test_model):
            result = await agent.run("What is 2+2?")

        assert result.output == "success (no tool calls)"

    async def test_override_with_custom_result_text(self):
        """Verify custom_output_text customizes TestModel output."""
        m = TestModel(custom_output_text="Paris")
        agent = Agent("claude-code:sonnet")

        with agent.override(model=m):
            result = await agent.run("What is the capital of France?")

        assert result.output == "Paris"

    async def test_override_preserves_agent_tools(self, test_model: TestModel):
        """Verify that Agent.override preserves registered tools."""

        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        agent = Agent("claude-code:sonnet", tools=[add])

        with agent.override(model=test_model):
            await agent.run("Add 3 and 5")

        # TestModel should have seen the tool definitions
        params = test_model.last_model_request_parameters
        assert params is not None
        tool_names = [t.name for t in params.function_tools]
        assert "add" in tool_names

    async def test_override_preserves_system_prompt(self, test_model: TestModel):
        """Verify system prompt is passed through the override."""
        agent = Agent(
            "claude-code:sonnet",
            system_prompt="You are a math tutor.",
        )

        with agent.override(model=test_model):
            result = await agent.run("Help me with fractions")

        assert result.output is not None

    async def test_override_with_structured_output(self, test_model: TestModel):
        """Verify structured output works with model override."""

        class CityInfo(BaseModel):
            name: str
            country: str

        agent = Agent("claude-code:sonnet", output_type=CityInfo)

        with agent.override(model=test_model):
            result = await agent.run("Tell me about Paris")

        # TestModel will produce schema-conforming output
        assert isinstance(result.output, CityInfo)

    async def test_override_does_not_affect_other_agents(
        self, test_model: TestModel
    ):
        """Verify override is scoped to the context manager."""
        agent1 = Agent("claude-code:sonnet")
        agent2 = Agent("claude-code:sonnet")

        with agent1.override(model=test_model):
            result = await agent1.run("Test")
            assert result.output == "success (no tool calls)"

        # agent2 was never overridden - verify it still has its original model
        # (We can't call it because ALLOW_MODEL_REQUESTS=False, but the
        # model attribute should remain unchanged)
        assert agent2.model is not None


# ===== Section 2: FunctionModel for Custom Responses =====


class TestFunctionModelInteraction:
    """Test using FunctionModel to simulate custom Claude Code responses."""

    async def test_function_model_returns_text(self):
        """Verify FunctionModel can return plain text responses."""

        def mock_claude(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            return ModelResponse(parts=[TextPart("Hello from mock Claude!")])

        agent = Agent("claude-code:sonnet")

        with agent.override(model=FunctionModel(mock_claude)):
            result = await agent.run("Say hello")

        assert result.output == "Hello from mock Claude!"

    async def test_function_model_inspects_messages(self):
        """Verify FunctionModel receives the full message history."""
        captured_messages: list[list[ModelMessage]] = []

        def mock_claude(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            captured_messages.append(messages)
            return ModelResponse(parts=[TextPart("OK")])

        agent = Agent("claude-code:sonnet")

        with agent.override(model=FunctionModel(mock_claude)):
            await agent.run("First message")

        assert len(captured_messages) == 1
        # Should contain at least the user message
        assert len(captured_messages[0]) >= 1

    async def test_function_model_with_tool_calls(self):
        """Verify FunctionModel can simulate tool call responses."""
        call_count = 0

        def my_tool(x: int) -> int:
            """Double a number."""
            nonlocal call_count
            call_count += 1
            return x * 2

        def mock_claude(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            # First call: invoke the tool
            if len(messages) == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="my_tool",
                            args={"x": 5},
                            tool_call_id="call_001",
                        )
                    ]
                )
            # Second call: return the final text
            return ModelResponse(
                parts=[TextPart("The result is 10")]
            )

        agent = Agent("claude-code:sonnet", tools=[my_tool])

        with agent.override(model=FunctionModel(mock_claude)):
            result = await agent.run("Double 5")

        assert call_count == 1
        assert "10" in result.output

    async def test_function_model_with_structured_output(self):
        """Verify FunctionModel can return structured output via tool call."""

        class MathResult(BaseModel):
            result: int
            explanation: str

        def mock_claude(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="final_result",
                        args={
                            "result": 42,
                            "explanation": "The answer to everything",
                        },
                        tool_call_id="call_structured_001",
                    )
                ]
            )

        agent = Agent("claude-code:sonnet", output_type=MathResult)

        with agent.override(model=FunctionModel(mock_claude)):
            result = await agent.run("What is the answer?")

        assert isinstance(result.output, MathResult)
        assert result.output.result == EXPECTED_ANSWER
        assert result.output.explanation == "The answer to everything"

    async def test_function_model_multi_turn(self):
        """Verify FunctionModel handles multi-turn conversations."""
        turn_count = 0

        def mock_claude(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            nonlocal turn_count
            turn_count += 1
            return ModelResponse(
                parts=[TextPart(f"Response turn {turn_count}")]
            )

        agent = Agent("claude-code:sonnet")

        with agent.override(model=FunctionModel(mock_claude)):
            result1 = await agent.run("First question")
            result2 = await agent.run(
                "Second question",
                message_history=result1.all_messages(),
            )

        assert "turn 1" in result1.output
        assert "turn 2" in result2.output
        assert turn_count == EXPECTED_MULTI_TURN_COUNT


# ===== Section 3: capture_run_messages =====


class TestCaptureRunMessages:
    """Test message capture for inspecting agent-model exchanges."""

    async def test_capture_basic_exchange(self, test_model: TestModel):
        """Verify capture_run_messages captures the full exchange."""
        agent = Agent("claude-code:sonnet")

        with capture_run_messages() as messages, agent.override(
            model=test_model
        ):
            await agent.run("Hello")

        assert len(messages) >= EXPECTED_MIN_MESSAGES
        # First message should be a request containing the user prompt
        first_msg = messages[0]
        assert isinstance(first_msg, ModelRequest)
        has_user_prompt = any(
            isinstance(p, UserPromptPart) for p in first_msg.parts
        )
        assert has_user_prompt

    async def test_capture_with_tool_calls(self, test_model: TestModel):
        """Verify capture captures tool call exchanges."""

        def greet(name: str) -> str:
            """Greet by name."""
            return f"Hello, {name}!"

        agent = Agent("claude-code:sonnet", tools=[greet])

        with capture_run_messages() as messages, agent.override(
            model=test_model
        ):
            await agent.run("Greet Alice")

        # Should capture the tool call and tool return messages
        has_tool_call = any(
            isinstance(msg, ModelResponse)
            and any(isinstance(p, ToolCallPart) for p in msg.parts)
            for msg in messages
        )
        assert has_tool_call, "Should have captured a tool call"

        has_tool_return = any(
            isinstance(msg, ModelRequest)
            and any(isinstance(p, ToolReturnPart) for p in msg.parts)
            for msg in messages
        )
        assert has_tool_return, "Should have captured a tool return"

    async def test_capture_system_prompt(self, test_model: TestModel):
        """Verify system prompt appears in captured messages."""
        agent = Agent(
            "claude-code:sonnet",
            system_prompt="You are a helpful bot.",
        )

        with capture_run_messages() as messages, agent.override(
            model=test_model
        ):
            await agent.run("Hi")

        has_system = any(
            isinstance(msg, ModelRequest)
            and any(isinstance(p, SystemPromptPart) for p in msg.parts)
            for msg in messages
        )
        assert has_system, "Should have captured the system prompt"


# ===== Section 4: ClaudeCodeModel Internal Methods =====


class TestClaudeCodeModelInternals:
    """Test internal methods of ClaudeCodeModel without calling CLI."""

    def test_model_name_property(self, claude_code_model: ClaudeCodeModel):
        """Verify model_name format."""
        assert claude_code_model.model_name == "claude-code:sonnet"

    def test_model_name_with_preset(self):
        """Verify model_name format with provider preset."""
        model = ClaudeCodeModel(
            model_name="sonnet", provider_preset="deepseek"
        )
        assert model.model_name == "claude-code:deepseek:sonnet"

    def test_system_property(self, claude_code_model: ClaudeCodeModel):
        """Verify system identifier."""
        assert claude_code_model.system == "claude-code"

    def test_build_options_defaults(self, claude_code_model: ClaudeCodeModel):
        """Verify default options are set correctly."""
        settings = claude_code_model._build_options(
            None, ModelRequestParameters()
        )

        assert settings["model"] == "sonnet"
        assert settings["dangerously_skip_permissions"] is True
        assert settings["use_temp_workspace"] is True
        assert settings["retry_on_rate_limit"] is True
        assert settings["timeout_seconds"] == DEFAULT_TIMEOUT
        assert settings["use_sandbox_runtime"] is True

    def test_build_options_with_model_settings(
        self, claude_code_model: ClaudeCodeModel
    ):
        """Verify model_settings overrides are applied."""
        model_settings = {
            "working_directory": "/custom/path",
            "timeout_seconds": CUSTOM_TIMEOUT,
            "verbose": True,
        }

        settings = claude_code_model._build_options(
            model_settings, ModelRequestParameters()
        )

        assert settings["working_directory"] == "/custom/path"
        assert settings["timeout_seconds"] == CUSTOM_TIMEOUT
        assert settings["verbose"] is True

    def test_build_options_with_hooks(
        self, claude_code_model: ClaudeCodeModel
    ):
        """Verify hooks are passed through model_settings."""
        hooks = [
            {
                "matcher": {"event": "tool_use"},
                "commands": ["echo $TOOL_NAME"],
            }
        ]
        model_settings = {"hooks": hooks}

        settings = claude_code_model._build_options(
            model_settings, ModelRequestParameters()
        )

        assert settings.get("__hooks__") == hooks

    def test_check_has_tool_results_empty(
        self, claude_code_model: ClaudeCodeModel
    ):
        """Verify no tool results in empty messages."""
        messages: list[ModelMessage] = []
        assert claude_code_model._check_has_tool_results(messages) is False

    def test_check_has_tool_results_with_results(
        self, claude_code_model: ClaudeCodeModel
    ):
        """Verify detection of tool results in messages."""
        messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="test_tool",
                        content="result data",
                        tool_call_id="call_123",
                    )
                ]
            )
        ]
        assert claude_code_model._check_has_tool_results(messages) is True

    def test_check_has_tool_results_without_results(
        self, claude_code_model: ClaudeCodeModel
    ):
        """Verify no false positives for non-tool messages."""
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content="Hello")])
        ]
        assert claude_code_model._check_has_tool_results(messages) is False

    def test_xml_to_markdown_conversion(
        self, claude_code_model: ClaudeCodeModel
    ):
        """Verify XML description to markdown conversion."""
        xml = "<summary>Get weather data</summary><returns><description>Temperature in celsius</description></returns>"
        result = claude_code_model._xml_to_markdown(xml)
        assert "Get weather data." in result
        assert "Returns: Temperature in celsius" in result

    def test_xml_to_markdown_plain_text(
        self, claude_code_model: ClaudeCodeModel
    ):
        """Verify plain text passes through unchanged."""
        plain = "Just a plain description"
        result = claude_code_model._xml_to_markdown(plain)
        assert result == plain

    def test_create_usage(self, sample_claude_response: ClaudeJSONResponse):
        """Verify usage extraction from Claude response."""
        usage = ClaudeCodeModel._create_usage(sample_claude_response)

        assert isinstance(usage, RequestUsage)
        assert usage.input_tokens == EXPECTED_INPUT_TOKENS
        assert usage.output_tokens == EXPECTED_OUTPUT_TOKENS
        assert usage.details is not None
        assert usage.details["duration_ms"] == EXPECTED_DURATION_MS
        assert usage.details["num_turns"] == 1

    def test_create_usage_empty_response(self):
        """Verify usage handles empty response gracefully."""
        response = ClaudeJSONResponse()
        usage = ClaudeCodeModel._create_usage(response)

        assert usage.input_tokens == 0
        assert usage.output_tokens == 0

    def test_get_model_name_from_response(
        self,
        claude_code_model: ClaudeCodeModel,
        sample_claude_response: ClaudeJSONResponse,
    ):
        """Verify model name extraction from response."""
        name = claude_code_model._get_model_name(sample_claude_response)
        assert name == "claude-sonnet-4-20250514"

    def test_get_model_name_fallback(
        self, claude_code_model: ClaudeCodeModel
    ):
        """Verify model name fallback when not in response."""
        response = ClaudeJSONResponse()
        name = claude_code_model._get_model_name(response)
        assert name == "sonnet"

    def test_validate_json_schema_valid(
        self, claude_code_model: ClaudeCodeModel
    ):
        """Verify JSON schema validation passes for correct data."""
        schema = {
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        data = {"name": "Alice", "age": EXPECTED_ALICE_AGE}

        error = claude_code_model._validate_json_schema(data, schema)
        assert error is None

    def test_validate_json_schema_missing_field(
        self, claude_code_model: ClaudeCodeModel
    ):
        """Verify validation catches missing required fields."""
        schema = {
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        data = {"name": "Alice"}

        error = claude_code_model._validate_json_schema(data, schema)
        assert error is not None
        assert "age" in error

    def test_validate_json_schema_wrong_type(
        self, claude_code_model: ClaudeCodeModel
    ):
        """Verify validation catches type mismatches."""
        schema = {
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        }
        data = {"count": "not_a_number"}

        error = claude_code_model._validate_json_schema(data, schema)
        assert error is not None
        assert "count" in error

    def test_extract_json_robust_from_markdown(
        self, claude_code_model: ClaudeCodeModel
    ):
        """Verify JSON extraction from markdown code fences."""
        text = '```json\n{"name": "Alice", "age": 30}\n```'
        schema = {
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            }
        }

        result = claude_code_model._extract_json_robust(text, schema)
        assert result == {"name": "Alice", "age": 30}

    def test_extract_json_robust_from_plain_json(
        self, claude_code_model: ClaudeCodeModel
    ):
        """Verify JSON extraction from plain JSON text."""
        text = '{"name": "Bob", "age": 25}'
        schema = {
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            }
        }

        result = claude_code_model._extract_json_robust(text, schema)
        assert result == {"name": "Bob", "age": 25}

    def test_extract_json_robust_single_field_autowrap(
        self, claude_code_model: ClaudeCodeModel
    ):
        """Verify auto-wrapping of single-field schema."""
        text = "42"
        schema = {"properties": {"value": {"type": "integer"}}}

        result = claude_code_model._extract_json_robust(text, schema)
        assert result == {"value": 42}


# ===== Section 5: Response Conversion =====


class TestResponseConversion:
    """Test _convert_response method paths."""

    def test_convert_plain_text_response(
        self, claude_code_model: ClaudeCodeModel
    ):
        """Verify plain text response conversion."""
        response: ClaudeJSONResponse = {
            "result": "Hello, World!",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        settings: ClaudeCodeSettings = {"__working_directory": "/tmp"}

        model_response = claude_code_model._convert_response(
            response, output_tools=[], function_tools=[], settings=settings
        )

        assert len(model_response.parts) == 1
        assert isinstance(model_response.parts[0], TextPart)
        assert model_response.parts[0].content == "Hello, World!"

    def test_convert_function_selection_none(
        self, claude_code_model: ClaudeCodeModel
    ):
        """Verify 'none' function selection is parsed correctly."""
        response: ClaudeJSONResponse = {
            "result": "CHOICE: none",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        settings: ClaudeCodeSettings = {
            "__function_selection_mode__": True,
            "__available_functions__": {"add": "mock_tool"},
            "__working_directory": "/tmp",
        }

        model_response = claude_code_model._convert_response(
            response,
            output_tools=[],
            function_tools=["mock"],
            settings=settings,
        )

        assert len(model_response.parts) == 1
        part = model_response.parts[0]
        assert isinstance(part, TextPart)
        assert "none" in part.content.lower()
        assert settings.get("__function_selection_result__") == "none"

    def test_convert_function_selection_specific(
        self, claude_code_model: ClaudeCodeModel
    ):
        """Verify specific function selection is parsed."""
        response: ClaudeJSONResponse = {
            "result": "CHOICE: add",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        settings: ClaudeCodeSettings = {
            "__function_selection_mode__": True,
            "__available_functions__": {"add": "mock_tool"},
            "__working_directory": "/tmp",
        }

        claude_code_model._convert_response(
            response,
            output_tools=[],
            function_tools=["mock"],
            settings=settings,
        )

        assert settings.get("__function_selection_result__") == "selected"
        assert settings.get("__selected_function__") == "add"


# ===== Section 6: Message Formatting =====


class TestMessageFormatting:
    """Test message conversion between Pydantic AI and Claude CLI formats."""

    def test_format_simple_user_message(self):
        """Verify simple user message formatting."""
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content="Hello")])
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            result = format_messages_for_claude(messages, working_dir=tmpdir)

        assert "Request: Hello" in result

    def test_format_system_plus_user(self):
        """Verify system + user message ordering."""
        messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    SystemPromptPart(content="Be helpful"),
                    UserPromptPart(content="Question"),
                ]
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            result = format_messages_for_claude(messages, working_dir=tmpdir)

        assert result.index("System:") < result.index("Request:")

    def test_format_skip_system_prompt(self):
        """Verify skip_system_prompt flag works."""
        messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    SystemPromptPart(content="System instruction"),
                    UserPromptPart(content="User question"),
                ]
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            result = format_messages_for_claude(
                messages, skip_system_prompt=True, working_dir=tmpdir
            )

        assert "System:" not in result
        assert "Request: User question" in result

    def test_format_tool_return_creates_file(self):
        """Verify tool return content is written to file."""
        messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="search",
                        content="Found 42 results",
                        tool_call_id="call_abc",
                    )
                ]
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            result = format_messages_for_claude(messages, working_dir=tmpdir)

            # Verify file reference in prompt
            assert "tool_result_1_search.txt" in result

            # Verify file was created with correct content
            import os

            file_path = os.path.join(tmpdir, "tool_result_1_search.txt")
            assert os.path.exists(file_path)
            with open(file_path) as f:
                assert f.read() == "Found 42 results"

    def test_format_multi_turn_conversation(self):
        """Verify multi-turn conversation formatting."""
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content="Q1")]),
            ModelResponse(parts=[TextPart(content="A1")]),
            ModelRequest(parts=[UserPromptPart(content="Q2")]),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            result = format_messages_for_claude(messages, working_dir=tmpdir)

        assert "Request: Q1" in result
        assert "Assistant: A1" in result
        assert "Request: Q2" in result

    def test_extract_text_from_response_with_prefix(self):
        """Verify Assistant: prefix is stripped."""
        assert extract_text_from_response("Assistant: Hello") == "Hello"

    def test_extract_text_from_response_without_prefix(self):
        """Verify text without prefix passes through."""
        assert extract_text_from_response("Hello world") == "Hello world"


# ===== Section 7: Conversation Context =====


class TestConversationContext:
    """Test conversation context analysis."""

    def test_empty_context(self):
        """Verify empty message list context."""
        ctx = build_conversation_context([])
        assert ctx["num_messages"] == 0
        assert ctx["has_system_prompt"] is False

    def test_full_conversation_context(self):
        """Verify context counts for complex conversation."""
        messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    SystemPromptPart(content="System"),
                    UserPromptPart(content="User 1"),
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(content="Response 1"),
                    ToolCallPart(
                        tool_name="tool",
                        args={},
                        tool_call_id="c1",
                    ),
                ]
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="tool",
                        content="result",
                        tool_call_id="c1",
                    ),
                    UserPromptPart(content="User 2"),
                ]
            ),
            ModelResponse(parts=[TextPart(content="Final")]),
        ]

        ctx = build_conversation_context(messages)

        assert ctx["num_messages"] == 4
        assert ctx["has_system_prompt"] is True
        assert ctx["num_user_messages"] == 2
        assert ctx["num_assistant_messages"] == 2
        assert ctx["num_tool_calls"] == 1
        assert ctx["num_tool_returns"] == 1


# ===== Section 8: Response Utils =====


class TestResponseUtils:
    """Test response utility functions."""

    def test_create_tool_call_part(self):
        """Verify tool call part creation with unique IDs."""
        part = create_tool_call_part("my_tool", {"x": 1})

        assert isinstance(part, ToolCallPart)
        assert part.tool_name == "my_tool"
        assert part.args == {"x": 1}
        assert part.tool_call_id.startswith("call_")

    def test_create_tool_call_unique_ids(self):
        """Verify each tool call gets a unique ID."""
        part1 = create_tool_call_part("tool", {})
        part2 = create_tool_call_part("tool", {})

        assert part1.tool_call_id != part2.tool_call_id

    def test_extract_model_parameters_none(self):
        """Verify None parameters return empty lists."""
        output, function = extract_model_parameters(None)
        assert output == []
        assert function == []

    def test_extract_model_parameters_populated(self):
        """Verify parameters extraction from request params."""
        params = ModelRequestParameters()
        output, function = extract_model_parameters(params)
        # Default ModelRequestParameters has empty tool lists
        assert output == []
        assert function == []

    def test_get_working_directory_from_settings(self):
        """Verify working directory extraction."""
        settings: ClaudeCodeSettings = {
            "__working_directory": "/my/custom/dir"
        }
        assert get_working_directory(settings) == "/my/custom/dir"

    def test_get_working_directory_default(self):
        """Verify default working directory."""
        settings: ClaudeCodeSettings = {}
        assert get_working_directory(settings) == "/tmp"

    def test_get_working_directory_custom_default(self):
        """Verify custom default value."""
        settings: ClaudeCodeSettings = {}
        assert (
            get_working_directory(settings, default="/custom")
            == "/custom"
        )


# ===== Section 9: Provider Configuration =====


class TestProviderConfiguration:
    """Test ClaudeCodeProvider setup and model creation."""

    def test_provider_creation(self):
        """Verify basic provider creation."""
        provider = ClaudeCodeProvider()
        assert provider.name == "claude-code"
        assert provider.cli_path is None

    def test_provider_with_cli_path(self):
        """Verify provider with custom CLI path."""
        provider = ClaudeCodeProvider(cli_path="/usr/local/bin/claude")
        assert provider.cli_path == "/usr/local/bin/claude"

    def test_provider_creates_model(self):
        """Verify model creation from provider."""
        provider = ClaudeCodeProvider()
        model = provider.create_model("sonnet")

        assert isinstance(model, ClaudeCodeModel)
        assert model.model_name == "claude-code:sonnet"

    def test_provider_creates_model_with_preset(self):
        """Verify model creation with provider preset."""
        provider = ClaudeCodeProvider()
        model = provider.create_model("sonnet", provider_preset="deepseek")

        assert isinstance(model, ClaudeCodeModel)
        assert model.model_name == "claude-code:deepseek:sonnet"

    def test_provider_repr(self):
        """Verify string representation."""
        provider = ClaudeCodeProvider()
        assert "ClaudeCodeProvider" in repr(provider)

    def test_model_from_string_format(self):
        """Verify model string 'claude-code:sonnet' resolves correctly."""
        # This tests the registration pathway
        agent = Agent("claude-code:sonnet")
        assert agent.model is not None


# ===== Section 10: Error Handling Paths =====


class TestErrorHandlingPaths:
    """Test error detection and handling in the interaction layer."""

    def test_oauth_error_detection(self):
        """Verify OAuth error detection from JSON CLI output."""
        # Simulate a JSON error response from Claude CLI with token expired
        import json

        from pydantic_ai_claude_code.core.oauth_handler import (
            detect_oauth_error,
        )

        stdout = json.dumps({
            "is_error": True,
            "result": "Error: token expired. Please run /login to re-authenticate.",
        })
        is_oauth, msg = detect_oauth_error(stdout, "")
        assert is_oauth is True
        assert msg is not None
        assert "token expired" in msg.lower()

    def test_no_oauth_error(self):
        """Verify no false positive OAuth detection."""
        from pydantic_ai_claude_code.core.oauth_handler import (
            detect_oauth_error,
        )

        is_oauth, msg = detect_oauth_error("Normal response text", "")
        assert is_oauth is False
        assert msg is None

    def test_rate_limit_detection(self):
        """Verify rate limit detection with proper pattern."""
        from pydantic_ai_claude_code.core.retry_logic import (
            detect_rate_limit,
        )

        # detect_rate_limit expects "limit reached.*resets <TIME>" pattern
        error_text = "Rate limit reached for model. resets 3PM"
        is_limited, reset_time = detect_rate_limit(error_text)
        assert is_limited is True
        assert reset_time == "3PM"

    def test_no_rate_limit(self):
        """Verify no false positive rate limit detection."""
        from pydantic_ai_claude_code.core.retry_logic import (
            detect_rate_limit,
        )

        is_limited, reset_time = detect_rate_limit("Everything is fine")
        assert is_limited is False
        assert reset_time is None

    def test_cli_infrastructure_failure_detection(self):
        """Verify CLI infrastructure failure detection."""
        from pydantic_ai_claude_code.core.retry_logic import (
            detect_cli_infrastructure_failure,
        )

        # Must match actual patterns: Cannot find module, MODULE_NOT_FOUND, ENOENT, EACCES
        assert detect_cli_infrastructure_failure("Cannot find module 'yoga.wasm'") is True
        assert detect_cli_infrastructure_failure("MODULE_NOT_FOUND") is True
        assert detect_cli_infrastructure_failure("ENOENT: no such file") is True
        assert detect_cli_infrastructure_failure("EACCES: permission denied") is True
        assert detect_cli_infrastructure_failure("Normal output") is False

    def test_calculate_wait_time(self):
        """Verify wait time calculation from reset time string."""
        from pydantic_ai_claude_code.core.retry_logic import (
            calculate_wait_time,
        )

        # calculate_wait_time takes a time string like "3PM"
        wait = calculate_wait_time("3PM")
        # Should return a non-negative integer (seconds until 3PM + 1 min buffer)
        assert isinstance(wait, int)
        assert wait >= 0

        # Invalid time string should return fallback (300 seconds)
        wait_fallback = calculate_wait_time("invalid")
        assert wait_fallback == 300


# ===== Section 11: Settings Merge Logic =====


class TestSettingsMerge:
    """Test the three-layer settings merge in ClaudeCodeModel._build_options."""

    def test_defaults_layer(self, claude_code_model: ClaudeCodeModel):
        """Verify default settings are the base layer."""
        settings = claude_code_model._build_options(
            None, ModelRequestParameters()
        )
        assert settings["model"] == "sonnet"
        assert settings["dangerously_skip_permissions"] is True

    def test_model_settings_override_defaults(
        self, claude_code_model: ClaudeCodeModel
    ):
        """Verify model_settings overrides defaults."""
        settings = claude_code_model._build_options(
            {"timeout_seconds": 60}, ModelRequestParameters()
        )
        assert settings["timeout_seconds"] == 60

    def test_extra_cli_args_passthrough(
        self, claude_code_model: ClaudeCodeModel
    ):
        """Verify extra CLI args are passed through."""
        settings = claude_code_model._build_options(
            {"extra_cli_args": ["--no-color"]},
            ModelRequestParameters(),
        )
        assert settings["extra_cli_args"] == ["--no-color"]

    def test_debug_save_prompts_setting(
        self, claude_code_model: ClaudeCodeModel
    ):
        """Verify debug_save_prompts is passed through."""
        settings = claude_code_model._build_options(
            {"debug_save_prompts": "/debug/output"},
            ModelRequestParameters(),
        )
        assert settings["debug_save_prompts"] == "/debug/output"


# ===== Section 12: Structured Output File Handling =====


class TestStructuredOutputFileHandling:
    """Test structured output file read/write paths."""

    def test_read_structured_output_json_file(
        self, claude_code_model: ClaudeCodeModel
    ):
        """Verify reading valid JSON from structured output file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"name": "Alice", "age": 30}, f)
            f.flush()
            filepath = f.name

        schema = {
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }

        data, error = claude_code_model._read_structured_output_file(
            filepath, schema, {}
        )

        assert error is None
        assert data == {"name": "Alice", "age": 30}

    def test_read_structured_output_missing_file(
        self, claude_code_model: ClaudeCodeModel
    ):
        """Verify handling of missing output file."""
        data, error = claude_code_model._read_structured_output_file(
            "/nonexistent/file.json",
            {"properties": {}},
            {},
        )

        assert data is None
        assert error is None  # Missing file returns None, None

    def test_read_structured_output_invalid_json(
        self, claude_code_model: ClaudeCodeModel
    ):
        """Verify handling of invalid JSON content."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("not valid json {{{")
            f.flush()
            filepath = f.name

        data, error = claude_code_model._read_structured_output_file(
            filepath, {"properties": {}}, {}
        )

        assert data is None
        assert error is not None
        assert "formatted correctly" in error.lower() or "content" in error.lower()


# ===== Section 13: Smoke Test =====


class TestSmokeTest:
    """Smoke test verifying the public API can be imported and trivial calls succeed.

    Per testing instructions: "For libraries: assert the public API can be
    imported and a trivial call succeeds."
    """

    def test_public_api_importable(self):
        """Verify all public API symbols are importable."""
        import pydantic_ai_claude_code as pkg

        public_symbols = [
            "ClaudeCodeModel",
            "ClaudeCodeProvider",
            "ClaudeCodeSettings",
            "ClaudeOAuthError",
            "MCPTool",
            "ProviderPreset",
            "__version__",
            "calculate_wait_time",
            "detect_cli_infrastructure_failure",
            "detect_oauth_error",
            "detect_rate_limit",
            "get_preset",
            "get_presets_by_category",
            "list_presets",
            "load_all_presets",
        ]

        for symbol in public_symbols:
            assert hasattr(pkg, symbol), f"Missing public symbol: {symbol}"
            assert getattr(pkg, symbol) is not None, f"Symbol is None: {symbol}"

    def test_version_is_valid(self):
        """Verify package version is a non-empty string."""
        from pydantic_ai_claude_code import __version__

        assert isinstance(__version__, str)
        assert len(__version__) > 0

    def test_trivial_model_instantiation(self):
        """Verify ClaudeCodeModel can be instantiated without side effects."""
        model = ClaudeCodeModel(model_name="sonnet")
        assert model.model_name == "claude-code:sonnet"
        assert model.system == "claude-code"

    def test_trivial_provider_instantiation(self):
        """Verify ClaudeCodeProvider can be instantiated without side effects."""
        provider = ClaudeCodeProvider()
        assert provider.name == "claude-code"

    def test_trivial_agent_creation(self):
        """Verify Agent with claude-code model string can be created."""
        agent = Agent("claude-code:sonnet")
        assert agent is not None

    def test_presets_loadable(self):
        """Verify provider presets can be loaded from YAML."""
        from pydantic_ai_claude_code import list_presets

        presets = list_presets()
        assert isinstance(presets, list)
        assert len(presets) > 0

    def test_trivial_run_with_test_model(self):
        """Verify a trivial agent.run succeeds end-to-end with TestModel."""
        agent = Agent("claude-code:sonnet")
        m = TestModel(custom_output_text="smoke-ok")

        with agent.override(model=m):
            result = agent.run_sync("ping")

        assert result.output == "smoke-ok"
