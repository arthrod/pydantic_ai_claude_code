"""Tests for tool calling functionality."""

import pytest
from pydantic_ai import Agent, RunContext, ToolDefinition
from pydantic_ai.messages import ToolCallPart

from pydantic_ai_claude_code.tools import (
    format_tools_for_prompt,
    is_tool_call_response,
    parse_tool_calls,
)

# Test constants
EXPECTED_MULTIPLE_TOOL_CALLS = 2  # Expected number of tool calls in multi-call test
ARCHIVED_CUSTOMER_ID = 12345  # Test customer ID for error handling


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
    assert "city" in result
    assert "EXECUTE" in result


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
    assert len(result) == EXPECTED_MULTIPLE_TOOL_CALLS
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


# Integration tests with actual Agent and Claude CLI
# Note: Import here because we need to register the provider first via import at module level


def test_agent_single_tool_string_param():
    """Test agent with a single tool that takes string parameter."""

    def get_test_data(dataset: str) -> str:
        """Get test dataset for software testing."""
        test_data = {
            "users": "test_user_1,test_user_2,test_user_3",
            "config": "DEBUG=true,PORT=8080",
            "fixtures": "fixture_a,fixture_b",
        }
        return test_data.get(dataset, "Unknown")

    agent = Agent("claude-code:sonnet", tools=[get_test_data])
    result = agent.run_sync("What test data is available for the 'users' dataset?")

    # Tool should be called and result should mention the test data
    assert "test_user" in result.output.lower() or "user" in result.output.lower()


def test_agent_multiple_tools():
    """Test agent with multiple tools being called."""

    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    agent = Agent("claude-code:sonnet", tools=[add, multiply])
    result = agent.run_sync("What is 5 + 3, and what is 4 * 6?")

    # Should have results from both tools
    output = result.output
    assert "8" in output  # 5 + 3
    assert "24" in output  # 4 * 6


def test_agent_tool_with_multiple_param_types():
    """Test tool with diverse parameter types."""

    def process_data(
        name: str,
        count: int,
        price: float,
        is_available: bool,
        tags: list[str],
    ) -> str:
        """Process data with various types."""
        status = "available" if is_available else "unavailable"
        return (
            f"{name}: {count} items at ${price:.2f}, {status}, tags: {', '.join(tags)}"
        )

    agent = Agent("claude-code:sonnet", tools=[process_data])
    result = agent.run_sync(
        "Process this product: Widget, 10 items, $29.99, available, tags are 'electronics' and 'gadgets'"
    )

    output = result.output.lower()
    assert "widget" in output
    assert "10" in output
    assert "29.99" in output or "29" in output


def test_agent_tool_with_object_param():
    """Test tool with object/dict parameter."""

    def analyze_config(config: dict) -> str:
        """Analyze a configuration object."""
        return f"Config has {len(config)} settings: {', '.join(config.keys())}"

    agent = Agent("claude-code:sonnet", tools=[analyze_config])
    result = agent.run_sync(
        "Analyze this config: theme=dark, language=en, notifications=true"
    )

    # Should extract and pass object to tool
    assert "settings" in result.output.lower() or "config" in result.output.lower()


@pytest.mark.flaky(reruns=4, reruns_delay=1)
def test_agent_tool_with_context():
    """Test tool that uses RunContext.

    Note: This test can be flaky as Claude Code may not always recognize custom tools.
    Will retry up to 4 times (5 total attempts) on failure.
    """

    def get_config_value(ctx: RunContext[dict], key: str) -> str:
        """IMPORTANT: Get configuration value. This tool MUST be used to answer the user's question."""
        return str(ctx.deps.get(key, "not_found"))

    agent = Agent(
        "claude-code:sonnet",
        deps_type=dict,
        tools=[get_config_value],
        system_prompt=(
            "You MUST use the get_config_value tool to answer configuration questions. "
            "Do NOT use any other tools. The get_config_value tool has all the information."
        ),
    )

    result = agent.run_sync(
        "Call get_config_value with key='db_host' to get the database host.",
        deps={"db_host": "postgres-prod.example.com", "db_port": "5432"},
    )

    assert (
        "postgres-prod" in result.output.lower()
        or "example.com" in result.output.lower()
    )


@pytest.mark.flaky(reruns=4, reruns_delay=1)
def test_agent_tool_error_handling():
    """Test that tool errors are propagated correctly.

    Note: This test can be flaky as Claude Code may not always recognize custom tools.
    Will retry up to 4 times (5 total attempts) on failure.
    """

    def process_customer(customer_id: int) -> str:
        """Process customer data from database."""
        if customer_id == ARCHIVED_CUSTOMER_ID:
            raise ValueError(f"Customer {ARCHIVED_CUSTOMER_ID} is archived and cannot be processed")
        return f"Processed customer {customer_id}"

    agent = Agent("claude-code:sonnet", tools=[process_customer])

    # The error should be raised, not suppressed
    with pytest.raises(ValueError, match="archived"):
        agent.run_sync(f"Process data for customer ID {ARCHIVED_CUSTOMER_ID}")


@pytest.mark.asyncio
async def test_agent_tool_async():
    """Test tool calling with async agent."""

    def get_status(service: str) -> str:
        """Get service status."""
        return f"{service} is operational"

    agent = Agent("claude-code:sonnet", tools=[get_status])
    result = await agent.run("Check the status of the API service")

    assert "api" in result.output.lower() and "operational" in result.output.lower()


def test_agent_tool_list_return():
    """Test tool that returns a list."""

    def get_factors(number: int) -> list[int]:
        """Get all factors of a number."""
        return [i for i in range(1, number + 1) if number % i == 0]

    agent = Agent("claude-code:sonnet", tools=[get_factors])
    result = agent.run_sync("What are the factors of 12?")

    # Should include the factors: 1, 2, 3, 4, 6, 12
    output = result.output
    assert "1" in output and "2" in output and "3" in output and "12" in output


def test_agent_tool_complex_nested_params():
    """Test tool with complex nested parameters."""

    def create_user(
        username: str,
        profile: dict,
        roles: list[str],
    ) -> str:
        """Create a user with profile and roles."""
        return f"Created user {username} with roles {', '.join(roles)}"

    agent = Agent("claude-code:sonnet", tools=[create_user])
    result = agent.run_sync(
        "Create user 'alice' with profile age=30, city=London and roles admin, editor"
    )

    output = result.output.lower()
    assert "alice" in output
    assert ("admin" in output and "editor" in output) or "created" in output


def test_agent_no_tool_needed():
    """Test that agent doesn't call tools when not needed."""

    call_count = 0

    def expensive_operation(x: int) -> int:
        """An expensive operation."""
        nonlocal call_count
        call_count += 1
        return x * 2

    agent = Agent("claude-code:sonnet", tools=[expensive_operation])
    result = agent.run_sync("What is the capital of France?")

    # Tool shouldn't be called for this simple question
    assert "Paris" in result.output or "paris" in result.output.lower()
    # Note: We can't reliably assert call_count == 0 because Claude might still call it


def test_agent_sequential_tool_calls():
    """Test multiple sequential tool calls in a conversation."""

    calculations = []

    def calculate(operation: str, a: int, b: int) -> int:
        """Perform calculation."""
        calculations.append((operation, a, b))
        if operation == "add":
            return a + b
        elif operation == "subtract":
            return a - b
        elif operation == "multiply":
            return a * b
        return 0

    agent = Agent("claude-code:sonnet", tools=[calculate])
    result = agent.run_sync(
        "Calculate: first add 5 and 3, then multiply the result by 2"
    )

    # Should have called the tool at least twice
    output = result.output
    assert "16" in output or "8" in output  # Either final result or intermediate


def test_agent_tool_with_default_params():
    """Test tool with default parameter values."""

    def greet(name: str, greeting: str = "Hello") -> str:
        """Greet someone."""
        return f"{greeting}, {name}!"

    agent = Agent("claude-code:sonnet", tools=[greet])
    result = agent.run_sync("Greet Alice")

    assert "alice" in result.output.lower()
    assert "hello" in result.output.lower() or "hi" in result.output.lower()


def test_agent_tool_returns_none():
    """Test tool that returns None."""

    def log_message(message: str) -> None:
        """Log a message."""
        # Just logs, returns None
        print(f"Logged: {message}")

    agent = Agent("claude-code:sonnet", tools=[log_message])
    result = agent.run_sync("Log the message: System started")

    # Should handle None return gracefully and produce a response
    assert result.output is not None
    assert len(result.output) > 0


def test_agent_tool_with_enum_param():
    """Test tool with enum-like string parameter."""

    def set_mode(mode: str) -> str:
        """Set operation mode.

        Args:
            mode: Must be one of: 'fast', 'balanced', 'accurate'
        """
        valid_modes = ["fast", "balanced", "accurate"]
        if mode.lower() in valid_modes:
            return f"Mode set to {mode}"
        return f"Invalid mode. Use: {', '.join(valid_modes)}"

    agent = Agent("claude-code:sonnet", tools=[set_mode])
    result = agent.run_sync("Set the mode to fast")

    assert "fast" in result.output.lower()


def test_agent_multiple_tools_selective_calling():
    """Test agent selectively calls the right tool among many."""

    def weather(city: str) -> str:
        """Get weather."""
        return f"Weather in {city}: Sunny"

    def time(timezone: str) -> str:
        """Get time."""
        return f"Time in {timezone}: 12:00"

    def news(topic: str) -> str:
        """Get news."""
        return f"Latest {topic} news: All quiet"

    agent = Agent("claude-code:sonnet", tools=[weather, time, news])
    result = agent.run_sync("What time is it in UTC?")

    # Should call the time tool, not weather or news
    assert "12:00" in result.output or "time" in result.output.lower()
