"""Tests for CCTools class."""

import pytest
from pydantic_ai_claude_code.cc_tools import CCTools, ToolDefinition, PermissionResult


class TestToolRegistration:
    """Test tool registration functionality."""

    def test_register_tool_basic(self):
        """Test basic tool registration."""
        tools = CCTools()

        tool = tools.register_tool(
            name="test_tool",
            description="A test tool",
            parameters_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        )

        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.permission_mode == "ask"
        assert "test_tool" in [t.name for t in tools.get_all_tools()]

    def test_register_tool_with_permission_mode(self):
        """Test tool registration with different permission modes."""
        tools = CCTools()

        tools.register_tool(
            name="allowed_tool",
            description="Auto-allowed",
            parameters_schema={},
            permission_mode="allow",
        )

        tools.register_tool(
            name="denied_tool",
            description="Auto-denied",
            parameters_schema={},
            permission_mode="deny",
        )

        assert "allowed_tool" in tools.get_allowed_tools()
        assert "denied_tool" in tools.get_disallowed_tools()
        assert "denied_tool" not in tools.get_allowed_tools()

    def test_decorator_registration(self):
        """Test tool registration via decorator."""
        tools = CCTools()

        @tools.tool(
            name="decorated_tool",
            description="Decorated tool",
            schema={"type": "object"},
        )
        async def my_handler(x: str) -> str:
            return f"processed: {x}"

        tool = tools.get_tool("decorated_tool")
        assert tool is not None
        assert tool.handler is not None

    def test_get_all_tools(self):
        """Test getting all registered tools."""
        tools = CCTools()

        tools.register_tool("tool1", "Desc 1", {})
        tools.register_tool("tool2", "Desc 2", {})
        tools.register_tool("tool3", "Desc 3", {})

        all_tools = tools.get_all_tools()
        assert len(all_tools) == 3


class TestPermissions:
    """Test permission handling."""

    @pytest.mark.asyncio
    async def test_can_use_tool_unknown(self):
        """Test permission check for unknown tool."""
        tools = CCTools()

        result = await tools.can_use_tool("unknown", {})

        assert result.behavior == "deny"
        assert "Unknown tool" in result.message

    @pytest.mark.asyncio
    async def test_can_use_tool_allow_mode(self):
        """Test permission check for allow mode tool."""
        tools = CCTools()

        tools.register_tool(
            name="auto_allow",
            description="Auto-allowed tool",
            parameters_schema={},
            permission_mode="allow",
        )

        result = await tools.can_use_tool("auto_allow", {})

        assert result.behavior == "allow"

    @pytest.mark.asyncio
    async def test_can_use_tool_deny_mode(self):
        """Test permission check for deny mode tool."""
        tools = CCTools()

        tools.register_tool(
            name="auto_deny",
            description="Auto-denied tool",
            parameters_schema={},
            permission_mode="deny",
        )

        result = await tools.can_use_tool("auto_deny", {})

        assert result.behavior == "deny"
        assert "disabled" in result.message

    @pytest.mark.asyncio
    async def test_can_use_tool_with_callback(self):
        """Test permission check with custom callback."""
        tools = CCTools()

        tools.register_tool(
            name="ask_tool",
            description="Ask permission",
            parameters_schema={},
            permission_mode="ask",
        )

        # Set custom callback that always allows
        async def custom_callback(name, input, context):
            return PermissionResult(
                behavior="allow",
                message="Custom allowed",
            )

        tools.set_permission_callback(custom_callback)

        result = await tools.can_use_tool("ask_tool", {})

        assert result.behavior == "allow"
        assert result.message == "Custom allowed"

    @pytest.mark.asyncio
    async def test_can_use_tool_sanitizes_input(self):
        """Test that permission check sanitizes input."""
        tools = CCTools()

        tools.register_tool(
            name="typed_tool",
            description="Tool with typed params",
            parameters_schema={
                "type": "object",
                "properties": {
                    "count": {"type": "integer"},
                    "name": {"type": "string"},
                },
            },
            permission_mode="allow",
        )

        result = await tools.can_use_tool(
            "typed_tool",
            {"count": "42", "name": 123},  # Wrong types
        )

        assert result.behavior == "allow"
        # Input should be coerced
        assert result.updated_input["count"] == 42
        assert result.updated_input["name"] == "123"


class TestToolExecution:
    """Test tool execution."""

    @pytest.mark.asyncio
    async def test_execute_tool_basic(self):
        """Test basic tool execution."""
        tools = CCTools()

        @tools.tool(
            name="adder",
            description="Add numbers",
            schema={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
            },
        )
        async def add(a: int, b: int) -> int:
            return a + b

        result = await tools.execute_tool("adder", {"a": 2, "b": 3})
        assert result == 5

    @pytest.mark.asyncio
    async def test_execute_tool_records_history(self):
        """Test that tool execution is recorded in history."""
        tools = CCTools()

        @tools.tool(
            name="greeter",
            description="Greet someone",
            schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
            },
        )
        async def greet(name: str) -> str:
            return f"Hello, {name}!"

        await tools.execute_tool("greeter", {"name": "World"})

        history = tools.get_history()
        assert len(history) == 1
        assert history[0]["tool"] == "greeter"
        assert history[0]["args"] == {"name": "World"}
        assert history[0]["result"] == "Hello, World!"
        assert history[0]["success"] is True

    @pytest.mark.asyncio
    async def test_execute_tool_unknown_raises(self):
        """Test that executing unknown tool raises."""
        tools = CCTools()

        with pytest.raises(ValueError, match="Unknown tool"):
            await tools.execute_tool("nonexistent", {})

    @pytest.mark.asyncio
    async def test_execute_tool_no_handler_raises(self):
        """Test that executing tool without handler raises."""
        tools = CCTools()

        tools.register_tool(
            name="no_handler",
            description="Tool without handler",
            parameters_schema={},
            handler=None,
        )

        with pytest.raises(ValueError, match="No handler"):
            await tools.execute_tool("no_handler", {})

    @pytest.mark.asyncio
    async def test_execute_tool_error_recorded(self):
        """Test that tool errors are recorded in history."""
        tools = CCTools()

        @tools.tool(
            name="failer",
            description="Always fails",
            schema={},
        )
        async def fail() -> str:
            raise ValueError("Intentional failure")

        with pytest.raises(ValueError, match="Intentional failure"):
            await tools.execute_tool("failer", {})

        history = tools.get_history()
        assert len(history) == 1
        assert history[0]["success"] is False
        assert "Intentional failure" in history[0]["error"]


class TestTypeCoercion:
    """Test type coercion functionality."""

    def test_coerce_to_string(self):
        """Test coercion to string."""
        tools = CCTools()

        schema = {"type": "string"}
        assert tools._coerce_value(123, schema) == "123"
        assert tools._coerce_value(True, schema) == "True"

    def test_coerce_to_integer(self):
        """Test coercion to integer."""
        tools = CCTools()

        schema = {"type": "integer"}
        assert tools._coerce_value("42", schema) == 42
        assert tools._coerce_value(3.14, schema) == 3

    def test_coerce_to_number(self):
        """Test coercion to number."""
        tools = CCTools()

        schema = {"type": "number"}
        assert tools._coerce_value("3.14", schema) == 3.14
        assert tools._coerce_value(42, schema) == 42.0

    def test_coerce_to_boolean(self):
        """Test coercion to boolean."""
        tools = CCTools()

        schema = {"type": "boolean"}
        assert tools._coerce_value("true", schema) is True
        assert tools._coerce_value("false", schema) is False
        assert tools._coerce_value("yes", schema) is True
        assert tools._coerce_value(1, schema) is True
        assert tools._coerce_value(0, schema) is False

    def test_coerce_to_array(self):
        """Test coercion to array."""
        tools = CCTools()

        schema = {"type": "array"}
        assert tools._coerce_value("single", schema) == ["single"]
        assert tools._coerce_value([1, 2, 3], schema) == [1, 2, 3]

    def test_coerce_invalid_returns_original(self):
        """Test that invalid coercion returns original."""
        tools = CCTools()

        schema = {"type": "integer"}
        # Can't convert "not a number" to int, returns original
        assert tools._coerce_value("not a number", schema) == "not a number"


class TestPydanticAIConversion:
    """Test conversion to pydantic_ai format."""

    def test_to_pydantic_ai_tools(self):
        """Test conversion to pydantic_ai tool format."""
        tools = CCTools()

        tools.register_tool(
            name="tool1",
            description="First tool",
            parameters_schema={"type": "object"},
            permission_mode="allow",
        )

        tools.register_tool(
            name="tool2",
            description="Second tool",
            parameters_schema={"type": "string"},
            permission_mode="deny",  # Should be excluded
        )

        pydantic_tools = tools.to_pydantic_ai_tools()

        assert len(pydantic_tools) == 1
        assert pydantic_tools[0]["name"] == "tool1"
        assert pydantic_tools[0]["description"] == "First tool"
        assert pydantic_tools[0]["parameters_json_schema"] == {"type": "object"}


class TestHistory:
    """Test history management."""

    @pytest.mark.asyncio
    async def test_clear_history(self):
        """Test clearing history."""
        tools = CCTools()

        @tools.tool(name="test", description="Test", schema={})
        async def test_func():
            return "ok"

        await tools.execute_tool("test", {})
        assert len(tools.get_history()) == 1

        tools.clear_history()
        assert len(tools.get_history()) == 0

    @pytest.mark.asyncio
    async def test_history_is_copy(self):
        """Test that get_history returns a copy."""
        tools = CCTools()

        @tools.tool(name="test", description="Test", schema={})
        async def test_func():
            return "ok"

        await tools.execute_tool("test", {})

        history = tools.get_history()
        history.append({"fake": "entry"})

        # Original should be unchanged
        assert len(tools.get_history()) == 1
