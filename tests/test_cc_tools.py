"""Tests for CCTools advanced tool management."""

import pytest

from pydantic_ai_claude_code import CCTools, ToolDefinition
from pydantic_ai_claude_code.sdk_original_files.types import ToolPermissionContext


class TestToolDefinition:
    """Tests for ToolDefinition class."""

    def test_tool_definition_creation(self):
        """Test basic tool definition creation."""
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters_schema={"type": "object", "properties": {}},
        )

        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.permission_mode == "ask"
        assert tool.handler is None

    def test_tool_definition_with_handler(self):
        """Test tool definition with handler."""

        async def handler(x: int) -> int:
            return x * 2

        tool = ToolDefinition(
            name="double",
            description="Double a number",
            parameters_schema={"type": "object"},
            handler=handler,
        )

        assert tool.handler is handler

    def test_tool_to_sdk_format(self):
        """Test conversion to SDK format."""
        tool = ToolDefinition(
            name="search",
            description="Search for information",
            parameters_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        )

        sdk_format = tool.to_sdk_format()

        assert sdk_format["name"] == "search"
        assert sdk_format["description"] == "Search for information"
        assert "input_schema" in sdk_format


class TestCCToolsBasic:
    """Basic tests for CCTools class."""

    def test_cctools_initialization(self):
        """Test CCTools initialization."""
        tools = CCTools()

        assert len(tools._tools) == 0
        assert len(tools._tool_history) == 0

    def test_register_tool(self):
        """Test tool registration."""
        tools = CCTools()

        tool = tools.register_tool(
            name="test",
            description="Test tool",
            parameters_schema={"type": "object"},
        )

        assert tool.name == "test"
        assert "test" in tools._tools
        assert tools.get_tool("test") is tool

    def test_register_multiple_tools(self):
        """Test registering multiple tools."""
        tools = CCTools()

        tools.register_tool("tool1", "First tool", {"type": "object"})
        tools.register_tool("tool2", "Second tool", {"type": "object"})
        tools.register_tool("tool3", "Third tool", {"type": "object"})

        assert len(tools.list_tools()) == 3
        assert "tool1" in tools.list_tools()
        assert "tool2" in tools.list_tools()
        assert "tool3" in tools.list_tools()

    def test_tool_decorator(self):
        """Test tool decorator syntax."""
        tools = CCTools()

        @tools.tool("calculate", "Perform calculation", {"type": "object"})
        async def calculate(x: int, y: int) -> int:
            return x + y

        assert "calculate" in tools._tools
        tool = tools.get_tool("calculate")
        assert tool is not None
        assert tool.handler is calculate


class TestCCToolsPermissions:
    """Tests for CCTools permission handling."""

    def test_get_allowed_tools(self):
        """Test getting allowed tools."""
        tools = CCTools()

        tools.register_tool("allowed1", "Allowed", {"type": "object"}, permission_mode="allow")
        tools.register_tool("allowed2", "Allowed", {"type": "object"}, permission_mode="ask")
        tools.register_tool("denied", "Denied", {"type": "object"}, permission_mode="deny")

        allowed = tools.get_allowed_tools()

        assert "allowed1" in allowed
        assert "allowed2" in allowed
        assert "denied" not in allowed

    def test_get_disallowed_tools(self):
        """Test getting disallowed tools."""
        tools = CCTools()

        tools.register_tool("allowed", "Allowed", {"type": "object"}, permission_mode="allow")
        tools.register_tool("denied", "Denied", {"type": "object"}, permission_mode="deny")

        disallowed = tools.get_disallowed_tools()

        assert "denied" in disallowed
        assert "allowed" not in disallowed

    @pytest.mark.asyncio
    async def test_can_use_tool_allow(self):
        """Test permission check for allowed tool."""
        tools = CCTools()
        tools.register_tool("test", "Test", {"type": "object"}, permission_mode="allow")

        context: ToolPermissionContext = {
            "session_id": "test-session",
            "turn_number": 1,
            "tool_history": [],
            "working_directory": "/tmp",
        }

        result = await tools.can_use_tool("test", {"arg": "value"}, context)

        assert result["behavior"] == "allow"

    @pytest.mark.asyncio
    async def test_can_use_tool_deny(self):
        """Test permission check for denied tool."""
        tools = CCTools()
        tools.register_tool("test", "Test", {"type": "object"}, permission_mode="deny")

        context: ToolPermissionContext = {
            "session_id": "test-session",
            "turn_number": 1,
            "tool_history": [],
            "working_directory": "/tmp",
        }

        result = await tools.can_use_tool("test", {"arg": "value"}, context)

        assert result["behavior"] == "deny"
        assert "disabled" in result.get("message", "")

    @pytest.mark.asyncio
    async def test_can_use_tool_unknown(self):
        """Test permission check for unknown tool."""
        tools = CCTools()

        context: ToolPermissionContext = {
            "session_id": "test-session",
            "turn_number": 1,
            "tool_history": [],
            "working_directory": "/tmp",
        }

        result = await tools.can_use_tool("unknown", {}, context)

        assert result["behavior"] == "deny"
        assert "Unknown" in result.get("message", "")

    @pytest.mark.asyncio
    async def test_custom_permission_callback(self):
        """Test custom permission callback."""
        tools = CCTools()
        tools.register_tool("test", "Test", {"type": "object"}, permission_mode="ask")

        async def custom_callback(tool_name, tool_input, context):
            if tool_input.get("blocked"):
                return {"behavior": "deny", "message": "Blocked by callback"}
            return {"behavior": "allow", "updated_input": None}

        tools.set_permission_callback(custom_callback)

        context: ToolPermissionContext = {
            "session_id": "test",
            "turn_number": 1,
            "tool_history": [],
            "working_directory": "/tmp",
        }

        # Test allow
        result = await tools.can_use_tool("test", {"blocked": False}, context)
        assert result["behavior"] == "allow"

        # Test deny
        result = await tools.can_use_tool("test", {"blocked": True}, context)
        assert result["behavior"] == "deny"


class TestCCToolsExecution:
    """Tests for CCTools tool execution."""

    @pytest.mark.asyncio
    async def test_execute_tool(self):
        """Test tool execution."""
        tools = CCTools()

        @tools.tool("add", "Add numbers", {"type": "object"})
        async def add(x: int, y: int) -> int:
            return x + y

        result = await tools.execute_tool("add", {"x": 2, "y": 3})

        assert result == 5

    @pytest.mark.asyncio
    async def test_execute_tool_records_history(self):
        """Test that tool execution records history."""
        tools = CCTools()

        @tools.tool("test", "Test", {"type": "object"})
        async def test_handler(value: str) -> str:
            return value.upper()

        await tools.execute_tool("test", {"value": "hello"})

        history = tools.get_history()
        assert len(history) == 1
        assert history[0]["tool"] == "test"
        assert history[0]["result"] == "HELLO"
        assert history[0]["success"] is True

    @pytest.mark.asyncio
    async def test_execute_tool_error_handling(self):
        """Test tool execution error handling."""
        tools = CCTools()

        @tools.tool("fail", "Will fail", {"type": "object"})
        async def fail_handler() -> None:
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await tools.execute_tool("fail", {})

        history = tools.get_history()
        assert len(history) == 1
        assert history[0]["success"] is False
        assert "Test error" in history[0]["error"]

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self):
        """Test executing non-existent tool."""
        tools = CCTools()

        with pytest.raises(ValueError, match="Tool not found"):
            await tools.execute_tool("nonexistent", {})

    @pytest.mark.asyncio
    async def test_execute_tool_no_handler(self):
        """Test executing tool without handler."""
        tools = CCTools()
        tools.register_tool("no_handler", "No handler", {"type": "object"})

        with pytest.raises(ValueError, match="No handler"):
            await tools.execute_tool("no_handler", {})

    def test_clear_history(self):
        """Test clearing history."""
        tools = CCTools()
        tools._tool_history.append({"tool": "test"})

        tools.clear_history()

        assert len(tools.get_history()) == 0


class TestCCToolsSanitization:
    """Tests for CCTools input sanitization."""

    def test_coerce_string(self):
        """Test string coercion."""
        tools = CCTools()

        result = tools._coerce_value(123, {"type": "string"})
        assert result == "123"

    def test_coerce_integer(self):
        """Test integer coercion."""
        tools = CCTools()

        result = tools._coerce_value("42", {"type": "integer"})
        assert result == 42

    def test_coerce_number(self):
        """Test number coercion."""
        tools = CCTools()

        result = tools._coerce_value("3.14", {"type": "number"})
        assert result == 3.14

    def test_coerce_boolean(self):
        """Test boolean coercion."""
        tools = CCTools()

        assert tools._coerce_value("true", {"type": "boolean"}) is True
        assert tools._coerce_value("false", {"type": "boolean"}) is False
        assert tools._coerce_value("yes", {"type": "boolean"}) is True

    def test_coerce_array(self):
        """Test array coercion."""
        tools = CCTools()

        result = tools._coerce_value("single", {"type": "array"})
        assert result == ["single"]

    def test_sanitize_input(self):
        """Test input sanitization."""
        tools = CCTools()
        tools.register_tool(
            "test",
            "Test",
            {
                "type": "object",
                "properties": {
                    "count": {"type": "integer"},
                    "name": {"type": "string"},
                },
            },
        )

        result = tools._sanitize_input("test", {"count": "42", "name": 123})

        assert result["count"] == 42
        assert result["name"] == "123"


class TestCCToolsPermissionRules:
    """Tests for permission rules."""

    def test_add_permission_rule(self):
        """Test adding permission rules."""
        tools = CCTools()
        tools.register_tool("write", "Write file", {"type": "object"})

        tools.add_permission_rule("write", {
            "allowed_paths": ["/tmp/*", "/home/user/*"],
        })

        assert "write" in tools._permission_rules

    @pytest.mark.asyncio
    async def test_permission_rule_allowed_path(self):
        """Test permission rule with allowed path."""
        tools = CCTools()
        tools.register_tool("write", "Write", {"type": "object"}, permission_mode="ask")
        tools.add_permission_rule("write", {"allowed_paths": ["/tmp/*"]})

        context: ToolPermissionContext = {
            "session_id": "test",
            "turn_number": 1,
            "tool_history": [],
            "working_directory": "/tmp",
        }

        # Allowed path
        result = await tools.can_use_tool("write", {"path": "/tmp/test.txt"}, context)
        assert result["behavior"] == "allow"

        # Denied path
        result = await tools.can_use_tool("write", {"path": "/etc/passwd"}, context)
        assert result["behavior"] == "deny"

    @pytest.mark.asyncio
    async def test_permission_rule_denied_patterns(self):
        """Test permission rule with denied patterns."""
        tools = CCTools()
        tools.register_tool("write", "Write", {"type": "object"}, permission_mode="ask")
        tools.add_permission_rule("write", {"denied_patterns": ["*.exe", "*.sh"]})

        context: ToolPermissionContext = {
            "session_id": "test",
            "turn_number": 1,
            "tool_history": [],
            "working_directory": "/tmp",
        }

        # Denied pattern
        result = await tools.can_use_tool("write", {"path": "malware.exe"}, context)
        assert result["behavior"] == "deny"


class TestCCToolsConversion:
    """Tests for tool conversion methods."""

    def test_to_pydantic_tools(self):
        """Test conversion to pydantic_ai tool format."""
        tools = CCTools()
        tools.register_tool(
            "search",
            "Search for info",
            {"type": "object", "properties": {"query": {"type": "string"}}},
        )
        tools.register_tool(
            "calculate",
            "Do math",
            {"type": "object", "properties": {"expr": {"type": "string"}}},
        )

        pydantic_tools = tools.to_pydantic_tools()

        assert len(pydantic_tools) == 2
        assert pydantic_tools[0]["name"] in ["search", "calculate"]
        assert "parameters_json_schema" in pydantic_tools[0]


class TestCCToolsImports:
    """Tests for package imports."""

    def test_import_cctools(self):
        """Test importing CCTools from package."""
        from pydantic_ai_claude_code import CCTools

        tools = CCTools()
        assert tools is not None

    def test_import_tool_definition(self):
        """Test importing ToolDefinition from package."""
        from pydantic_ai_claude_code import ToolDefinition

        tool = ToolDefinition(
            name="test",
            description="Test",
            parameters_schema={"type": "object"},
        )
        assert tool is not None
