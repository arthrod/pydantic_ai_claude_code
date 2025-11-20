"""
CCTools - Claude Code Tools manager for advanced tool orchestration.

This module provides a tool management system that follows pydantic_ai patterns
while leveraging SDK capabilities for permission management.
"""

from typing import Any, Callable, Awaitable, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """Tool definition following pydantic_ai patterns."""

    name: str
    description: str
    parameters_schema: dict[str, Any]
    handler: Optional[Callable[..., Awaitable[Any]]] = None
    permission_mode: str = "ask"  # ask, allow, deny

    def to_sdk_format(self) -> dict[str, Any]:
        """Convert to SDK tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters_schema,
        }


@dataclass
class PermissionResult:
    """Result of a permission check."""

    behavior: str  # "allow" or "deny"
    message: Optional[str] = None
    updated_input: Optional[dict[str, Any]] = None


class CCTools:
    """
    Claude Code Tools manager - advanced tool orchestration.

    Follows pydantic_ai patterns while leveraging SDK capabilities.

    Example:
        ```python
        tools = CCTools()

        # Register a tool with decorator
        @tools.tool(
            name="calculate",
            description="Perform calculation",
            schema={"type": "object", "properties": {"expr": {"type": "string"}}}
        )
        async def calculate(expr: str) -> str:
            return str(eval(expr))

        # Or register directly
        tools.register_tool(
            name="search",
            description="Search the web",
            parameters_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            permission_mode="allow"
        )
        ```
    """

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self._permission_callback: Optional[Callable] = None
        self._tool_history: list[dict[str, Any]] = []

    def register_tool(
        self,
        name: str,
        description: str,
        parameters_schema: dict[str, Any],
        handler: Optional[Callable] = None,
        permission_mode: str = "ask",
    ) -> ToolDefinition:
        """
        Register a tool with the system.

        Args:
            name: Tool name
            description: Tool description
            parameters_schema: JSON schema for parameters
            handler: Optional async function to handle tool calls
            permission_mode: Permission mode (ask, allow, deny)

        Returns:
            ToolDefinition: The registered tool
        """
        tool = ToolDefinition(
            name=name,
            description=description,
            parameters_schema=parameters_schema,
            handler=handler,
            permission_mode=permission_mode,
        )
        self._tools[name] = tool
        logger.debug("Registered tool: %s (mode: %s)", name, permission_mode)
        return tool

    def tool(
        self,
        name: str,
        description: str,
        schema: dict[str, Any],
        permission_mode: str = "ask",
    ):
        """
        Decorator for tool registration (pydantic_ai style).

        Args:
            name: Tool name
            description: Tool description
            schema: JSON schema for parameters
            permission_mode: Permission mode

        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            self.register_tool(
                name=name,
                description=description,
                parameters_schema=schema,
                handler=func,
                permission_mode=permission_mode,
            )
            return func
        return decorator

    async def can_use_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        context: Optional[dict[str, Any]] = None,
    ) -> PermissionResult:
        """
        Permission callback implementation.

        Args:
            tool_name: Name of the tool
            tool_input: Input arguments
            context: Optional context information

        Returns:
            PermissionResult with allow/deny decision
        """
        tool = self._tools.get(tool_name)

        if not tool:
            logger.warning("Permission check for unknown tool: %s", tool_name)
            return PermissionResult(
                behavior="deny",
                message=f"Unknown tool: {tool_name}",
            )

        # Check tool-specific permission mode
        if tool.permission_mode == "allow":
            # Auto-allow with optional input modification
            sanitized = self._sanitize_input(tool_name, tool_input)
            logger.debug("Auto-allowing tool: %s", tool_name)
            return PermissionResult(
                behavior="allow",
                updated_input=sanitized,
            )
        elif tool.permission_mode == "deny":
            logger.debug("Auto-denying tool: %s", tool_name)
            return PermissionResult(
                behavior="deny",
                message=f"Tool {tool_name} is disabled",
            )

        # Default to custom callback if provided
        if self._permission_callback:
            return await self._permission_callback(tool_name, tool_input, context)

        # Default allow for "ask" mode without callback
        return PermissionResult(behavior="allow")

    def set_permission_callback(
        self,
        callback: Callable[[str, dict, Optional[dict]], Awaitable[PermissionResult]]
    ) -> None:
        """
        Set custom permission callback.

        Args:
            callback: Async function that returns PermissionResult
        """
        self._permission_callback = callback

    def _sanitize_input(
        self,
        tool_name: str,
        tool_input: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Sanitize tool input based on schema.

        Args:
            tool_name: Tool name
            tool_input: Input to sanitize

        Returns:
            Sanitized input dict
        """
        tool = self._tools.get(tool_name)
        if not tool:
            return tool_input

        schema = tool.parameters_schema
        sanitized = {}

        # Apply schema validation/sanitization
        properties = schema.get("properties", {})
        for param_name, param_schema in properties.items():
            if param_name in tool_input:
                value = tool_input[param_name]
                # Apply type coercion/validation
                sanitized[param_name] = self._coerce_value(value, param_schema)

        return sanitized

    def _coerce_value(self, value: Any, schema: dict[str, Any]) -> Any:
        """
        Coerce value to match schema type.

        Args:
            value: Value to coerce
            schema: JSON schema for the value

        Returns:
            Coerced value
        """
        param_type = schema.get("type")

        try:
            if param_type == "string":
                return str(value)
            elif param_type == "integer":
                return int(value)
            elif param_type == "number":
                return float(value)
            elif param_type == "boolean":
                if isinstance(value, str):
                    return value.lower() in ("true", "1", "yes")
                return bool(value)
            elif param_type == "array" and not isinstance(value, list):
                return [value]
        except (ValueError, TypeError):
            pass

        return value

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """
        Get a tool by name.

        Args:
            name: Tool name

        Returns:
            ToolDefinition or None
        """
        return self._tools.get(name)

    def get_all_tools(self) -> list[ToolDefinition]:
        """Get all registered tools."""
        return list(self._tools.values())

    def get_allowed_tools(self) -> list[str]:
        """Get list of allowed tool names."""
        return [
            name for name, tool in self._tools.items()
            if tool.permission_mode != "deny"
        ]

    def get_disallowed_tools(self) -> list[str]:
        """Get list of disallowed tool names."""
        return [
            name for name, tool in self._tools.items()
            if tool.permission_mode == "deny"
        ]

    async def execute_tool(
        self,
        tool_name: str,
        args: dict[str, Any]
    ) -> Any:
        """
        Execute a tool if it has a handler.

        Args:
            tool_name: Name of the tool
            args: Arguments to pass

        Returns:
            Result from the tool handler

        Raises:
            ValueError: If tool has no handler
        """
        tool = self._tools.get(tool_name)

        if not tool:
            raise ValueError(f"Unknown tool: {tool_name}")

        if not tool.handler:
            raise ValueError(f"No handler for tool: {tool_name}")

        # Record in history
        history_entry = {
            "tool": tool_name,
            "args": args,
            "timestamp": datetime.now().isoformat(),
        }
        self._tool_history.append(history_entry)

        logger.debug("Executing tool: %s with args: %s", tool_name, args)

        # Execute handler
        try:
            result = await tool.handler(**args)
            history_entry["result"] = result
            history_entry["success"] = True
            return result
        except Exception as e:
            history_entry["error"] = str(e)
            history_entry["success"] = False
            raise

    def get_history(self) -> list[dict[str, Any]]:
        """
        Get tool execution history.

        Returns:
            List of history entries
        """
        return self._tool_history.copy()

    def clear_history(self) -> None:
        """Clear tool execution history."""
        self._tool_history = []

    def to_pydantic_ai_tools(self) -> list[dict[str, Any]]:
        """
        Convert tools to pydantic_ai format.

        Returns:
            List of tool definitions in pydantic_ai format
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters_json_schema": tool.parameters_schema,
            }
            for tool in self._tools.values()
            if tool.permission_mode != "deny"
        ]
