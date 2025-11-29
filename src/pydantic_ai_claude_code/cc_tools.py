"""
CCTools - Claude Code Tools Manager.

Advanced tool orchestration following pydantic_ai patterns while
leveraging SDK capabilities.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from .sdk_original_files.types import (
    CanUseTool,
    PermissionResultAllow,
    PermissionResultDeny,
    ToolPermissionContext,
)

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """Tool definition following pydantic_ai patterns."""

    name: str
    description: str
    parameters_schema: Dict[str, Any]
    handler: Optional[Callable[..., Awaitable[Any]]] = None
    permission_mode: str = "ask"  # ask, allow, deny

    def to_sdk_format(self) -> Dict[str, Any]:
        """Convert to SDK tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters_schema,
        }


class CCTools:
    """
    Claude Code Tools manager - advanced tool orchestration.

    Provides:
    - Tool registration and management
    - Permission callbacks for fine-grained control
    - Input sanitization and validation
    - Execution history tracking
    - Integration with pydantic_ai tool patterns

    Example:
        ```python
        tools = CCTools()

        # Register a tool
        tools.register_tool(
            name="search",
            description="Search for information",
            parameters_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            },
            permission_mode="allow"
        )

        # Use decorator syntax
        @tools.tool("calculate", "Perform calculation", {"type": "object"})
        async def calculate(expression: str):
            return eval(expression)

        # Create provider with tools
        provider = ClaudeCodeProvider(
            allowed_tools=tools.get_allowed_tools(),
            disallowed_tools=tools.get_disallowed_tools(),
        )
        ```
    """

    def __init__(self) -> None:
        """Initialize the tools manager."""
        self._tools: Dict[str, ToolDefinition] = {}
        self._permission_callback: Optional[CanUseTool] = None
        self._tool_history: List[Dict[str, Any]] = []
        self._permission_rules: Dict[str, Dict[str, Any]] = {}

        logger.debug("Initialized CCTools manager")

    def register_tool(
        self,
        name: str,
        description: str,
        parameters_schema: Dict[str, Any],
        handler: Optional[Callable[..., Awaitable[Any]]] = None,
        permission_mode: str = "ask",
    ) -> ToolDefinition:
        """
        Register a tool with the system.

        Args:
            name: Tool name (must be unique)
            description: Human-readable description
            parameters_schema: JSON Schema for parameters
            handler: Optional async function to handle tool calls
            permission_mode: "ask", "allow", or "deny"

        Returns:
            ToolDefinition instance

        Example:
            ```python
            tools.register_tool(
                name="read_file",
                description="Read contents of a file",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"}
                    },
                    "required": ["path"]
                },
                permission_mode="allow"
            )
            ```
        """
        tool = ToolDefinition(
            name=name,
            description=description,
            parameters_schema=parameters_schema,
            handler=handler,
            permission_mode=permission_mode,
        )
        self._tools[name] = tool

        logger.debug(
            "Registered tool '%s' with permission_mode='%s'",
            name,
            permission_mode,
        )
        return tool

    def tool(
        self,
        name: str,
        description: str,
        schema: Dict[str, Any],
        permission_mode: str = "ask",
    ) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
        """
        Decorator for tool registration (pydantic_ai style).

        Args:
            name: Tool name
            description: Tool description
            schema: Parameter schema
            permission_mode: Permission mode

        Returns:
            Decorator function

        Example:
            ```python
            @tools.tool(
                "search",
                "Search the web",
                {"type": "object", "properties": {"query": {"type": "string"}}}
            )
            async def search_handler(query: str):
                return f"Results for: {query}"
            ```
        """

        def decorator(
            func: Callable[..., Awaitable[Any]]
        ) -> Callable[..., Awaitable[Any]]:
            self.register_tool(
                name=name,
                description=description,
                parameters_schema=schema,
                handler=func,
                permission_mode=permission_mode,
            )
            return func

        return decorator

    def set_permission_callback(self, callback: CanUseTool) -> None:
        """
        Set custom permission callback.

        Args:
            callback: Async function for permission decisions

        Example:
            ```python
            async def my_callback(tool_name, tool_input, context):
                if tool_name == "dangerous_tool":
                    return {"behavior": "deny", "message": "Not allowed"}
                return {"behavior": "allow", "updated_input": None}

            tools.set_permission_callback(my_callback)
            ```
        """
        self._permission_callback = callback
        logger.debug("Set custom permission callback")

    def add_permission_rule(
        self,
        tool_name: str,
        rule: Dict[str, Any],
    ) -> None:
        """
        Add a permission rule for a specific tool.

        Args:
            tool_name: Tool to apply rule to
            rule: Rule configuration

        Example:
            ```python
            tools.add_permission_rule("write_file", {
                "allowed_paths": ["/tmp/*", "/home/user/projects/*"],
                "denied_patterns": ["*.exe", "*.sh"],
                "max_size_bytes": 1024 * 1024,
            })
            ```
        """
        self._permission_rules[tool_name] = rule
        logger.debug("Added permission rule for tool '%s'", tool_name)

    async def can_use_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        context: ToolPermissionContext,
    ) -> Union[PermissionResultAllow, PermissionResultDeny]:
        """
        Permission callback implementation.

        Args:
            tool_name: Name of the tool being called
            tool_input: Input arguments for the tool
            context: Context about the current session

        Returns:
            Permission result (allow or deny)
        """
        tool = self._tools.get(tool_name)

        if not tool:
            logger.warning("Permission check for unknown tool: %s", tool_name)
            return PermissionResultDeny(
                behavior="deny",
                message=f"Unknown tool: {tool_name}",
            )

        # Check tool-specific permission mode
        if tool.permission_mode == "allow":
            sanitized = self._sanitize_input(tool_name, tool_input)
            logger.debug("Auto-allowing tool '%s'", tool_name)
            return PermissionResultAllow(
                behavior="allow",
                updated_input=sanitized if sanitized != tool_input else None,
            )
        elif tool.permission_mode == "deny":
            logger.debug("Auto-denying tool '%s'", tool_name)
            return PermissionResultDeny(
                behavior="deny",
                message=f"Tool {tool_name} is disabled",
            )

        # Apply permission rules if defined
        if tool_name in self._permission_rules:
            result = self._apply_permission_rules(
                tool_name, tool_input, self._permission_rules[tool_name]
            )
            if result:
                return result

        # Use custom callback if provided
        if self._permission_callback:
            result = await self._permission_callback(tool_name, tool_input, context)
            return result

        # Default: allow
        logger.debug("Default allowing tool '%s'", tool_name)
        return PermissionResultAllow(behavior="allow", updated_input=None)

    def _apply_permission_rules(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        rules: Dict[str, Any],
    ) -> Optional[Union[PermissionResultAllow, PermissionResultDeny]]:
        """Apply permission rules to tool input."""
        import fnmatch

        # Check allowed paths
        if "allowed_paths" in rules and "path" in tool_input:
            path = tool_input["path"]
            allowed = any(
                fnmatch.fnmatch(path, pattern) for pattern in rules["allowed_paths"]
            )
            if not allowed:
                return PermissionResultDeny(
                    behavior="deny",
                    message=f"Path not in allowed list: {path}",
                )

        # Check denied patterns
        if "denied_patterns" in rules:
            for key, value in tool_input.items():
                if isinstance(value, str):
                    for pattern in rules["denied_patterns"]:
                        if fnmatch.fnmatch(value, pattern):
                            return PermissionResultDeny(
                                behavior="deny",
                                message=f"Input matches denied pattern: {pattern}",
                            )

        return None

    def _sanitize_input(
        self, tool_name: str, tool_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Sanitize tool input based on schema."""
        tool = self._tools.get(tool_name)
        if not tool:
            return tool_input

        schema = tool.parameters_schema
        sanitized: Dict[str, Any] = {}

        # Apply schema validation/sanitization
        properties = schema.get("properties", {})
        for param_name, param_schema in properties.items():
            if param_name in tool_input:
                value = tool_input[param_name]
                sanitized[param_name] = self._coerce_value(value, param_schema)

        # Keep any extra fields not in schema
        for key, value in tool_input.items():
            if key not in sanitized:
                sanitized[key] = value

        return sanitized

    def _coerce_value(self, value: Any, schema: Dict[str, Any]) -> Any:
        """Coerce value to match schema type."""
        param_type = schema.get("type")

        if param_type == "string" and not isinstance(value, str):
            return str(value)
        elif param_type == "integer" and not isinstance(value, int):
            try:
                return int(value)
            except (ValueError, TypeError):
                return value
        elif param_type == "number" and not isinstance(value, (int, float)):
            try:
                return float(value)
            except (ValueError, TypeError):
                return value
        elif param_type == "boolean" and not isinstance(value, bool):
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes")
            return bool(value)
        elif param_type == "array" and not isinstance(value, list):
            return [value]

        return value

    def get_allowed_tools(self) -> List[str]:
        """Get list of allowed tool names."""
        return [
            name
            for name, tool in self._tools.items()
            if tool.permission_mode != "deny"
        ]

    def get_disallowed_tools(self) -> List[str]:
        """Get list of disallowed tool names."""
        return [
            name
            for name, tool in self._tools.items()
            if tool.permission_mode == "deny"
        ]

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    async def execute_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
    ) -> Any:
        """
        Execute a tool if it has a handler.

        Args:
            tool_name: Name of the tool to execute
            args: Arguments to pass to the handler

        Returns:
            Result from the tool handler

        Raises:
            ValueError: If tool not found or no handler
        """
        tool = self._tools.get(tool_name)

        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")

        if not tool.handler:
            raise ValueError(f"No handler for tool: {tool_name}")

        # Record in history
        history_entry: Dict[str, Any] = {
            "tool": tool_name,
            "args": args,
            "timestamp": datetime.now().isoformat(),
        }
        self._tool_history.append(history_entry)

        try:
            # Execute handler
            result = await tool.handler(**args)

            # Record result
            history_entry["result"] = result
            history_entry["success"] = True

            logger.debug("Executed tool '%s' successfully", tool_name)
            return result

        except Exception as e:
            # Record error
            history_entry["error"] = str(e)
            history_entry["success"] = False

            logger.error("Tool '%s' execution failed: %s", tool_name, e)
            raise

    def get_history(self) -> List[Dict[str, Any]]:
        """Get tool execution history."""
        return self._tool_history.copy()

    def clear_history(self) -> None:
        """Clear tool execution history."""
        self._tool_history.clear()
        logger.debug("Cleared tool history")

    def to_pydantic_tools(self) -> List[Dict[str, Any]]:
        """
        Convert tools to pydantic_ai tool format.

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
        ]
