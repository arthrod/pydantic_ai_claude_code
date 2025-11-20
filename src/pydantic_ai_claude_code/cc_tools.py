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
        """
        Convert the tool into an SDK-compatible dictionary describing its public metadata.
        
        Returns:
            dict: A dictionary with keys:
                - "name": the tool's name (str)
                - "description": the tool's description (str)
                - "input_schema": the tool's parameters schema (dict)
        """
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
        """
        Initialize the CCTools manager and its internal state.
        
        Initializes:
        - _tools: mapping of registered tool names to ToolDefinition instances.
        - _permission_callback: optional async callback for permission decisions (unset).
        - _tool_history: chronological list of executed tool records (empty).
        """
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
        Register a tool and store its definition for later use.
        
        Parameters:
            name (str): Unique tool identifier.
            description (str): Human-readable description of the tool.
            parameters_schema (dict[str, Any]): JSON schema describing the tool's input parameters.
            handler (Optional[Callable]): Callable invoked when the tool is executed (may be async).
            permission_mode (str): Permission behavior for the tool; must be one of "ask", "allow", or "deny".
        
        Returns:
            ToolDefinition: The created and registered tool definition.
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
        Decorator to register a function as a tool with metadata and a permission mode.
        
        Parameters:
            name (str): Tool identifier used for registration and lookup.
            description (str): Human-readable summary of the tool's purpose.
            schema (dict[str, Any]): JSON schema describing the tool's input parameters.
            permission_mode (str): Permission behavior for the tool; typically "ask", "allow", or "deny".
        
        Returns:
            A decorator that registers the decorated function as the tool and returns the original function.
        """
        def decorator(func: Callable) -> Callable:
            """
            Register the decorated callable as a tool using the surrounding decorator factory and return it unchanged.
            
            Parameters:
                func (Callable): The function to register as a tool.
            
            Returns:
                Callable: The original `func`, unmodified.
            """
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
        Determine whether the specified tool may be used with the given input and context.
        
        Parameters:
            tool_name (str): The registered tool's name to check.
            tool_input (dict[str, Any]): Candidate input for the tool; may be sanitized or modified when allowed.
            context (Optional[dict[str, Any]]): Optional additional context to inform permission decisions.
        
        Returns:
            PermissionResult: Decision with `behavior` set to `"allow"` or `"deny"`. When allowed, `updated_input` may contain a sanitized/normalized input; when denied, `message` may explain the denial.
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
        Register a custom async permission callback used to decide tool access.
        
        Parameters:
            callback (Callable[[str, dict, Optional[dict]], Awaitable[PermissionResult]]):
                Async function invoked with (tool_name, tool_input, context) that must
                return a PermissionResult indicating whether the tool use is allowed,
                optionally providing a message and/or updated_input. The callback will
                be used by can_use_tool when evaluating permission decisions.
        """
        self._permission_callback = callback

    def _sanitize_input(
        self,
        tool_name: str,
        tool_input: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Sanitize and coerce a tool's input dictionary according to the tool's parameters schema.
        
        Parameters:
            tool_name (str): Name of the registered tool whose schema will be used.
            tool_input (dict[str, Any]): Raw input mapping of parameter names to values.
        
        Returns:
            dict[str, Any]: A sanitized input dict containing only keys defined in the tool's schema "properties",
            with each value coerced via the tool manager's coercion rules. If the tool is not registered, returns
            the original `tool_input` unchanged.
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
        Coerce a value to the type indicated by a JSON-schema-like `schema`.
        
        Parameters:
            value (Any): The input value to coerce.
            schema (dict[str, Any]): Schema dictionary expected to contain a `"type"` key
                with one of: `"string"`, `"integer"`, `"number"`, `"boolean"`, or `"array"`.
        
        Returns:
            Any: The value converted to the schema's type when conversion is possible;
            otherwise the original `value`.
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
        Retrieve the registered tool with the given name.
        
        Returns:
            The ToolDefinition if found, otherwise None.
        """
        return self._tools.get(name)

    def get_all_tools(self) -> list[ToolDefinition]:
        """
        Retrieve all registered tools.
        
        Returns:
            A list of ToolDefinition objects representing the tools currently registered with this manager.
        """
        return list(self._tools.values())

    def get_allowed_tools(self) -> list[str]:
        """
        List names of registered tools whose permission_mode is not "deny".
        
        Returns:
            allowed_tools (list[str]): Names of tools that are allowed (permission_mode != "deny").
        """
        return [
            name for name, tool in self._tools.items()
            if tool.permission_mode != "deny"
        ]

    def get_disallowed_tools(self) -> list[str]:
        """
        List tool names configured with permission mode "deny".
        
        Returns:
            list[str]: Tool names whose `permission_mode` is `"deny"`.
        """
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
        Execute a registered tool's handler with the given keyword arguments and record the execution in history.
        
        Parameters:
            tool_name (str): The name of the registered tool to execute.
            args (dict[str, Any]): Mapping of keyword arguments to pass to the tool's handler.
        
        Returns:
            The value returned by the tool's handler.
        
        Raises:
            ValueError: If the tool is unknown or has no handler.
            Exception: Any exception raised by the handler is propagated.
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
        Return a shallow copy of the tool execution history.
        
        The returned list contains history entries in insertion order; modifying the returned list will not affect the manager's internal history.
        
        Returns:
            history (list[dict[str, Any]]): A list of history entry dictionaries (shallow-copied).
        """
        return self._tool_history.copy()

    def clear_history(self) -> None:
        """Clear tool execution history."""
        self._tool_history = []

    def to_pydantic_ai_tools(self) -> list[dict[str, Any]]:
        """
        Return a list of registered tools formatted for pydantic_ai, excluding tools with permission_mode equal to "deny".
        
        Each item is a dictionary containing `name`, `description`, and `parameters_json_schema` derived from the tool's parameters_schema.
        
        Returns:
            list[dict[str, Any]]: List of tool definitions ready for pydantic_ai consumption.
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