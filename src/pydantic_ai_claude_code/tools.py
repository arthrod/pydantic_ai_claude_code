"""Tool calling utilities for Claude Code model."""

import json
import logging
import re
import uuid
from typing import Any

from pydantic_ai import ToolDefinition
from pydantic_ai.messages import ToolCallPart

logger = logging.getLogger(__name__)


def format_tools_for_prompt(tools: list[ToolDefinition]) -> str:
    """Format tool definitions for inclusion in system prompt.

    Args:
        tools: List of tool definitions

    Returns:
        Formatted string describing tools for Claude
    """
    if not tools:
        logger.debug("No tools to format for prompt")
        return ""

    logger.debug("Formatting %d tools for prompt", len(tools))

    # Build simple natural descriptions
    func_descriptions = []
    for tool in tools:
        logger.debug("Adding function to prompt: %s", tool.name)
        params = tool.parameters_json_schema.get("properties", {})
        param_list = ", ".join(params.keys())
        func_descriptions.append(
            f"- {tool.name}({param_list}): {tool.description or 'No description'}"
        )

    tools_prompt = f"""
IMPORTANT PROTOCOL - Read carefully:

If you need data from external functions:
1. First turn: Output ONLY "EXECUTE: function_name(params)" - nothing else
2. You'll receive "Tool Result (function_name): <data>"
3. Second turn: Answer the user's question using that data - DO NOT call EXECUTE again

Example:
User: What's 5 + 3?
You: EXECUTE: add(a=5, b=3)
Tool Result (add): 8
You: The answer is 8.

Available functions:
{chr(10).join(func_descriptions)}

CRITICAL: If you see "Tool Result" in the conversation, that means you already called the function. Use that result to answer - DO NOT output EXECUTE again!
"""
    return tools_prompt


def _parse_value_from_string(value_str: str, param_name: str) -> Any:
    """Parse a value string into appropriate Python type.

    Args:
        value_str: String representation of value
        param_name: Parameter name (for logging)

    Returns:
        Parsed value
    """
    try:
        lower_val = value_str.lower()

        # Handle booleans
        if lower_val in ("true", "false"):
            return lower_val == "true"

        # Handle lists
        if value_str.startswith("["):
            return json.loads(value_str)

        # Handle numbers
        if "." in value_str and value_str.replace(".", "").replace("-", "").isdigit():
            return float(value_str)

        if value_str.isdigit() or (value_str.startswith("-") and value_str[1:].isdigit()):
            return int(value_str)

    except (ValueError, json.JSONDecodeError) as e:
        logger.warning(
            "Failed to parse value '%s' for param '%s': %s",
            value_str,
            param_name,
            e,
        )

    # Default: return as string
    return value_str


def _extract_value_from_args(args_str: str, index: int) -> tuple[str, int]:
    """Extract a value from argument string starting at index.

    Args:
        args_str: Full arguments string
        index: Starting index

    Returns:
        Tuple of (value_string, new_index)
    """
    i = index

    # Handle lists
    if args_str[i] == "[":
        bracket_count = 0
        start = i
        while i < len(args_str):
            if args_str[i] == "[":
                bracket_count += 1
            elif args_str[i] == "]":
                bracket_count -= 1
                if bracket_count == 0:
                    i += 1
                    break
            i += 1
        return args_str[start:i], i

    # Handle quoted strings
    if args_str[i] in ('"', "'"):
        quote = args_str[i]
        i += 1
        start = i
        while i < len(args_str) and args_str[i] != quote:
            i += 1
        value_str = args_str[start:i]
        if i < len(args_str):
            i += 1  # Skip closing quote
        return value_str, i

    # Handle unquoted values (numbers, booleans, etc.)
    start = i
    while i < len(args_str) and args_str[i] not in (",", ")"):
        i += 1
    return args_str[start:i].strip(), i


def _parse_execute_arguments(args_str: str) -> dict[str, Any]:
    """Parse arguments from EXECUTE format string.

    Args:
        args_str: Arguments string (e.g., "a=1, b='hello'")

    Returns:
        Dictionary of parsed arguments
    """
    args: dict[str, Any] = {}
    i = 0

    while i < len(args_str):
        # Skip whitespace
        while i < len(args_str) and args_str[i].isspace():
            i += 1
        if i >= len(args_str):
            break

        # Find parameter name
        name_match = re.match(r"(\w+)\s*=\s*", args_str[i:])
        if not name_match:
            break

        param_name = name_match.group(1)
        i += name_match.end()

        # Extract parameter value
        if i < len(args_str):
            value_str, i = _extract_value_from_args(args_str, i)
            args[param_name] = _parse_value_from_string(value_str, param_name)

        # Skip comma and spaces
        while i < len(args_str) and args_str[i] in (",", " "):
            i += 1

    return args


def _parse_execute_format(response_text: str) -> list[ToolCallPart] | None:
    """Parse EXECUTE format tool calls.

    Args:
        response_text: Response text

    Returns:
        List of ToolCallPart or None if not found
    """
    execute_pattern = r"EXECUTE:\s*(\w+)\((.*?)\)"
    match = re.search(execute_pattern, response_text, re.DOTALL)

    if not match:
        return None

    tool_name = match.group(1)
    args_str = match.group(2).strip()

    args = _parse_execute_arguments(args_str) if args_str else {}

    logger.debug("Parsed EXECUTE format: %s with %d args", tool_name, len(args))
    return [
        ToolCallPart(
            tool_name=tool_name,
            args=args,
            tool_call_id=f"call_{uuid.uuid4().hex[:16]}",
        )
    ]


def _parse_legacy_json_format(response_text: str) -> list[ToolCallPart] | None:
    """Parse legacy JSON format tool calls.

    Args:
        response_text: Response text

    Returns:
        List of ToolCallPart or None if not found
    """
    cleaned = response_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and data.get("type") == "tool_calls":
            calls = data.get("calls", [])
            if isinstance(calls, list) and calls:
                logger.debug("Parsed legacy JSON format")
                return _convert_to_tool_call_parts(calls)
    except json.JSONDecodeError:
        pass

    return None


def parse_tool_calls(response_text: str) -> list[ToolCallPart] | None:
    """Parse tool calls from Claude's response.

    Supports both new EXECUTE format and legacy JSON format.

    Args:
        response_text: The response text from Claude

    Returns:
        List of ToolCallPart objects if tool calls detected, None otherwise
    """
    logger.debug("Parsing tool calls from response (%d chars)", len(response_text))

    # Try EXECUTE format first
    result = _parse_execute_format(response_text)
    if result:
        return result

    # Try legacy JSON format
    result = _parse_legacy_json_format(response_text)
    if result:
        return result

    logger.debug("No tool calls found in response")
    return None


def _convert_to_tool_call_parts(calls: list[Any]) -> list[ToolCallPart] | None:
    """Convert call dicts to ToolCallPart objects.

    Args:
        calls: List of call dictionaries

    Returns:
        List of ToolCallPart objects or None if invalid
    """
    tool_call_parts = []
    for call in calls:
        if not isinstance(call, dict):
            logger.warning("Skipping non-dict tool call: %s", type(call))
            continue

        tool_name = call.get("tool_name")
        args = call.get("args", {})

        if not tool_name:
            logger.warning("Skipping tool call without tool_name")
            continue

        logger.debug(
            "Converting tool call: %s with %d args",
            tool_name,
            len(args) if isinstance(args, dict) else 0,
        )

        tool_call_parts.append(
            ToolCallPart(
                tool_name=tool_name,
                args=args,
                tool_call_id=f"call_{uuid.uuid4().hex[:16]}",
            )
        )

    logger.debug(
        "Converted %d tool calls to ToolCallPart objects", len(tool_call_parts)
    )
    return tool_call_parts if tool_call_parts else None


def is_tool_call_response(response_text: str) -> bool:
    """Check if response contains tool calls.

    Args:
        response_text: The response text from Claude

    Returns:
        True if response contains tool calls
    """
    return parse_tool_calls(response_text) is not None
