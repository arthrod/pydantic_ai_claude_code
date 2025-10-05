"""Tool calling utilities for Claude Code model."""

import json
import logging
import uuid

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

    tool_descriptions = []
    for tool in tools:
        logger.debug("Adding tool to prompt: %s", tool.name)
        schema_str = json.dumps(tool.parameters_json_schema, indent=2)
        desc = f"""
## Tool: {tool.name}

Description: {tool.description or "No description provided"}

Parameters Schema:
```json
{schema_str}
```
"""
        tool_descriptions.append(desc)

    tools_prompt = f"""
# Available Tools

You have access to the following tools. To use a tool, respond with a JSON object in this EXACT format:

```json
{{
  "type": "tool_calls",
  "calls": [
    {{"tool_name": "tool_name_here", "args": {{"param1": "value1", "param2": "value2"}}}}
  ]
}}
```

You can call multiple tools in one response by adding more objects to the "calls" array.

IMPORTANT:
- Only call tools when necessary to answer the user's question
- If you can answer directly, provide a normal text response instead
- When calling tools, ONLY return the JSON - no explanatory text before or after
- Tool names must match exactly
- Args must conform to the parameter schema

Available Tools:
{"".join(tool_descriptions)}
"""
    return tools_prompt


def parse_tool_calls(response_text: str) -> list[ToolCallPart] | None:
    """Parse tool calls from Claude's response.

    Args:
        response_text: The response text from Claude

    Returns:
        List of ToolCallPart objects if tool calls detected, None otherwise
    """
    import re

    logger.debug("Parsing tool calls from response (%d chars)", len(response_text))

    # Strategy 1: Try parsing the whole response directly (handle simple cases)
    cleaned = response_text.strip()

    # Remove markdown code blocks if present at start/end
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # Try to parse as JSON directly
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and data.get("type") == "tool_calls":
            calls = data.get("calls", [])
            if isinstance(calls, list) and calls:
                logger.debug(
                    "Successfully parsed tool calls using strategy 1 (direct parse)"
                )
                return _convert_to_tool_call_parts(calls)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract JSON from markdown code blocks anywhere in text
    # Match ```json ... ``` or ``` ... ```
    code_block_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    matches = re.findall(code_block_pattern, response_text, re.DOTALL)

    for match in matches:
        try:
            data = json.loads(match)
            if isinstance(data, dict) and data.get("type") == "tool_calls":
                calls = data.get("calls", [])
                if isinstance(calls, list) and calls:
                    logger.debug(
                        "Successfully parsed tool calls using strategy 2 (code block extraction)"
                    )
                    return _convert_to_tool_call_parts(calls)
        except json.JSONDecodeError:
            continue

    # Strategy 3: Extract JSON objects from anywhere in text using regex
    # Match { ... } objects (handles nested braces)
    json_pattern = r"\{(?:[^{}]|\{[^{}]*\})*\}"
    matches = re.findall(json_pattern, response_text, re.DOTALL)

    for match in matches:
        try:
            data = json.loads(match)
            if isinstance(data, dict) and data.get("type") == "tool_calls":
                calls = data.get("calls", [])
                if isinstance(calls, list) and calls:
                    logger.debug(
                        "Successfully parsed tool calls using strategy 3 (regex extraction)"
                    )
                    return _convert_to_tool_call_parts(calls)
        except json.JSONDecodeError:
            continue

    logger.debug("No tool calls found in response")
    return None


def _convert_to_tool_call_parts(calls: list) -> list[ToolCallPart] | None:
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
