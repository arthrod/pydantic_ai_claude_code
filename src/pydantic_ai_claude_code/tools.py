"""Tool calling utilities for Claude Code model."""

import json
import uuid
from typing import Any

from pydantic_ai import ToolDefinition
from pydantic_ai.messages import ToolCallPart


def format_tools_for_prompt(tools: list[ToolDefinition]) -> str:
    """Format tool definitions for inclusion in system prompt.

    Args:
        tools: List of tool definitions

    Returns:
        Formatted string describing tools for Claude
    """
    if not tools:
        return ""

    tool_descriptions = []
    for tool in tools:
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
    # Clean up response text
    cleaned = response_text.strip()

    # Remove markdown code blocks if present
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # Try to parse as JSON
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    # Check if it's a tool calls response
    if not isinstance(data, dict) or data.get("type") != "tool_calls":
        return None

    calls = data.get("calls", [])
    if not isinstance(calls, list) or not calls:
        return None

    # Convert to ToolCallPart objects
    tool_call_parts = []
    for call in calls:
        if not isinstance(call, dict):
            continue

        tool_name = call.get("tool_name")
        args = call.get("args", {})

        if not tool_name:
            continue

        tool_call_parts.append(
            ToolCallPart(
                tool_name=tool_name,
                args=args,
                tool_call_id=f"call_{uuid.uuid4().hex[:16]}",
            )
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
