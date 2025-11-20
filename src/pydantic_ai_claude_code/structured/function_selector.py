"""Function selection for Claude Code model.

This module provides the two-phase function selection protocol that
the Claude Agent SDK doesn't have. It allows Claude to first select
which function to call, then collect the arguments for that function.
"""

import logging
import re
from typing import Any

from ..structure_converter import build_structure_instructions

logger = logging.getLogger(__name__)


def build_function_selection_prompt(
    function_tools: list[dict[str, Any]]
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Build prompt for function selection phase.

    Creates a prompt that presents available functions to Claude and
    asks it to select one or indicate "none" for direct response.

    Args:
        function_tools: List of function tool definitions

    Returns:
        Tuple of (prompt_string, available_functions_dict)
    """
    available_functions: dict[str, dict[str, Any]] = {}

    lines = [
        "# Function Selection",
        "",
        "Based on the user's request, select which function to call:",
        "",
    ]

    for idx, tool in enumerate(function_tools, 1):
        name = tool.get("name", f"function_{idx}")
        description = tool.get("description", "No description")
        parameters = tool.get("parameters_json_schema", {})

        available_functions[name] = {
            "description": description,
            "parameters": parameters,
        }

        lines.append(f"{idx}. **{name}**: {description}")

    lines.extend([
        f"{len(function_tools) + 1}. **none**: Answer directly without calling a function",
        "",
        "## Instructions",
        "",
        "Read the user's request from `user_request.md` and determine:",
        "- If the request needs one of the functions above, select it",
        "- If the request can be answered directly without a function, select 'none'",
        "",
        "## Response Format",
        "",
        "Respond with ONLY a single line in this exact format:",
        "",
        "```",
        "CHOICE: <function_name>",
        "```",
        "",
        "or",
        "",
        "```",
        "CHOICE: none",
        "```",
        "",
        "Do not include any other text in your response.",
    ])

    return "\n".join(lines), available_functions


def parse_function_selection(response_text: str) -> str | None:
    """Parse function selection from Claude's response.

    Args:
        response_text: Claude's response text

    Returns:
        Selected function name, "none", or None if parsing fails
    """
    # Look for CHOICE: pattern
    match = re.search(r"CHOICE:\s*(\w+)", response_text, re.IGNORECASE)
    if match:
        selection = match.group(1).strip().lower()
        logger.debug("Parsed function selection: %s", selection)
        return selection

    # Fallback: try to find a function name directly
    logger.warning("Could not parse CHOICE from response: %s", response_text[:200])
    return None


def build_argument_collection_prompt(
    function_name: str,
    function_description: str,
    parameters_schema: dict[str, Any],
    temp_dir: str,
) -> str:
    """Build prompt for argument collection phase.

    Creates instructions for Claude to extract function arguments
    from the user's request and write them to the file structure.

    Args:
        function_name: Name of the selected function
        function_description: Description of the function
        parameters_schema: JSON schema for function parameters
        temp_dir: Temporary directory for output structure

    Returns:
        Prompt string for argument collection
    """
    # Use the structure converter to build instructions
    return build_structure_instructions(
        parameters_schema,
        temp_dir,
        tool_name=function_name,
        tool_description=function_description,
    )


def build_retry_prompt(
    original_prompt: str,
    schema: dict[str, Any],
    temp_dir: str,
    error_message: str,
) -> str:
    """Build retry prompt when argument validation fails.

    Args:
        original_prompt: The original argument collection prompt
        schema: Parameter schema
        temp_dir: New temporary directory for retry
        error_message: Error message from validation failure

    Returns:
        Retry prompt with error context
    """
    lines = [
        "# Retry: Argument Collection",
        "",
        f"**Previous attempt failed with error:** {error_message}",
        "",
        "Please try again, ensuring all required fields are properly filled out.",
        "",
        "---",
        "",
    ]

    # Re-build the structure instructions with new temp dir
    retry_instructions = build_structure_instructions(
        schema,
        temp_dir,
    )

    lines.append(retry_instructions)

    return "\n".join(lines)
