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
    """
    Constructs a prompt that asks the model to choose one of the available functions or to respond directly.
    
    Builds a human-readable prompt enumerating each provided tool and appends strict instructions requiring the model to reply with a single-line choice "CHOICE: <function_name>" or "CHOICE: none". The function also returns a mapping of available functions to their descriptions and parameter schemas for downstream validation.
    
    Args:
        function_tools: List of tool definitions. Each tool is expected to be a dict and may include:
            - "name" (str): optional function name; a default name will be generated if missing.
            - "description" (str): optional human-readable description.
            - "parameters_json_schema" (dict): optional JSON Schema describing the function's parameters.
    
    Returns:
        tuple: A pair (prompt_string, available_functions) where:
            - prompt_string (str): The complete prompt text to present to the model.
            - available_functions (dict): Mapping from function name to a dict with keys:
                - "description" (str): the tool's description.
                - "parameters" (dict): the tool's parameters JSON Schema (may be empty).
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
    """
    Extract the selected function name from Claude's response.
    
    Parameters:
        response_text (str): Raw text returned by Claude containing a single line like "CHOICE: <function_name>" or "CHOICE: none".
    
    Returns:
        str | None: Lowercase function name selected, the string "none" if no function should be called, or `None` if no valid CHOICE line could be parsed.
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
    """
    Builds a prompt instructing Claude to extract the selected function's arguments and write them into a structured output under the given temporary directory.
    
    Parameters:
        function_name (str): Name of the selected function.
        function_description (str): Short description of the selected function.
        parameters_schema (dict[str, Any]): JSON Schema describing expected function parameters.
        temp_dir (str): Path to the temporary directory where the model should write the structured output.
    
    Returns:
        str: Prompt string that directs the model to produce and store arguments conforming to `parameters_schema` in `temp_dir`.
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
    """
    Build a retry prompt that requests corrected function arguments after a validation error.
    
    Includes the provided validation error message and regenerates the structured argument-collection instructions using the given schema and a new temporary directory.
    
    Parameters:
        original_prompt (str): The original argument collection prompt (kept for context).
        schema (dict[str, Any]): JSON schema describing the function parameters to collect.
        temp_dir (str): Path to a new temporary directory where the structured output should be written.
        error_message (str): Validation error message explaining why the previous attempt failed.
    
    Returns:
        str: A complete retry prompt containing the error context and regenerated structure instructions.
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