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
    Create a user-facing prompt that lists available functions and asks Claude to choose one or 'none'.
    
    Parameters:
        function_tools (list[dict[str, Any]]): List of function tool definitions; each item may include keys like "name", "description", and "parameters_json_schema".
    
    Returns:
        tuple[str, dict[str, dict[str, Any]]]: A tuple where the first element is the composed prompt string and the second is an `available_functions` mapping from function name to a dict containing its "description" and "parameters".
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
        response_text (str): Raw text returned by Claude containing a single-line choice in the form `CHOICE: <name>`.
    
    Returns:
        str | None: The selected function name in lowercase, the literal string `'none'` if "none" was chosen, or `None` if no valid `CHOICE:` line can be parsed.
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
    Create a prompt that instructs the model to extract the selected function's arguments from the user's request and write them into a structured set of files in the given temporary directory.
    
    Parameters:
        function_name (str): The selected function's name to include in the instructions.
        function_description (str): Short description of the function to provide context for argument extraction.
        parameters_schema (dict[str, Any]): JSON Schema describing the expected parameters and their types.
        temp_dir (str): Path to the temporary directory where the structured output files should be written.
    
    Returns:
        prompt (str): A prompt string directing the model how to extract arguments and produce the structured output.
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
    Construct a retry prompt that instructs Claude to re-attempt argument collection after a validation error.
    
    Parameters:
        original_prompt (str): The original argument collection prompt that preceded the error.
        schema (dict[str, Any]): JSON schema describing the function's parameters to validate and collect.
        temp_dir (str): Path to a temporary directory where the structured output should be written for the retry.
        error_message (str): Validation error message to present to the model.
    
    Returns:
        str: A prompt string containing the error context and updated structure instructions for the retry.
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