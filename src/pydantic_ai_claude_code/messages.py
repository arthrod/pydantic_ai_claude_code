"""Message conversion utilities for Claude Code model."""

import logging
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

logger = logging.getLogger(__name__)


def _create_tool_result_file(
    tool_name: str, content: str, counter: int
) -> tuple[str, Path, str]:
    """Create temporary file for tool result and return file info.

    Args:
        tool_name: Name of the tool that produced the result
        content: Tool result content
        counter: Sequential counter for unique filenames

    Returns:
        Tuple of (filename, temp_file_path, prompt_reference)
    """
    filename = f"tool_result_{counter}_{tool_name}.txt"

    # Create temp file with tool result content
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix=".txt",
        prefix=f"tool_result_{tool_name}_",
        delete=False
    ) as tf:
        tf.write(str(content))
        temp_file = Path(tf.name)

    prompt_reference = (
        f"Additional Information: The results from the {tool_name} tool "
        f"are available in the file {filename}. Read this file to see the information."
    )

    logger.debug(
        "Wrote tool result from %s to temp file %s (%d bytes)",
        tool_name,
        temp_file,
        len(str(content))
    )

    return filename, temp_file, prompt_reference


def _process_request_parts(
    req_parts: Sequence[ModelRequestPart],
    skip_system_prompt: bool,
    tool_result_counter: int,
) -> tuple[list[str], dict[str, Path], int]:
    """Process ModelRequest parts into prompt strings and files.

    Args:
        req_parts: Request message parts to process
        skip_system_prompt: Whether to skip system prompt parts
        tool_result_counter: Current tool result counter

    Returns:
        Tuple of (prompt_parts, additional_files, updated_counter)
    """
    parts: list[str] = []
    additional_files: dict[str, Path] = {}

    for req_part in req_parts:
        if isinstance(req_part, SystemPromptPart):
            if not skip_system_prompt:
                parts.insert(0, f"System: {req_part.content}")
        elif isinstance(req_part, UserPromptPart):
            parts.append(f"Request: {req_part.content}")
        elif isinstance(req_part, ToolReturnPart):
            tool_result_counter += 1
            filename, temp_file, prompt_ref = _create_tool_result_file(
                req_part.tool_name, str(req_part.content), tool_result_counter
            )
            additional_files[filename] = temp_file
            parts.append(prompt_ref)

    return parts, additional_files, tool_result_counter


def _process_response_parts(resp_parts: Sequence[ModelResponsePart]) -> list[str]:
    """Process ModelResponse parts into prompt strings.

    Args:
        resp_parts: Response message parts to process

    Returns:
        List of prompt strings
    """
    parts: list[str] = []

    for resp_part in resp_parts:
        if isinstance(resp_part, TextPart):
            parts.append(f"Assistant: {resp_part.content}")
        # ToolCallPart is intentionally skipped

    return parts


def format_messages_for_claude(
    messages: list[ModelMessage], *, skip_system_prompt: bool = False
) -> tuple[str, dict[str, Path]]:
    """Convert Pydantic AI messages to a prompt string for Claude CLI.

    Tool results are written to temporary files and returned as additional_files
    rather than being embedded in the prompt. This follows the same pattern as
    the additional_files feature.

    Args:
        messages: List of Pydantic AI messages
        skip_system_prompt: If True, skip SystemPromptPart from messages (used when we have tool results)

    Returns:
        Tuple of (formatted_prompt, additional_files_dict)
        - formatted_prompt: The prompt string with file references
        - additional_files_dict: Dict mapping destination filename -> temp file Path
    """
    logger.debug(
        "Formatting %d messages for Claude CLI (skip_system_prompt=%s)",
        len(messages),
        skip_system_prompt,
    )

    all_parts: list[str] = []
    all_files: dict[str, Path] = {}
    tool_result_counter = 0

    for message in messages:
        if isinstance(message, ModelRequest):
            msg_parts, msg_files, tool_result_counter = _process_request_parts(
                message.parts, skip_system_prompt, tool_result_counter
            )
            all_parts.extend(msg_parts)
            all_files.update(msg_files)
        elif isinstance(message, ModelResponse):
            msg_parts = _process_response_parts(message.parts)
            all_parts.extend(msg_parts)

    formatted_prompt = "\n\n".join(all_parts)
    logger.debug(
        "Formatted prompt: %d parts, %d total chars, %d tool result files",
        len(all_parts),
        len(formatted_prompt),
        len(all_files)
    )

    return formatted_prompt, all_files


def extract_text_from_response(response_text: str) -> str:
    """Extract the main text response from Claude's output.

    Args:
        response_text: Raw response text from Claude CLI

    Returns:
        Cleaned response text
    """
    # Remove any "Assistant:" prefix if present
    if response_text.startswith("Assistant: "):
        return response_text[len("Assistant: ") :]

    return response_text


def _count_request_parts(req_parts: Sequence[ModelRequestPart]) -> dict[str, Any]:
    """Count different types of request parts.

    Args:
        req_parts: Request message parts

    Returns:
        Dictionary with counts and flags
    """
    counts = {
        "has_system_prompt": False,
        "num_user_messages": 0,
        "num_tool_returns": 0,
    }

    for req_part in req_parts:
        if isinstance(req_part, SystemPromptPart):
            counts["has_system_prompt"] = True
        elif isinstance(req_part, UserPromptPart):
            counts["num_user_messages"] += 1
        elif isinstance(req_part, ToolReturnPart):
            counts["num_tool_returns"] += 1

    return counts


def _count_response_parts(resp_parts: Sequence[ModelResponsePart]) -> dict[str, int]:
    """Count different types of response parts.

    Args:
        resp_parts: Response message parts

    Returns:
        Dictionary with counts
    """
    counts = {
        "num_assistant_messages": 0,
        "num_tool_calls": 0,
    }

    for resp_part in resp_parts:
        if isinstance(resp_part, TextPart):
            counts["num_assistant_messages"] += 1
        elif isinstance(resp_part, ToolCallPart):
            counts["num_tool_calls"] += 1

    return counts


def build_conversation_context(messages: list[ModelMessage]) -> dict[str, Any]:
    """Build conversation context from message history.

    Args:
        messages: List of Pydantic AI messages

    Returns:
        Dictionary containing conversation metadata
    """
    context = {
        "num_messages": len(messages),
        "has_system_prompt": False,
        "num_user_messages": 0,
        "num_assistant_messages": 0,
        "num_tool_calls": 0,
        "num_tool_returns": 0,
    }

    for message in messages:
        if isinstance(message, ModelRequest):
            req_counts = _count_request_parts(message.parts)
            context["has_system_prompt"] |= req_counts["has_system_prompt"]
            context["num_user_messages"] += req_counts["num_user_messages"]
            context["num_tool_returns"] += req_counts["num_tool_returns"]
        elif isinstance(message, ModelResponse):
            resp_counts = _count_response_parts(message.parts)
            context["num_assistant_messages"] += resp_counts["num_assistant_messages"]
            context["num_tool_calls"] += resp_counts["num_tool_calls"]

    return context
