"""Message conversion utilities for Claude Code model."""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic_ai.messages import (
    BinaryContent,
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


def _create_binary_content_file(
    binary_content: BinaryContent, counter: int, working_dir: str
) -> str:
    """Create file for binary content in working directory and return prompt reference.

    Args:
        binary_content: BinaryContent object with data and metadata
        counter: Sequential counter for unique filenames
        working_dir: Working directory where file should be created

    Returns:
        Prompt reference string mentioning the file
    """
    # Get file extension from media type (e.g., "image/png" -> "png")
    try:
        extension = binary_content.format
    except ValueError:
        # If format lookup fails, extract from media_type
        extension = binary_content.media_type.split('/')[-1].split(';')[0]

    # Use identifier if provided, otherwise use counter
    base_name = binary_content.identifier or f"file_{counter}"
    # Sanitize identifier to be filename-safe
    safe_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in base_name)
    filename = f"{safe_name}.{extension}"
    file_path = Path(working_dir) / filename

    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Write binary data to file
    file_path.write_bytes(binary_content.data)

    # Use Claude Code CLI's @ syntax to reference the file
    prompt_reference = f"@{filename}"

    logger.debug(
        "Wrote binary content (%s) to %s (%d bytes)",
        binary_content.media_type,
        file_path,
        len(binary_content.data)
    )

    return prompt_reference


def _create_tool_result_file(
    tool_name: str, content: str, counter: int, working_dir: str
) -> str:
    """Create file for tool result in working directory and return prompt reference.

    Args:
        tool_name: Name of the tool that produced the result
        content: Tool result content
        counter: Sequential counter for unique filenames
        working_dir: Working directory where file should be created

    Returns:
        Prompt reference string mentioning the file
    """
    filename = f"tool_result_{counter}_{tool_name}.txt"
    file_path = Path(working_dir) / filename

    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Create file directly in working directory
    file_path.write_text(str(content))

    prompt_reference = (
        f"Additional Information: The results from the {tool_name} tool "
        f"are available in the file {filename}. Read this file to see the information."
    )

    logger.debug(
        "Wrote tool result from %s to %s (%d bytes)",
        tool_name,
        file_path,
        len(str(content))
    )

    return prompt_reference


def _process_user_prompt_part(
    req_part: UserPromptPart,
    binary_content_counter: int,
    working_dir: str,
) -> tuple[str, int]:
    """Process UserPromptPart into prompt string.

    Args:
        req_part: UserPromptPart to process
        binary_content_counter: Current binary content counter
        working_dir: Working directory for file creation

    Returns:
        Tuple of (prompt_string, updated_binary_counter)
    """
    if isinstance(req_part.content, str):
        return f"Request: {req_part.content}", binary_content_counter

    # content is a Sequence that may contain strings and BinaryContent
    prompt_parts = []
    for content_item in req_part.content:
        if isinstance(content_item, str):
            prompt_parts.append(content_item)
        elif isinstance(content_item, BinaryContent):
            binary_content_counter += 1
            file_ref = _create_binary_content_file(
                content_item, binary_content_counter, working_dir
            )
            prompt_parts.append(file_ref)

    return f"Request: {' '.join(prompt_parts)}", binary_content_counter


def _process_tool_return_part(
    req_part: ToolReturnPart,
    next_part: ModelRequestPart | None,
    tool_result_counter: int,
    binary_content_counter: int,
    working_dir: str,
) -> tuple[str | None, int, int]:
    """Process ToolReturnPart into prompt string.

    Args:
        req_part: ToolReturnPart to process
        next_part: Next part in sequence (for lookahead)
        tool_result_counter: Current tool result counter
        binary_content_counter: Current binary content counter
        working_dir: Working directory for file creation

    Returns:
        Tuple of (prompt_string or None, updated_tool_counter, updated_binary_counter)
    """
    logger.debug(
        "Processing ToolReturnPart: tool_name=%s, content_type=%s, content=%s",
        req_part.tool_name,
        type(req_part.content).__name__,
        str(req_part.content)[:100] if req_part.content else None
    )

    # Check if tool returned BinaryContent
    if isinstance(req_part.content, BinaryContent):
        binary_content_counter += 1
        file_ref = _create_binary_content_file(
            req_part.content, binary_content_counter, working_dir
        )
        prompt = f"Additional Information: The {req_part.tool_name} tool returned: {file_ref}"
        return prompt, tool_result_counter, binary_content_counter

    # Look ahead for UserPromptPart with BinaryContent
    has_binary_next = False
    if next_part and isinstance(next_part, UserPromptPart) and not isinstance(next_part.content, str):
        for content_item in next_part.content:
            if isinstance(content_item, BinaryContent):
                has_binary_next = True
                break

    if has_binary_next:
        logger.debug("Skipping ToolReturnPart text - binary content follows in next UserPromptPart")
        return None, tool_result_counter, binary_content_counter

    # Regular text/data result
    tool_result_counter += 1
    prompt_ref = _create_tool_result_file(
        req_part.tool_name, str(req_part.content), tool_result_counter, working_dir
    )
    return prompt_ref, tool_result_counter, binary_content_counter


def _process_request_parts(
    req_parts: Sequence[ModelRequestPart],
    skip_system_prompt: bool,
    tool_result_counter: int,
    binary_content_counter: int,
    working_dir: str,
) -> tuple[list[str], int, int]:
    """Process ModelRequest parts into prompt strings and files.

    Args:
        req_parts: Request message parts to process
        skip_system_prompt: Whether to skip system prompt parts
        tool_result_counter: Current tool result counter
        binary_content_counter: Current binary content counter
        working_dir: Working directory where files should be created

    Returns:
        Tuple of (prompt_parts, updated_tool_counter, updated_binary_counter)
    """
    parts: list[str] = []
    parts_list = list(req_parts)

    logger.debug("Processing %d request parts", len(parts_list))

    for i, req_part in enumerate(parts_list):
        logger.debug("  Part %d: type=%s", i, type(req_part).__name__)

        if isinstance(req_part, SystemPromptPart):
            if not skip_system_prompt:
                parts.insert(0, f"System: {req_part.content}")

        elif isinstance(req_part, UserPromptPart):
            prompt, binary_content_counter = _process_user_prompt_part(
                req_part, binary_content_counter, working_dir
            )
            parts.append(prompt)

        elif isinstance(req_part, ToolReturnPart):
            next_part = parts_list[i + 1] if i + 1 < len(parts_list) else None
            maybe_prompt, tool_result_counter, binary_content_counter = _process_tool_return_part(
                req_part, next_part, tool_result_counter, binary_content_counter, working_dir
            )
            if maybe_prompt is not None:
                parts.append(maybe_prompt)

    return parts, tool_result_counter, binary_content_counter


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
    messages: list[ModelMessage], *, skip_system_prompt: bool = False, working_dir: str
) -> str:
    """Convert Pydantic AI messages to a prompt string for Claude CLI.

    Tool results and binary content are written directly to files in the working
    directory and referenced in the prompt.

    Args:
        messages: List of Pydantic AI messages
        skip_system_prompt: If True, skip SystemPromptPart from messages (used when we
        have tool results)
        working_dir: Working directory where files should be created

    Returns:
        Formatted prompt string with file references for tool results and binary content
    """
    logger.debug(
        "Formatting %d messages for Claude CLI (skip_system_prompt=%s, working_dir=%s)",
        len(messages),
        skip_system_prompt,
        working_dir,
    )

    all_parts: list[str] = []
    tool_result_counter = 0
    binary_content_counter = 0

    logger.debug("Processing %d messages", len(messages))
    for msg_idx, message in enumerate(messages):
        logger.debug("Message %d: type=%s", msg_idx, type(message).__name__)
        if isinstance(message, ModelRequest):
            msg_parts, tool_result_counter, binary_content_counter = _process_request_parts(
                message.parts, skip_system_prompt, tool_result_counter, binary_content_counter, working_dir
            )
            all_parts.extend(msg_parts)
        elif isinstance(message, ModelResponse):
            msg_parts = _process_response_parts(message.parts)
            all_parts.extend(msg_parts)

    formatted_prompt = "\n\n".join(all_parts)
    logger.debug(
        "Formatted prompt: %d parts, %d total chars (%d binary files, %d tool results)",
        len(all_parts),
        len(formatted_prompt),
        binary_content_counter,
        tool_result_counter,
    )

    return formatted_prompt


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
