"""Message conversion utilities for Claude Code model."""

import logging
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

logger = logging.getLogger(__name__)


def format_messages_for_claude(messages: list[ModelMessage]) -> str:
    """Convert Pydantic AI messages to a prompt string for Claude CLI.

    Args:
        messages: List of Pydantic AI messages

    Returns:
        Formatted prompt string
    """
    logger.debug("Formatting %d messages for Claude CLI", len(messages))

    parts: list[str] = []

    for message in messages:
        if isinstance(message, ModelRequest):
            for req_part in message.parts:
                if isinstance(req_part, SystemPromptPart):
                    # System prompts are prepended
                    parts.insert(0, f"System: {req_part.content}")
                elif isinstance(req_part, UserPromptPart):
                    parts.append(f"User: {req_part.content}")
                elif isinstance(req_part, ToolReturnPart):
                    # Format tool returns as plain contextual data (no mention of tools/functions)
                    parts.append(f"Context: {req_part.content}")

        elif isinstance(message, ModelResponse):
            for resp_part in message.parts:
                if isinstance(resp_part, TextPart):
                    parts.append(f"Assistant: {resp_part.content}")
                elif isinstance(resp_part, ToolCallPart):
                    # Skip tool calls entirely - don't show Claude it requested functions
                    # Only show tool RESULTS (ToolReturnPart) in the conversation
                    pass

    formatted_prompt = "\n\n".join(parts)
    logger.debug(
        "Formatted prompt: %d parts, %d total chars", len(parts), len(formatted_prompt)
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
            for req_part in message.parts:
                if isinstance(req_part, SystemPromptPart):
                    context["has_system_prompt"] = True
                elif isinstance(req_part, UserPromptPart):
                    context["num_user_messages"] += 1
                elif isinstance(req_part, ToolReturnPart):
                    context["num_tool_returns"] += 1

        elif isinstance(message, ModelResponse):
            for resp_part in message.parts:
                if isinstance(resp_part, TextPart):
                    context["num_assistant_messages"] += 1
                elif isinstance(resp_part, ToolCallPart):
                    context["num_tool_calls"] += 1

    return context
