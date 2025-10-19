"""Utilities for creating and handling model responses."""

import uuid
from typing import Any

from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models import ModelRequestParameters

from .types import ClaudeCodeSettings


def create_tool_call_part(tool_name: str, args: dict[str, Any]) -> ToolCallPart:
    """Create a ToolCallPart with auto-generated tool_call_id.

    Args:
        tool_name: Name of the tool being called
        args: Arguments for the tool call

    Returns:
        ToolCallPart with unique tool_call_id
    """
    return ToolCallPart(
        tool_name=tool_name,
        args=args,
        tool_call_id=f"call_{uuid.uuid4().hex[:16]}",
    )


def extract_model_parameters(
    model_request_parameters: ModelRequestParameters | None,
) -> tuple[list[Any], list[Any]]:
    """Extract output and function tools from request parameters.

    Args:
        model_request_parameters: Model request parameters, may be None

    Returns:
        Tuple of (output_tools, function_tools)
    """
    output_tools = (
        model_request_parameters.output_tools if model_request_parameters else []
    )
    function_tools = (
        model_request_parameters.function_tools if model_request_parameters else []
    )
    return output_tools, function_tools


def get_working_directory(settings: ClaudeCodeSettings, default: str = "/tmp") -> str:
    """Get working directory from settings with fallback.

    Args:
        settings: ClaudeCode settings
        default: Default directory if not found in settings

    Returns:
        Working directory path
    """
    return settings.get("__working_directory", default)
