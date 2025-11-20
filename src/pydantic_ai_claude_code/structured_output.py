"""
Structured Output Handler - File-based JSON structured output.

This module provides the unique file-based structured output functionality
that the Claude Agent SDK doesn't provide. It uses directory structures
instead of direct JSON generation for more reliable output.
"""

from typing import Any
import logging

from .structure_converter import (
    build_structure_instructions,
    read_structure_from_filesystem,
)
from .structured.file_handler import (
    create_structured_output_path,
    read_structured_output,
)

logger = logging.getLogger(__name__)


class StructuredOutputHandler:
    """
    Handler for structured output using file-based JSON assembly.

    This handler uses our unique file/folder structure approach to
    generate structured outputs more reliably than direct JSON generation.
    """

    def __init__(self):
        self._output_dir: str | None = None

    async def execute(
        self,
        prompt: str,
        schema: dict[str, Any],
        sdk_options: Any,
    ) -> dict[str, Any]:
        """
        Execute a request with structured output.

        Args:
            prompt: The user prompt
            schema: JSON schema for output
            sdk_options: SDK options for execution

        Returns:
            Assembled structured output
        """
        from .utils import run_claude_async

        # Create output directory
        self._output_dir = create_structured_output_path()

        # Build instructions for file-based output
        instructions = build_structure_instructions(
            schema,
            self._output_dir,
            tool_name=None,
            tool_description=None,
        )

        # Combine with user prompt
        full_prompt = f"{instructions}\n\n---\n\n{prompt}"

        # Convert SDK options to settings format
        settings = self._sdk_options_to_settings(sdk_options)

        # Execute
        try:
            await run_claude_async(full_prompt, settings=settings)

            # Read assembled result
            from pathlib import Path
            result = read_structure_from_filesystem(
                schema,
                Path(self._output_dir),
            )

            return result

        finally:
            # Cleanup could be added here
            pass

    def _sdk_options_to_settings(self, sdk_options: Any) -> dict[str, Any]:
        """Convert SDK options to settings dict."""
        settings = {}

        if sdk_options is None:
            return settings

        # Map common fields
        if hasattr(sdk_options, "cwd"):
            settings["working_directory"] = sdk_options.cwd
        if hasattr(sdk_options, "allowed_tools"):
            settings["allowed_tools"] = sdk_options.allowed_tools
        if hasattr(sdk_options, "disallowed_tools"):
            settings["disallowed_tools"] = sdk_options.disallowed_tools
        if hasattr(sdk_options, "permission_mode"):
            settings["permission_mode"] = sdk_options.permission_mode
        if hasattr(sdk_options, "model"):
            settings["model"] = sdk_options.model

        return settings

    def get_output_directory(self) -> str | None:
        """Get the current output directory path."""
        return self._output_dir
