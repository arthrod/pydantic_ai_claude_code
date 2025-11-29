"""Structured output modules for Claude Code model.

This package contains functionality for handling structured output that
the Claude Agent SDK doesn't provide, including file-based JSON output
and two-phase function selection.
"""

from .converter import (
    write_structure_to_filesystem,
    read_structure_from_filesystem,
    build_structure_instructions,
)
from .file_handler import (
    create_structured_output_path,
    create_unstructured_output_path,
    read_structured_output,
    read_unstructured_output,
)
from .function_selector import (
    build_function_selection_prompt,
    parse_function_selection,
    build_argument_collection_prompt,
)

__all__ = [
    # Converter
    "write_structure_to_filesystem",
    "read_structure_from_filesystem",
    "build_structure_instructions",
    # File handler
    "create_structured_output_path",
    "create_unstructured_output_path",
    "read_structured_output",
    "read_unstructured_output",
    # Function selector
    "build_function_selection_prompt",
    "parse_function_selection",
    "build_argument_collection_prompt",
]
