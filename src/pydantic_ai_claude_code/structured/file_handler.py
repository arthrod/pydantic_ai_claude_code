"""File handler for structured and unstructured outputs.

This module provides file handling functionality for structured and
unstructured outputs that the Claude Agent SDK doesn't have.
"""

import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any

from ..structure_converter import read_structure_from_filesystem

logger = logging.getLogger(__name__)


def create_structured_output_path() -> str:
    """Create a temporary directory path for structured output.

    Returns:
        Path to temporary directory for structured output
    """
    unique_id = uuid.uuid4().hex[:8]
    return f"/tmp/claude_data_structure_{unique_id}"


def create_unstructured_output_path() -> str:
    """Create a temporary file path for unstructured output.

    Returns:
        Path to temporary file for unstructured output
    """
    unique_id = uuid.uuid4().hex[:8]
    return f"/tmp/claude_unstructured_output_{unique_id}.txt"


def read_structured_output(
    schema: dict[str, Any],
    output_dir: str,
) -> dict[str, Any]:
    """Read structured output from filesystem structure.

    Args:
        schema: JSON schema defining the expected structure
        output_dir: Directory containing the file/folder structure

    Returns:
        Assembled data dictionary

    Raises:
        RuntimeError: If structure doesn't match schema
    """
    output_path = Path(output_dir)

    if not output_path.exists():
        raise RuntimeError(
            f"Structured output directory not found: {output_dir}\n"
            f"Claude should have created this directory with the expected structure."
        )

    return read_structure_from_filesystem(schema, output_path)


def read_unstructured_output(output_file: str) -> str:
    """Read unstructured output from file.

    Args:
        output_file: Path to the output file

    Returns:
        Content of the output file

    Raises:
        RuntimeError: If file doesn't exist
    """
    output_path = Path(output_file)

    if not output_path.exists():
        raise RuntimeError(
            f"Unstructured output file not found: {output_file}\n"
            f"Claude should have created this file with the output."
        )

    content = output_path.read_text()
    logger.debug("Read %d characters from unstructured output file", len(content))

    return content


def cleanup_output_file(output_path: str) -> None:
    """Clean up temporary output file or directory.

    Args:
        output_path: Path to clean up
    """
    path = Path(output_path)
    try:
        if path.is_dir():
            import shutil
            shutil.rmtree(path)
            logger.debug("Cleaned up output directory: %s", output_path)
        elif path.exists():
            path.unlink()
            logger.debug("Cleaned up output file: %s", output_path)
    except Exception as e:
        logger.warning("Failed to clean up output at %s: %s", output_path, e)
