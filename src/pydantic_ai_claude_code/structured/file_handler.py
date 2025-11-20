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
    """
    Create a unique temporary directory path for structured output.
    
    Returns:
        A path string in /tmp of the form 'claude_data_structure_<8-hex-id>'.
    """
    unique_id = uuid.uuid4().hex[:8]
    return f"/tmp/claude_data_structure_{unique_id}"


def create_unstructured_output_path() -> str:
    """
    Create a filesystem path for a temporary unstructured output file in /tmp with a short unique identifier.
    
    Returns:
        file_path (str): Path to the temporary unstructured output file (e.g. `/tmp/claude_unstructured_output_<id>.txt`).
    """
    unique_id = uuid.uuid4().hex[:8]
    return f"/tmp/claude_unstructured_output_{unique_id}.txt"


def read_structured_output(
    schema: dict[str, Any],
    output_dir: str,
) -> dict[str, Any]:
    """
    Assemble structured data from files in a directory according to the provided schema.
    
    Parameters:
        schema (dict[str, Any]): Schema describing the expected file/directory structure and types.
        output_dir (str): Path to the directory containing files and subdirectories to read.
    
    Returns:
        dict[str, Any]: Assembled data dictionary matching the schema.
    
    Raises:
        RuntimeError: If the specified output directory does not exist.
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
    """
    Remove a file or directory tree at the given path.
    
    Parameters:
        output_path (str): Path to the file or directory to remove. If the path is a directory, its entire tree will be deleted.
    
    Notes:
        Exceptions raised during removal are caught and logged; this function does not propagate errors.
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