"""Utilities for generating temporary file and directory paths."""

import uuid
from pathlib import Path


def generate_output_file_path(working_dir: str, prefix: str, extension: str) -> str:
    """Generate a unique temporary file path.

    Args:
        working_dir: Directory where the file should be created
        prefix: Prefix for the filename (e.g., 'claude_structured_output')
        extension: File extension including the dot (e.g., '.json', '.txt')

    Returns:
        Absolute path to the unique temporary file
    """
    unique_id = uuid.uuid4().hex
    return str(Path(working_dir) / f"{prefix}_{unique_id}{extension}")


def generate_temp_directory_path(working_dir: str, prefix: str, short_id: bool = True) -> str:
    """Generate a unique temporary directory path.

    Args:
        working_dir: Parent directory where the temp directory should be created
        prefix: Prefix for the directory name (e.g., 'claude_data_structure')
        short_id: If True, use 8-character ID; otherwise use full UUID hex

    Returns:
        Absolute path to the unique temporary directory
    """
    unique_id = uuid.uuid4().hex[:8] if short_id else uuid.uuid4().hex
    return str(Path(working_dir) / f"{prefix}_{unique_id}")
