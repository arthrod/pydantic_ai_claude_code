"""File utilities for Claude Code model.

This module provides file system utilities for managing working directories,
copying additional files, and other file operations.
"""

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def get_next_call_subdirectory(base_dir: str) -> Path:
    """Get next numbered subdirectory for this CLI call to avoid overwrites.

    Args:
        base_dir: Base working directory

    Returns:
        Path to numbered subdirectory (e.g., base_dir/1/, base_dir/2/, etc.)
    """
    base_path = Path(base_dir)
    existing_subdirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.isdigit()]
    next_num = len(existing_subdirs) + 1

    subdir = base_path / str(next_num)
    subdir.mkdir(parents=True, exist_ok=True)
    logger.debug("Created call subdirectory: %s", subdir)

    return subdir


def copy_additional_files(cwd: str, additional_files: dict[str, Path]) -> None:
    """
    Copy files from the provided source Paths into the working directory under the specified destination names.
    
    Parameters:
        cwd (str): Destination working directory; destination paths are created relative to this directory.
        additional_files (dict[str, Path]): Mapping from destination filename (may include subdirectories) to the source Path to copy.
    
    Raises:
        FileNotFoundError: If a source path does not exist.
        ValueError: If a source path exists but is not a file.
    """
    for dest_name, source_path in additional_files.items():
        # Resolve relative paths from current working directory
        resolved_source = source_path.resolve()

        if not resolved_source.exists():
            raise FileNotFoundError(
                f"Additional file source not found: {source_path} "
                f"(resolved to {resolved_source})"
            )

        if not resolved_source.is_file():
            raise ValueError(
                f"Additional file source is not a file: {source_path} "
                f"(resolved to {resolved_source})"
            )

        # Create destination path (may include subdirectories)
        dest_path = Path(cwd) / dest_name

        # Create parent directories if needed
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy file (binary mode, preserves permissions and timestamps)
        shutil.copy2(resolved_source, dest_path)

        logger.info("Copied additional file: %s -> %s", source_path, dest_path)