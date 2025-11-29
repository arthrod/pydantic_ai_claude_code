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
    base_path.mkdir(parents=True, exist_ok=True)

    existing_numbers = [
        int(d.name)
        for d in base_path.iterdir()
        if d.is_dir() and d.name.isdigit()
    ]
    next_num = (max(existing_numbers) + 1) if existing_numbers else 1

    subdir = base_path / str(next_num)
    subdir.mkdir(parents=True, exist_ok=True)
    logger.debug("Created call subdirectory: %s", subdir)

    return subdir


def copy_additional_files(cwd: str, additional_files: dict[str, Path]) -> None:
    """
    Copy a set of source files into a target working directory, preserving file metadata.

    Parameters:
        cwd (str): Destination working directory path.
        additional_files (dict[str, Path]): Mapping from destination filename (may include subdirectories) to source Path.

    Raises:
        FileNotFoundError: If a source path does not exist.
        ValueError: If a source path exists but is not a regular file, or if destination escapes working directory.
    """
    dest_root = Path(cwd).resolve()

    for dest_name, source_path in additional_files.items():
        # Resolve relative paths from current working directory
        resolved_source = source_path.resolve()

        if not resolved_source.exists():
            raise FileNotFoundError(  # noqa: TRY003
                f"Additional file source not found: {source_path} "
                f"(resolved to {resolved_source})"
            )

        if not resolved_source.is_file():
            raise ValueError(  # noqa: TRY003
                f"Additional file source is not a file: {source_path} "
                f"(resolved to {resolved_source})"
            )

        # Create destination path (may include subdirectories)
        dest_path = dest_root / dest_name
        dest_resolved = dest_path.resolve()

        # Security check: ensure destination stays within working directory
        if dest_resolved != dest_root and dest_root not in dest_resolved.parents:
            raise ValueError(f"Destination path escapes working directory: {dest_name}")  # noqa: TRY003

        # Create parent directories if needed
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy file (binary mode, preserves permissions and timestamps)
        shutil.copy2(resolved_source, dest_path)

        logger.info("Copied additional file: %s -> %s", source_path, dest_path)